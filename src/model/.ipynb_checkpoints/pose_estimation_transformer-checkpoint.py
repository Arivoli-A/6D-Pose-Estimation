# pose estimation transformer.py : This file defines the architecture of pose estimation transformer


import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor, nested_tensor_from_tensor_list
# from .backbone import build_backbone
from utils.matcher import HungarianMatcher
from .deformable_transformer import create_layers
from .position_encoding import PositionEncodingSine, BoundingBoxEncodingSine
import copy

GROUP_NORM = 32
BACKBONE_TYPE = 'output'
BBOX_MODE = 'backbone'
LOSS = 'split'
NN_LAYER = 3
DTYPE = torch.float32
DTYPE_INT = torch.int32

class feed_fwd_nn(nn.Module): # Feed Forward NN for pose head
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(self.num_layers):
            layers+=self.block(i)
            
        self.network = nn.Sequential(*layers)
        
    def block(self,n_layer):
        layer = []
        
        if n_layer == 0:
            layer.append(nn.Linear(self.input_dim, self.hidden_dim))
            layer.append(nn.ReLU())
            
        elif n_layer == self.num_layers-1:
            layer.append(nn.Linear(self.hidden_dim, self.output_dim))
            
        else:
            layer.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layer.append(nn.ReLU())
        
        return layer
        
    def forward(self, x):
        return self.network(x)
    
class pose_estimation(nn.Module):
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, n_classes, backbone_type = BACKBONE_TYPE, bbox_mode = BBOX_MODE, loss = LOSS, n_nn_layer = NN_LAYER):
        # bbox_mode : 'gt', 'backbone', 'jitter'
        # loss : 'pose', 'rot' (6D) 
        # ref_point, query - given by backbone (no learned embedding)
        # output : Each output for each class ('specific')
        # No loss from each layer of FNN
        # backbone_type = 'input', 'output', 'none'

        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.n_queries = num_queries
        self.n_classes = n_classes + 1  # +1 for dummy/background class
        self.bbox_mode = bbox_mode
        self.loss = loss
        self.n_nn_layer = n_nn_layer
        self.num_feature_levels = num_feature_levels

        num_pred = transformer.decoder.num_layers
        
        if self.loss == 'split':
            self.t_dim = 3
            self.rot_dim = 6
            self.translation_head = feed_fwd_nn(hidden_dim, hidden_dim, self.t_dim * self.n_classes, self.n_nn_layer)
            self.rotation_head = feed_fwd_nn(hidden_dim, hidden_dim, self.rot_dim * self.n_classes, self.n_nn_layer)
            # Pose is predicted for each intermediate decoder layer for training with auxiliary losses
            self.translation_head = create_layers(self.translation_head, num_pred)
            self.rotation_head = create_layers(self.rotation_head, num_pred)
            
        elif self.loss == 'pose':
            self.pose_dim = 9
            self.pose_head = feed_fwd_nn(hidden_dim, hidden_dim, self.pose_dim * self.n_classes, self.n_nn_layer)
            # Pose is predicted for each intermediate decoder layer for training with auxiliary losses
            self.pose_head = create_layers(self.pose_head, num_pred)

        else:
            print(self.loss,' not implemented')

        input_proj_list = []
        
        if num_feature_levels > 1: # For encoder input : for backbone_type : 'input', 'output', 'none'
            
            # Use multi-scale features as input to the transformer
            num_backbone_out = len(backbone.strides)
            
            # If multi-scale then every intermediate backbone feature map is returned
            for n in range(num_backbone_out):
                in_channels = backbone.num_channels[n]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(GROUP_NORM, self.hidden_dim),
                ))
                
            # If more feature levels are required than backbone feature maps are available then the last feature map is
            # passed through an additional 3x3 Conv layer to create a new feature map.
            # This new feature map is then used as the baseline for the next feature map to calculate
            
            for n in range(num_feature_levels - num_backbone_out):  #num_feature_levels - num_backbone_out >=0
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(GROUP_NORM, self.hidden_dim),
                ))
                in_channels = hidden_dim
            
        else:
            #backbone's last feature map.
            input_proj_list.append(nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(GROUP_NORM, self.hidden_dim),
                ))
            
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            
        self.bbox_encoding = BoundingBoxEncodingSine(num_pos_feats=self.hidden_dim / 8)
        
    def forward(self, sample_data, target_data = None):
        # if backbone_type is 'input' : sample_data:NestedTensor : object.tensors, object.mask
        # if backbone_type is 'output' : sample_data:tuple : (features_list, pos_list, predictions_list)
        # if backbone_type is 'none' : sample_data:ndarray : images

        # target_data : dict
        # "boxes": tensor of size [n_obj, 4], contains the bounding box (x_c, y_c, w, h) of each object in each image
        #normalized to image size
        # "labels": tensor of size [n_obj, ], contains the label of each object in the image
        # "image_id": tensor of size [1],  contains the image id to which this annotation belongs to
        # "gt_poses": tensor of size [n_obj, 4,4 ], contains the relative pose for each object present
        # in the image w.r.t the camera.
        

        if self.backbone_type == 'input':
            image_size_data = [[sample.shape[-2], sample.shape[-1]] for sample in sample_data.tensors]
            features, pos, pred_objects = self.backbone(sample_data)
        elif self.backbone_type == 'output':
            features, pos, pred_objects = sample_data 
            
        # features : Feature outputs from backbone
        # pos : Position Encoding of features    
        # pred_objects: Bounding box predictions : [cx,cy,h,w], normalized : 0:3 - bounding box, 4 - confidences, 5 - labels
        
        pred_box_data = []
        pred_class_data = []
        query_embed_data = []
        n_boxes_per_sample_data = []
        
        
        if self.bbox_mode == 'backbone':
            # Iterate over batch and prepare each image in batch
            for bs, predictions in enumerate(pred_objects):
                # predictions : 0:3 - bounding box, 4 - confidences, 5 - labels 
                if predictions is None:
                    # Case: Backbone has not predicted anything for image add dummy boxes and mark that nothing has been predicted
                    n_boxes = 0
                    n_boxes_per_sample.append(n_boxes)
                    backbone_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                  dtype=DTYPE, device=features[0].decompose()[0].device)
                    query_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                               dtype=DTYPE, device=features[0].decompose()[0].device)
                    query_embed = query_embed.repeat(1, self.hidden_dim * 2)
                    backbone_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)], dtype=DTYPE_INT,
                                                    device=features[0].decompose()[0].device)
                else:
                    # Case: Backbone predicted something
                    backbone_boxes = predictions[:, :4]
                    n_boxes = len(backbone_boxes)

                    # Predicted classes by backbone // class 0 is "background"
                    # Scores predicted by the backbone are needed for top-k selection
                    backbone_scores = predictions[:, 4]
                    backbone_classes = predictions[:, 5].type(DTYPE_INT)

                    # For the current number of boxes determine the query encoding
                    query_embed = self.bbox_encoding(backbone_boxes)
                    
                    # encoding for query and key for attention
                    query_embed = query_embed.repeat(1, 2)

                    if n_boxes < self.n_queries:
                        # Fill up with dummy boxes to match the query size and add dummy encodings
                        dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries - n_boxes)],
                                                   dtype=DTYPE, device=backbone_boxes.device)
                        dummy_embed = torch.tensor([[-10] for i in range(self.n_queries - n_boxes)],
                                                   dtype=DTYPE, device=backbone_boxes.device)
                        dummy_embed = dummy_embed.repeat(1, self.hidden_dim * 2)
                        backbone_boxes = torch.cat([backbone_boxes, dummy_boxes], dim=0)
                        query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                        dummy_classes = torch.tensor([-1 for i in range(self.n_queries - n_boxes)],
                                                     dtype=DTYPE_INT, device=backbone_boxes.device)
                        backbone_classes = torch.cat([backbone_classes, dummy_classes], dim=0)
                        
                    elif n_boxes > self.n_queries:
                        # Number of boxes will be limited to the number of queries
                        n_boxes = self.n_queries
                        # Case: backbone predicts more output objects than queries available --> take top n_queries
                        # Sort scores to get the post top performing ones
                        backbone_scores, indices = torch.sort(backbone_scores, dim=0, descending=True)
                        backbone_classes = backbone_classes[indices]
                        backbone_boxes = backbone_boxes[indices, :]
                        query_embed = query_embed[indices, :]

                        # Take the top n predictions
                        backbone_scores = backbone_scores[:self.n_queries]
                        backbone_classes = backbone_classes[:self.n_queries]
                        backbone_boxes = backbone_boxes[:self.n_queries]
                        query_embed = query_embed[:self.n_queries]
                        
                    n_boxes_per_sample.append(n_boxes)
                    
                pred_box_data.append(backbone_boxes)
                pred_class_data.append(backbone_classes)
                query_embed_data.append(query_embed)
        
        elif self.bbox_mode in ['gt', 'jitter'] and target_data is not None:
            for t, target in enumerate(target_data):
                # GT : normalized cx, cy, w, h
                if self.bbox_mode == 'gt':
                    t_boxes = target["boxes"]
                elif self.bbox_mode == 'jitter':
                    t_boxes = target["jitter_boxes"]
                    
                n_boxes = len(t_boxes)
                n_boxes_per_sample.append(n_boxes)

                # Add classes
                t_classes = target["labels"]

                #query encoding
                query_embed = self.bbox_encoding(t_boxes) 
                
                # query and key data attention
                query_embed = query_embed.repeat(1, 2)

                # We always predict a fixed number of object poses per image set to the maximum number of objects
                # present in a single image throughout the whole dataset. Check whether this upper limit is reached,
                # otherwise fill up with dummy encodings. Dummy boxes will later be filtered out by the matcher and not used for cost calculation
                if n_boxes < self.n_queries:
                    dummy_boxes = torch.tensor([[-1, -1, -1, -1] for i in range(self.n_queries-n_boxes)],
                                               dtype=DTYPE, device=t_boxes.device)

                    dummy_embed = torch.tensor([[-10] for i in range(self.n_queries-n_boxes)],
                                               dtype=DTYPE, device=t_boxes.device)
                    dummy_embed = dummy_embed.repeat(1, self.hidden_dim*2)
                    t_boxes = torch.vstack((t_boxes, dummy_boxes))
                    query_embed = torch.cat([query_embed, dummy_embed], dim=0)
                    dummy_classes = torch.tensor([-1 for i in range(self.n_queries-n_boxes)],
                                               dtype=torch.int, device=t_boxes.device)
                    t_classes = torch.cat((t_classes, dummy_classes))
                    
                pred_box_data.append(t_boxes)
                query_embed_data.append(query_embed)
                pred_class_data.append(t_classes)

        query_embed_data = torch.stack(query_embed_data)
        pred_box_data = torch.stack(pred_box_data)
        pred_class_data = torch.stack(pred_class_data)


        src_data = []
        mask_data = []

        for lvl, feat in enumerate(features):
            # Iterate over each feature map of the backbone returned.
            # If num_feature_levels == 1 then the backbone will only return the last one

            # feat - NestedTensor : features : '.tensors', mask : '.mask'
            src, mask = feat.decompose()
            src_data.append(self.input_proj[lvl](src))
            mask_data.append(mask)
            assert mask is not None
            
        if self.num_feature_levels > len(src_data):
            # If more feature levels are required than the backbone provides then additional feature maps are created
            
            len_src_data = len(src_data)
            for lvl in range(len_src_data, self.num_feature_levels):
                
                if lvl == len_src_data:
                    src = self.input_proj[lvl](features[-1].tensors)
                else:
                    src = self.input_proj[lvl](src_data[-1])
                    
                m = sample_data.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = PositionEncodingSine(NestedTensor(src, mask)).to(src.dtype)
                
                src_data.append(src)
                mask_data.append(mask)
                pos.append(pos_l)
    
        
        reference_point_data = pred_box_data[:, :, :2]
    
        # Pass everything to the transformer
        hs, init_reference, _, _, _ = self.transformer(src_data, mask_data, pos, query_embed_data, reference_point_data)
    
        output_translation = []
        output_rotation = []
        output_pose = []
        
        bs, _ = pred_class_data.shape
        output_idx = torch.where(pred_class_data > 0, pred_class_data, 0).view(-1)
    
        # Iterate over the decoder outputs to calculate the intermediate and final outputs (translation and rotation)
        for lvl in range(hs.shape[0]):
            if self.loss == 'split':
                output_rotation = self.rotation_head[lvl](hs[lvl])
                output_translation = self.translation_head[lvl](hs[lvl])
                
                # Select the correct output according to the predicted class in the class-specific mode
                output_rotation = output_rotation.view(bs * self.n_queries, self.n_classes, -1)
                output_rotation = torch.cat([query[output_idx[i], :] for i, query in enumerate(output_rotation)]).view(
                    bs, self.n_queries, -1)
    
                output_translation = output_translation.view(bs * self.n_queries, self.n_classes, -1)
                output_translation = torch.cat(
                    [query[output_idx[i], :] for i, query in enumerate(output_translation)]).view(bs, self.n_queries,
                                                                                                  -1)

                output_rotation = self.process_rotation(output_rotation)
        
                output_rotation_data.append(output_rotation)
                output_translation_data.append(output_translation)
                
            elif self.loss == 'pose':
                output_pose = self.pose_head[lvl](hs[lvl])
                
                # Select the correct output according to the predicted class in the class-specific mode
                output_pose = output_pose.view(bs * self.n_queries, self.n_classes, -1)
                output_pose = torch.cat([query[output_idx[i], :] for i, query in enumerate(output_rotation)]).view(
                    bs, self.n_queries, -1)
                output_pose = self.process_pose(output_pose)
                
                output_pose_data.append(output_pose)
    
            
        if self.loss == 'split':
            output_rotation_data = torch.stack(output_rotation_data)
            output_translation_data = torch.stack(output_translation_data)
            out = {'pred_translation': output_translation_data[-1], 'pred_rotation': output_rotation_data[-1],
               'pred_boxes': pred_box_data, 'pred_classes': pred_class_data}
            
        elif self.loss == 'pose':
            output_pose_data = torch.stack(output_pose_data)
            out = {'pred_pose': output_pose_data[-1], 'pred_boxes': pred_box_data, 'pred_classes': pred_class_data}

        return out, n_boxes_per_sample

     def process_rotation(self, rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process
        
        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        
        From PoET : https://github.com/aau-cns/poet/tree/main
        """
        bs, n_q, _ = rot_6d.shape
        rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]
        
        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(bs, n_q, 3, 3)  
        return rot_matrix

    def process_pose(self, pose_9d):
        
        # Assume pose_9d has shape [bs, n_q, 9]
        bs, n_q, _ = pose_9d.shape
        
        # Rotation components
        m1 = pose_9d[:, :, 0:3]  # [bs, n_q, 3]
        m2 = pose_9d[:, :, 3:6]  # [bs, n_q, 3]
        
        # Gram-Schmidt process for SO(3) rotation matrix
        x = F.normalize(m1, p=2, dim=2)                    # [bs, n_q, 3]
        z = F.normalize(torch.cross(x, m2, dim=2), p=2, dim=2)
        y = torch.cross(z, x, dim=2)                       # all [bs, n_q, 3]
        
        # Stack rotation matrix: [bs, n_q, 3, 3]
        rot_matrix = torch.cat((
            x.unsqueeze(-1),  # [bs, n_q, 3, 1]
            y.unsqueeze(-1),
            z.unsqueeze(-1)
        ), dim=3)
        
        # Translation vector: [bs, n_q, 3, 1]
        trans = pose_9d[:, :, 6:9].unsqueeze(-1)
        
        # Concatenate rotation and translation: [bs, n_q, 3, 4]
        pose_matrix_3x4 = torch.cat((rot_matrix, trans), dim=3)
        
        # Bottom row: [0, 0, 0, 1] → [1, 1, 1, 4] → expand to [bs, n_q, 1, 4]
        bottom_row = torch.tensor([0, 0, 0, 1], device=pose_9d.device, dtype=pose_9d.dtype)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(bs, n_q, 1, 4)
        
        # Final pose matrix: [bs, n_q, 4, 4]
        pose_matrix = torch.cat((pose_matrix_3x4, bottom_row), dim=2)
        
        return pose_matrix

class SetCriterion(nn.Module):
    """ This class computes the loss for pose estimation
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction 
    """
    def __init__(self, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
    
    def loss_translation(self, outputs, targets, indices):
        """
        Compute the loss related to the translation of pose estimation, namely the mean square error (MSE).
        outputs must contain the key 'pred_translation', while targets must contain the key 'relative_position'
        Position / Translation are expected in [x, y, z] meters
        """
        idx = self._get_src_permutation_idx(indices)
        src_translation = outputs["pred_translation"][idx]
        tgt_translation = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_translation)
    
        loss_translation = F.mse_loss(src_translation, tgt_translation, reduction='none')
        loss_translation = torch.sum(loss_translation, dim=1)
        loss_translation = torch.sqrt(loss_translation)
        losses = {}
        losses["loss_trans"] = loss_translation.sum() / n_obj
        return losses
    
    def loss_rotation(self, outputs, targets, indices):
        """
        Compute the loss related to the rotation of pose estimation represented by a 3x3 rotation matrix.
        The function calculates the geodesic distance between the predicted and target rotation.
        L = arccos( 0.5 * (Trace(R\tilde(R)^T) -1)
        Calculates the loss in radiant.
        """
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_rot = outputs["pred_rotation"][idx]
        tgt_rot = torch.cat([t['relative_rotation'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(tgt_rot)
    
        product = torch.bmm(src_rot, tgt_rot.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        losses = {}
        losses["loss_rot"] = rad.sum() / n_obj
        return losses

     def loss_pose(self, outputs, inverse, indices):
        """
        Compute the loss related to the pose matrix of pose estimation.
        outputs must contain the key 'pred_pose', while targets must contain the key 'inverse_pose'
        """
        idx = self._get_src_permutation_idx(indices)
        src_pose = outputs["pred_pose"][idx]
        inv_pose = torch.cat([t['inverse_pose'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        n_obj = len(inv_pose)
    
        product = torch.bmm(src_pose, inv_pose)
         
        identity = torch.eye(4, device=product.device, dtype=product.dtype).unsqueeze(0).expand(n_obj, 4, 4)  # [n_obj, 4, 4]

        loss_pose = F.mse_loss(product, identity, reduction='none') # [n_obj, 4, 4]
        loss_pose = loss_pose.view(n_obj, -1).sum(dim=1) # # [n_obj,]
        losses = {}
        losses["loss_pose"] = loss_pose.sum() / n_obj
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'translation': self.loss_translation,
            'rotation': self.loss_rotation,
            'pose': self.loss_pose
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)
    
    def forward(self, outputs, targets, n_boxes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            n_boxes: Number of predicted objects per image
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'enc_outputs'}
    
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, n_boxes)
    
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))
   
        return losses
    
    
    
             
                
                
                
                
                
            
        
        
        
        