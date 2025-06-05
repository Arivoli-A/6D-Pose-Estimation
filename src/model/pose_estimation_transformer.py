# pose estimation transformer.py : This file defines the architecture of pose estimation transformer


import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
# from .backbone import build_backbone
from utils.matcher import HungarianMatcher
from .deformable_transformer import DeformableTransformer, create_layer
from .position_encoding import BoundingBoxEmbeddingSine
import copy

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
    def __init__(self, backbone, transformer, num_queries, num_feature_levels, n_classes, bbox_mode='gt', loss = 'split', n_nn_layer = 3):
        # bbox_mode : 'gt', 'backbone', 'jitter'
        # loss : 'pose', 'rot' (6D) 
        # ref_point, query - given by backbone (no learned embedding)
        # output : Each output for each class ('specific')
        # No loss from each layer of FNN

        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.backbone = backbone
        self.n_queries = num_queries
        self.n_classes = n_classes + 1  # +1 for dummy/background class
        self.bbox_mode = bbox_mode
        self.loss = loss
        self.n_nn_layer = n_nn_layer
        self.num_feature_levels = num_feature_levels
        
        if self.loss == 'split':
            self.t_dim = 3
            self.rot_dim = 6
            self.translation_head = feed_fwd_nn(hidden_dim, hidden_dim, self.t_dim * self.n_classes, self.n_nn_layer)
            self.rotation_head = MLP(hidden_dim, hidden_dim, self.rot_dim * self.n_classes, self.n_nn_layer)
        elif self.loss == 'pose':
            self.pose_dim = 9
            self.pose_head = feed_fwd_nn(hidden_dim, hidden_dim, self.pose_dim * self.n_classes, self.n_nn_layer)

        else:
            print(self.loss,' not implemented')

        if num_feature_levels > 1:
            # Use multi-scale features as input to the transformer
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            # If multi-scale then every intermediate backbone feature map is returned
            for n in range(num_backbone_outs):
                in_channels = backbone.num_channels[n]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            # If more feature levels are required than backbone feature maps are available then the last feature map is
            # passed through an additional 3x3 Conv layer to create a new feature map.
            # This new feature map is then used as the baseline for the next feature map to calculate
            # For details refer to the Deformable DETR paper's appendix.
            for n in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            # We only want to use the backbones last feature embedding map.
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        
        def forward(self,
    
        
        
        