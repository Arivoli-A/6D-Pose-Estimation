import torch
import torch.nn.functional as F
from torch import nn
from src.utils.misc import NestedTensor

from .position_encoding import PositionEncodingSine

class backbone_data_parser(nn.Module):
    # Create backbone class from data obtained from output of backbone model (Yolov8) ran seperately
    def __init__(self, strides, num_channels):
        super().__init__() 
        self.strides = strides
        self.num_channels = num_channels

    def forward(self, sample_data_batch):
        predictions_out = []
        out_batch = []
    
        for sample_data in sample_data_batch:
            prediction_out = sample_data['prediction']
            features = sample_data['features']
            mask_img = sample_data['img_mask']
    
            assert mask_img is not None
            #print(mask_img.shape)
            out = {}  # Dict[str, NestedTensor]
            for i, x in enumerate(features):
                # Resize mask to match feature spatial dims
                resized_mask = F.interpolate(mask_img[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # originally it was mask_img[None] : mask_img was nxhxw
                #print(resized_mask.shape)
                out[i] = NestedTensor(x, resized_mask)
    
            predictions_out.append(prediction_out)
            out_batch.append(out)

        
        # Convert out_batch and predictions_out to batch-friendly format
        # Example: Convert to a list of NestedTensors for each sample
        out_batch_dict = {}
        for i, out in enumerate(out_batch):
            for name, x in out.items():
                if name not in out_batch_dict:
                    out_batch_dict[name] = []
                out_batch_dict[name].append(x)
    
        # Convert lists to batched tensors where possible
        for name, tensor_list in out_batch_dict.items():
            #print(name)
            #print(tensor_list[0].tensors.shape)
            out_batch_dict[name] = NestedTensor(
                torch.cat([x.tensors for x in tensor_list],dim=0),  # Stack tensors along batch dimension
                torch.cat([x.mask for x in tensor_list],dim=0)     # Stack masks along batch dimension
            )
    
        # Return predictions and batched output
        return predictions_out, out_batch_dict

    
        

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_encoding):
        super().__init__(backbone, position_encoding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.backbone = backbone

    def forward(self, tensor_list):
        # if backbone_type is 'output' : tensor_list: {'features': features, 'prediction': pred_objects, 'img_mask' : img_mask}
        predictions, xs = self[0](tensor_list)
        out = [] #  List[NestedTensor] 
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, predictions

