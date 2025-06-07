import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import NestedTensor

from .position_encoding import PositionEncodingSine

class backbone_data_parser():
    # Create backbone class from data obtained from output of backbone model (Yolov8) ran seperately
    def __init__(self, strides, num_channels):
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, sample_data):
        prediction_out = sample_data['prediction']

        out = {} # Dict[str, NestedTensor] 
        for i in range(data['features'].shape[0]):
            x = sample_data['features'][i]
            m = sample_data['img_mask']
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # Mask resizing done from original image dim to feature dim
            out[name] = NestedTensor(x, mask)
        return prediction_out, out
    
        

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_encoding):
        super().__init__(backbone, position_encoding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        self.backbone = backbone

    def forward(self, tensor_list):
        # if backbone_type is 'output' : tensor_list: {'features': features, 'prediction': pred_objects, 'img_mask' : img_mask}
        predictions, xs = self[0].backbone(tensor_list)
        out = [] #  List[NestedTensor] 
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, predictions

