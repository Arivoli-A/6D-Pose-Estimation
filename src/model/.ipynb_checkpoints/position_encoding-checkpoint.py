# position_encoding : This file contains the definition of sine position encoding to be used in transformer. (https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py)

import math
import torch
from torch import nn

from src.utils.misc import NestedTensor

DTYPE = torch.float32

class PositionEncodingSine(nn.Module):
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be true if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        
        not_mask = ~mask   # Get valid regions of feature maps
        print(not_mask.shape)
        y_embed = not_mask.cumsum(dim = 1, dtype=DTYPE) # Assign position to valid regions. height (rows)
        x_embed = not_mask.cumsum(dim = 2, dtype=DTYPE) # Assign position to valid regions. width (cols)
        print(y_embed.shape)
        print(x_embed.shape)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale  # shifts the index so that position 1 becomes 0.5
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=DTYPE, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEncodingBoundingBoxSine(nn.Module):
    # Position Embedding for bounding box 
    def __init__(self, num_pos_feats=32, temperature=2):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, bbox: torch.Tensor):        
        dim_t = torch.arange(self.num_pos_feats, dtype=DTYPE, device=x.bbox)
        dim_t = self.temperature ** dim_t #self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        x_enc = bbox[:, 0, None] / dim_t
        y_enc = bbox[:, 1, None] / dim_t
        h_enc = bbox[:, 2, None] / dim_t
        w_enc = bbox[:, 3, None] / dim_t
        
        #x_enc = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #y_enc = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        x_enc = torch.cat((x_enc.sin(), x_enc.cos()), dim=-1)
        y_enc = torch.cat((y_enc.sin(), y_enc.cos()), dim=-1)
        w_enc = torch.cat((w_enc.sin(), w_enc.cos()), dim=-1)
        h_enc = torch.cat((h_enc.sin(), h_enc.cos()), dim=-1)
        pos_embed = torch.cat((x_enc, y_enc, w_enc, h_enc), dim=-1)
        return pos    