# deformable_transformer.py : This file defines the deformable transformer to be used in pose estimation

import copy

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, normal_

from src.model.deformable_attention.modules import MSDeformAttn

ACTIVATION = "relu"
DROPOUT = 0.1
N_HEADS = 8
N_LEVEL = 4
N_REF_POINTS = 4
LINEAR_SIZE = 1024
MODEL_SIZE = 256
DTYPE = torch.float32

N_ENC = 6
N_DEC = 8

def activation_fn(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()

def pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos

def create_layers(layer, n_layer):
    return nn.ModuleList([copy.deepcopy(layer) for i in range(n_layer)])

###############################################################################

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=MODEL_SIZE, d_ffn=LINEAR_SIZE, dropout=DROPOUT, activation=ACTIVATION, n_levels=N_LEVEL, n_heads=N_HEADS, n_points=N_REF_POINTS):
        super().__init__()

        # Multi head self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, X, pos, ref_points, spatial_dim, level_start_index, padding_mask=None):
        X_attn = self.self_attn(pos_embed(X, pos), ref_points, X, spatial_dim, level_start_index, padding_mask)
        X = X + self.dropout1(X_attn)
        X = self.norm1(X)

        X_linear = self.linear2(self.dropout2(self.activation(self.linear1(X))))
        X = X + self.dropout3(X_linear)
        X = self.norm2(X)
        
        return X


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=MODEL_SIZE, d_ffn=LINEAR_SIZE, dropout=DROPOUT, activation=ACTIVATION,n_levels=N_LEVEL, n_heads=N_HEADS, n_points=N_REF_POINTS):
        super().__init__()

         # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, X_obj_q, query_pos, ref_points, X_enc, spatial_dim, level_start_index, padding_mask=None):
        
        # Self attention
        Q = K = pos_embed(X_obj_q, query_pos)
        X_attn = self.self_attn(Q.transpose(0,1),K.transpose(0,1),X_obj_q.transpose(0,1))[0].transpose(0,1)
        X_q = X_obj_q + self.dropout1(X_attn)
        X_q = self.norm1(X_q)

        # Cross attention
        X_attn = self.cross_attn(pos_embed(X_q, query_pos),ref_points, X_enc, spatial_dim, level_start_index, padding_mask)
        X_q = X_q + self.dropout2(X_attn)
        X_q = self.norm2(X_q)

        # Feed forward layer

        X_linear = self.linear2(self.dropout3(self.activation(self.linear1(X_q))))
        X_q = X_q + self.dropout4(X_linear)
        X_q = self.norm3(X_q)

        return X_q

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = create_layers(encoder_layer, num_layers)
        self.num_layers = num_layers

    def reference_points(self, spatial_dim, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_dim):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=DTYPE, device=device),
                                          torch.linspace(0.5, W - 0.5, W, dtype=DTYPE, device=device)) # HxW
            
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H) # (1xHW) - normalization as per valid part of feature map
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            
            reference_points_list.append(ref)
            
        ref_points = torch.cat(reference_points_list, 1) 
        ref_points = ref_points[:, :, None] * valid_ratios[:, None]
        return ref_points
        
    def forward(self, X, spatial_dim, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = X
        reference_points = self.reference_points(spatial_dim, valid_ratios, device=X.device)
        
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_dim, level_start_index, padding_mask)

        return output

class DeformableTransformerDecoder(nn.Module):
    
    def __init__(self, decoder_layer, num_layers, return_intermediate = False):
        super().__init__()
        self.layers = create_layers(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
    def forward(self, X, reference_points, X_enc, enc_spatial_dim, enc_level_start_index, enc_valid_ratios,
                query_pos=None, enc_padding_mask=None):
        output = X

        intermediate = []
        intermediate_reference_points = []
        
        for l_id, layer in enumerate(self.layers):
           
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * enc_valid_ratios[:, None]
            
            output = layer(output, query_pos, reference_points_input, X_enc, enc_spatial_dim, enc_level_start_index, enc_padding_mask)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
        
######################################################################

class DeformableTransformer(nn.Module):
    
    def __init__(self, d_model=MODEL_SIZE, nhead=N_HEADS,
                 num_encoder_layers=N_ENC, num_decoder_layers=N_DEC, dim_feedforward=LINEAR_SIZE, dropout=DROPOUT,
                 activation=ACTIVATION, return_intermediate_dec=True,
                 num_feature_levels=N_LEVEL, dec_n_points=N_REF_POINTS,  enc_n_points=N_REF_POINTS):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:  
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        normal_(self.level_embed)

    def forward(self, X, mask_data, pos_embed_data, query_embed, reference_points=None):
        X_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_dim_data = []
        
        for lvl, (x, mask, pos_embed) in enumerate(zip(X, mask_data, pos_embed_data)):
            n, c, h, w = x.shape
            spatial_dim = (h,w)
            spatial_dim_data.append(spatial_dim)
            
            x = x.flatten(2).transpose(1,2) # nx(h*w)xc
            mask = mask.flatten(1) # nx(h*w)
            pos_embed = pos_embed.flatten(2).transpose(1,2) # nx(h*w)xc
            
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # nx(h*w)xc + 1x1xc
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
            X_flatten.append(x)
            mask_flatten.append(mask)
        
        X_flatten = torch.cat(X_flatten, 1)  # n, sum(h_i*w_i), c
        mask_flatten = torch.cat(mask_flatten, 1) # n, sum(h_i*w_i)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # n, sum(h_i*w_i), c

        spatial_dim_data = torch.as_tensor(spatial_dim_data, dtype=torch.long, device=X_flatten.device) # n
        level_start_index = torch.cat((spatial_dim_data.new_zeros((1, )), spatial_dim_data.prod(1).cumsum(0)[:-1])) # n+1 
        valid_ratio_data = torch.stack([self.valid_ratio(m) for m in mask_data], 1) # nxn_lvlx2
        
         # encoder
        X_enc = self.encoder(X_flatten, spatial_dim_data, level_start_index, valid_ratio_data, lvl_pos_embed_flatten, mask_flatten)

        n,_,c = X_enc.shape
        
        if len(query_embed.shape) == 2: # (num_queries, 2c), creating batch
            query_embed, tgt = torch.split(query_embed, c, dim=1) #  (num_queries, c), (num_queries, c)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1) # (1, num_queries, C) -> (bs, num_queries, C)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1) 
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
            
        if reference_points is None:
            reference_points = self.reference_points(query_embed).sigmoid()
            
        init_reference_out = reference_points

        # decoder
        hs, inter_references_out = self.decoder(tgt, reference_points, X_enc, spatial_dim_data, level_start_index, valid_ratio_data, query_embed, mask_flatten)

        
        return hs, init_reference_out, inter_references_out, None, None

    def valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

        
        
        

    
    
    

        
        
        
        
        