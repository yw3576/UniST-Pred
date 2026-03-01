# -*- coding: utf-8 -*-

from torch import nn
from .model_gtn import GTN
from .model_tsmixer import TSMixerModel
import torch
import numpy as np
import torch.nn.functional as F
import copy

# ---- Squeeze-and-Excitation on (B, C, L) ----
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4, activation=nn.ReLU()):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            activation,
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):  # x: (B, C, L)
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)     # (B, C)
        e = self.fc(s).view(b, c, 1)    # (B, C, 1)
        return x * e                    # (B, C, L)


# ---- Residual fusion block with SE, taking (B, L, C) and returning (B, L, C) ----
class ResidualFusionBlockSE(nn.Module):
    """
    ResNeXt-style 1x1 residual block with SE.

    Input:  (B, L, C)
    Output: (B, L, C)  (same C)
    """
    def __init__(self, channels=13, widen_factor=2, groups=13,
                 activation=nn.ReLU(), se_reduction=4, dropout_p=0.2):
        super().__init__()
        hidden = channels * widen_factor
        #print(hidden)
        #print(groups)
        
        
        assert hidden % groups == 0, "hidden must be divisible by groups"

        self.conv1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=1, groups=groups)
        self.se    = SEBlock(hidden, reduction=se_reduction, activation=activation)
        self.conv3 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.dropout = nn.Dropout1d(dropout_p)  # or nn.Dropout(dropout_p)
        self.activation = activation

    def forward(self, x_bl):  # x_bl: (B, L, C)
        # internal: (B, C, L) for Conv1d
        x = x_bl.permute(0, 2, 1)      # (B, C, L)
        identity = x

        out = self.activation(self.conv1(x))    # (B, hidden, L)
        out = self.activation(self.conv2(out))  # (B, hidden, L)
        out = self.se(out)                      # (B, hidden, L)
        out = self.conv3(out)                   # (B, C, L)

        out = identity + out                    # residual
        out = self.activation(out)              # (B, C, L)
        return out.permute(0, 2, 1)             # back to (B, L, C)


# ---- Fusion module: TS-Mixer + GTN -> stacked residual SE blocks ----
class FusionSE24(nn.Module):
    """
    y_ts, y_gtn: (B, L, 12)
    Concatenate along channel -> (B, L, 24), then N residual+SE blocks.
    Output: (B, L, 24)
    """
    def __init__(self, channels=13, num_blocks=3, widen_factor=2, groups=13,
                 activation=nn.ReLU(), se_reduction=4):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualFusionBlockSE(
                channels=channels,
                widen_factor=widen_factor,
                groups=groups,
                activation=activation,
                se_reduction=se_reduction
            ) for _ in range(3)]
        )

    def forward(self, y_ts, y_gtn):
        # both (B, L, 12)
        if y_ts.dim() != 3 or y_gtn.dim() != 3:
            raise ValueError("Expected y_ts, y_gtn as (B, L, 12)")
        x = torch.cat([y_ts, y_gtn], dim=-1)  # (B, L, 24)
        x = self.blocks(x)                    # (B, L, 24)
        return x

class UniST_Pred(nn.Module):
    """Include Reversible instance normalization https://openreview.net/pdf?id=cGDAkQo1C0p
    """    

    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers, args, input_length: int, forecast_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int,  dropout: float, eps: float = 1e-8, reduction: int=2):
        super(UniST_Pred, self).__init__()
        self.gtn = GTN(num_edge=num_edge,
                    num_channels=num_channels,
                    w_in = w_in,
                    w_out = w_out,
                    num_class=num_class,
                    num_layers=num_layers,
                    num_nodes=num_nodes,
                    args=args)
        
        self.tsmixer = TSMixerModel(
            input_length=input_length,
            forecast_length=forecast_length,
            no_feats=no_feats,
            feat_mixing_hidden_channels=feat_mixing_hidden_channels,
            no_mixer_layers=no_mixer_layers,
            dropout=dropout
            )
        self.num_class = num_class
        self.forecast_length = forecast_length
        self.act = nn.Sigmoid()
        #self.mean_ = nn.Parameter(torch.zeros(no_feats))
        #self.std_ = nn.Parameter(torch.ones(no_feats))
        
        #self.final = nn.Linear(2,1)
        self.ratio = nn.Parameter(torch.zeros(1))
        
        self.residual = FusionSE24(channels=1+self.forecast_length, groups=1+self.forecast_length, se_reduction=reduction)
        self.w_in = w_in
        self.final = nn.Linear(1+self.forecast_length, self.forecast_length)
        #self.final = nn.Linear(13, 1)
        
    def forward(self, A, X, new_X, target_x, num_nodes=None, eval=False, node_labels=None):
        # X shape: N, sequence_length, nodes
        # each GNT output shape: nodes, 1
        # tsmixer input shape: N, sequence_length, nodes
        X = X.permute(0,2,1)  # shape: N, nodes, sequence_length
        shape = X.shape
        N = shape[0]
        DEVICE = X.device
        
        #print(X.shape)
        out_total = torch.zeros((N, shape[1], self.num_class+self.forecast_length))
        out_total = out_total.to(DEVICE)
        
        for i in range(N):
            x_new = X[i] #shape nodes, sequence length
            X_ = torch.zeros(shape[1], self.w_in).to(DEVICE)
            if self.w_in==1 or self.w_in==2:
                X_[:,0] = x_new[:,-1]
            if self.w_in==(self.forecast_length+1) or self.w_in==(self.forecast_length+1):
                X_[:,:self.forecast_length] = x_new[:,-self.forecast_length:]
                
            if self.w_in == 2 or self.w_in==(self.forecast_length+1):
                X_[:,1] = torch.tensor(new_X[i], dtype=torch.float32).to(DEVICE)

            x_e,_ = self.gtn(A, X_, target_x, num_nodes=None, eval=False, node_labels=None)
            x_e = torch.unsqueeze(x_e, 0) #shape: 1, nodes, 1

        
            x_new = torch.unsqueeze(x_new,0) 
            x_new = x_new.permute(0,2,1) # shape: 1, seuqence length, nodes
        
            out = self.tsmixer(x_new) # shape: 1, target length, nodes
            out = out.permute(0,2,1) # shape: 1, nodes, target length
            
            out_total[i:i+1, :, :1] = x_e
            out_total[i:i+1, :, 1:] = out
        
        #out = out + torch.normal(self.mean_, self.act(self.std_)*0.1)
        #out_total = out_total.permute(0,2,1)
        out_total = self.residual(out_total[:,:,:1], out_total[:,:,1:])
        #print(out_total.shape)
        #out_total = out_total.permute(0,2,1)
        out_total = self.final(out_total)
        #out_total = torch.squeeze(out_total, 3)
        out_total = out_total.permute(0,2,1)
        #print(out_total.shape)
        
        return out_total

        


