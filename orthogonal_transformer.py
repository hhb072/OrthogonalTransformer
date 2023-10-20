# --------------------------------------------------------
# Orthogonal Transformer
# Copyright (c) 2022 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        
class LayerNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x)     
 
class HouseholderTransform(nn.Module): 
    def __init__(self, scale=1, transpose=True):
        super().__init__()
        
        self.scale = scale
        self.transpose = transpose
        if scale == 0:
            return
                    
        self.n_dim = int(4 ** scale)
        self.h = self.w = int(2 ** scale)
        
        self.num_householders = self.n_dim
        
        self.u = nn.Parameter(torch.randn(self.num_householders, self.n_dim))
        self.I = nn.Parameter(torch.eye(self.n_dim, self.n_dim).unsqueeze(0))
        self.I.requires_grad = False
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'u'}
    
    def get_weight(self, ):
        u = F.normalize(self.u, dim=1)
        w = self.I - 2 * u.unsqueeze(-1) @ u.unsqueeze(1)
        w = torch.chain_matmul(*[x.squeeze(0) for x in w.chunk(self.num_householders, dim=0)])
        return w.reshape(self.n_dim, 1, self.h, self.w)
        
    def forward(self, x, dec, weight=None): 
        if self.scale == 0:
            if dec:
                return x.unsqueeze(1), None
            else:
                return x.flatten(1, 2)
        
        if dec:
          b, c, h, w = x.shape
          weight = self.get_weight()
          output = F.conv2d(x.reshape(b*c, 1, h, w), weight, stride=self.h, padding=0)
          output = output.reshape(b, c, output.shape[1], output.shape[2], output.shape[3])           
          if self.transpose:
            output = output.transpose(1,2)

          return output, weight
        else:
          b, n, c, h, w = x.shape
          # if weight is None:
            # weight = self.get_weight()         
          if self.transpose:            
            x = x.transpose(1,2).flatten(0,1) #(b*c,n,h,w) 
          output = F.conv_transpose2d(x, weight, stride=self.h, padding=0)
          output = output.reshape(b, c, output.shape[2], output.shape[3])
        return output 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., downsample=False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
              
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 5, 2 if downsample else 1, 2, groups=hidden_features)        
        # self.act2 = act_layer()
        # self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)        
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):  
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x) 
        x = self.conv1(x)        
        # x = self.act2(x)
        # x = self.drop(x)        
        # x = self.conv2(x)
        x = self.fc2(x)               
        x = self.drop(x)
        return x
  
class Attention(nn.Module):
    def __init__(self, dim, num_wavelets, window_size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., dpe=True):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_wavelets = num_wavelets
        
        self.dpe = dpe
        
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
        
        # self.shuffle = nn.Conv2d(num_wavelets*3, num_wavelets*3, 1, 1, 0, groups=3) if num_wavelets > 1 else nn.Identity()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # if not dpe:
            # # define a parameter table of relative position bias
            # self.relative_position_bias_table = nn.Parameter(
                # torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # # get pair-wise relative position index for each token inside the window
            # coords_h = torch.arange(self.window_size[0])
            # coords_w = torch.arange(self.window_size[1])
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            # relative_coords[:, :, 1] += self.window_size[1] - 1
            # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, K, N, C = x.shape     
        
        # if K > 1:
            # q, k, v = self.qkv(x).chunk(3, dim=-1)
            # q, k, v = self.shuffle(torch.cat((q, k, v), dim=1)).reshape(B, K*3, N, self.num_heads, C // self.num_heads).transpose(2, 3).chunk(3, dim=1)
            # q = q.flatten(0, 1)
            # k = k.flatten(0, 1)
            # v = v.flatten(0, 1)
        # else:
        q, k, v = self.qkv(x).reshape(B*K, N, self.num_heads, C // self.num_heads * 3).transpose(1, 2).chunk(3, dim=-1)
                    
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # if not self.dpe:
            # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                # self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1) # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, K, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        K = self.num_wavelets
        # flops += N * self.dim * K * K if K > 1 else 0
        flops += N * self.dim * 3 * self.dim * K
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N * K
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads) * K
        # x = self.proj(x)
        flops += N * self.dim * self.dim * K
        return flops
 
class HouseholderAttention(nn.Module):
    def __init__(self, dim, scale, window_size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dpe=True, shuffle=False, layerscale=False, downsample=False):
        super().__init__()
        
        self.scale2 = int(2 ** scale)
        self.num_wavelets = num_wavelets = int(4 ** scale)
        self.window_size = window_size
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.layerscale = layerscale
        
        init_values = 1.0e-5
        
        self.dpe = nn.Conv2d(dim[0]*num_wavelets, dim[0]*num_wavelets, 3, 1, 1, groups=dim[0]*num_wavelets) if dpe else None
                
        self.householder = HouseholderTransform(scale)
                
        self.norm1 = LayerNorm1d(dim[0])
        self.attn = Attention(
            dim[0], num_wavelets, window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, dpe=dpe)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = LayerNorm2d(dim[0])
        self.mlp2 = Mlp(in_features=dim[0], hidden_features=int(dim[0] * mlp_ratio), out_features=dim[1], act_layer=act_layer, drop=drop, downsample=downsample)
        
        self.downsample = nn.Sequential(LayerNorm2d(dim[0]), nn.Conv2d(dim[0], dim[1], 3, 2, 1)) if downsample else nn.Identity()
        
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, num_wavelets, 1, dim[0]),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim[1], 1, 1),requires_grad=True)
            
    def get_householder_attention(self, x):
        _, _, H, W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.scale2 - W % self.scale2) % self.scale2
        pad_b = (self.scale2 - H % self.scale2) % self.scale2
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, h, w = x.shape
    
        x, weight = self.householder(x, True)
        # print(weight)
        b, n, c, h, w = x.shape       
        x = x.flatten(-2).transpose(-2, -1).contiguous()
        x = self.attn(self.norm1(x))
        x = x.transpose(-2, -1).contiguous().reshape(b, n, c, h, w)
        x = self.householder(x, False, weight)
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        
        return x
               
           
    def forward(self, x):
        '''
            x: (b, c, h, w)            
        '''                        
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.get_householder_attention(x))
            x = self.downsample(x) + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x)))
        else:
            x = x + self.drop_path(self.get_householder_attention(x))
            x = self.downsample(x) + self.drop_path(self.mlp2(self.norm2(x)))                
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, scale, window_size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dpe=True, shuffle=False, layerscale=False, downsample=False):
        super().__init__()
        
        self.scale = scale
        self.num_wavelets = num_wavelets = int(4 ** scale)
        self.window_size = window_size
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        self.layerscale = layerscale
        
        init_values = 1.0e-5 
        
        
        self.dpe = nn.Conv2d(dim[0], dim[0], 3, 1, 1, groups=dim[0]) if dpe else None
        # self.shuffle = nn.Conv2d(num_wavelets, num_wavelets, 1, 1, 0, bias=False) if (shuffle and self.num_wavelets>1) else None
                        
        self.norm1 = LayerNorm1d(dim[0])
        self.attn = Attention(
            dim[0], num_wavelets, window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, dpe=dpe)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # self.norm2 = nn.BatchNorm2d(dim)
        self.norm2 = LayerNorm2d(dim[0])
        self.mlp2 = Mlp(in_features=dim[0], hidden_features=int(dim[0] * mlp_ratio), out_features=dim[1], act_layer=act_layer, drop=drop, downsample=downsample)
        
        self.downsample = nn.Sequential(LayerNorm2d(dim[0]), nn.Conv2d(dim[0], dim[1], 3, 2, 1)) if downsample else nn.Identity()    
        
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, 1, 1, dim[0]),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim[1], 1, 1),requires_grad=True)
            
    def get_window_attention(self, x):        
        b, c, H, W = x.shape
        if self.scale > 0:
            h0, w0 = self.window_size
        else:
            h0, w0 = H, W
        
        pad_l = pad_t = 0
        pad_r = (w0 - W % w0) % w0
        pad_b = (h0 - H % h0) % h0
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, h, w = x.shape
        
        x = x.reshape(b, c, h//h0, h0, w//w0, w0).permute(0, 2, 4, 3, 5, 1).contiguous().reshape(b, (h//h0)*(w//w0), h0*w0, c)
        x = self.attn(self.norm1(x))       
        x = x.reshape(b, h//h0, w//w0, h0, w0, c).permute(0, 5, 1, 3, 2, 4).contiguous().reshape(b, c, h, w)
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        
        return x
        
    def forward(self, x):
        '''
            x: (b, c, h, w)            
        '''
                       
        # mlp
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.get_window_attention(x))
            x = self.downsample(x) + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.get_window_attention(x))
            x = self.downsample(x) + self.drop_path(self.mlp2(self.norm2(x))) 
       
        return x
        
class BasicLayer(nn.Module):        
    def __init__(self, attn, num_layers, dim, scale, window_size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, dpe=True, shuffle=False, layerscale=True):
        super().__init__()
        
        self.num_wavelets = num_wavelets = int(4 ** scale)
        self.window_size = window_size
                
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([attn[i]([dim[0], dim[0] if i<num_layers-1 else dim[1]], scale, window_size, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                 drop_path[i] if isinstance(drop_path, list) else drop_path, act_layer, norm_layer, dpe=dpe, shuffle=shuffle, layerscale=layerscale, downsample=(downsample and (i==num_layers-1))) for i in range(num_layers)])
                 
        # if downsample is not None:
            # self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        # else:
            # self.downsample = None
         
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # if self.downsample is not None:
            # x = self.downsample(x)
        return x
       
class WaveletEmbed(nn.Module):        
    def __init__(self, n_convs=3, wavelet_level=3, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.n_convs = n_convs
        
        m_proj = [nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(embed_dim//2),
                    nn.ReLU(True),
                    nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    ]
        for _ in range(n_convs):
            m_proj += [nn.ReLU(True),nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        ]
        if n_convs > 1:
            m_proj += [nn.ReLU(True),nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
                    LayerNorm2d(embed_dim),]
        self.proj = nn.Sequential(*m_proj)
        
            
    def forward(self, x):
        x = self.proj(x) # (b, c, h/p, w/p)        
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        
        self.reduction = nn.Conv2d(dim[0], dim[1], 3, 2, 1, bias=False)
        # self.norm = nn.BatchNorm2d(dim[1])
        self.norm = nn.LayerNorm(dim[0])

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xn = self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.reduction(xn)
        # x = self.norm(x)        
        return x
        
    def flops(self, HW):
        # H, W = self.input_resolution
        flops = HW//4 * self.dim[1] 
        flops += HW//4 * self.dim[0] * self.dim[1] * 9
        return flops

class OrthogonalTransformer(nn.Module):    

    def __init__(self, n_convs=3, wavelet_level=3, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, dpe=False, shuffle=False, layerscale=True, **kwargs):
        super().__init__()
        
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.wavelet_embed = WaveletEmbed(n_convs=n_convs, wavelet_level=wavelet_level, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0], norm_layer=norm_layer if self.patch_norm else None)
       
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # attn = [(WindowAttention if i%2 ==0 else WaveletAttention) for i in range(sum(depths))]
        attn = [(WindowAttention if i%2 ==0 else HouseholderAttention) for i in range(sum(depths))]
        # attn = [HouseholderAttention for i in range(sum(depths))]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(attn[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],
                               scale=wavelet_level-i_layer,
                               window_size=to_2tuple(window_size),
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               dpe=dpe,
                               shuffle=shuffle,
                               layerscale=layerscale)
            self.layers.append(layer)

        self.norm = LayerNorm2d(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.wavelet_embed(x)        
        x = self.pos_drop(x)
        
        for layer in self.layers:            
            x = layer(x)
        
        # x = self.norm(x)  # B, 1, L, C
        x = self.norm(x)
        x = self.avgpool(x).flatten(1)  # B C 1        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
       
@register_model
def orthogonal_small(window_size, drop, drop_path):
    model = OrthogonalTransformer(img_size=224,
                                n_convs = 2,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                                embed_dim=[64, 128, 256, 512],
                                depths=[3, 5, 13, 3 ],
                                num_heads=[2, 4, 8, 16],
                                mlp_ratio=4,
                                window_size=window_size,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=drop,
                                drop_path_rate=drop_path,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,                                
                                layerscale=False,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    return model    
