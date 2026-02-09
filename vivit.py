import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from timm import create_model
from custom_vit import apply_custom
from copy import deepcopy
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, num_classes, num_frames, image_size=224, patch_size=16, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, custom=True, skeleton_dim=12, skeleton_joint=25):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )
        # self.vit = create_model('vit_tiny_patch16_224_in21k', pretrained=True)
        # self.to_patch_embedding = self.vit.patch_embed
        # self._pos_embed = self.vit._pos_embed
        
        #if custom:
         #   apply_custom(vit)
        #self.fix_param(vit.blocks)

        # if custom:
        #     apply_custom(self.vit)
        #     self.fix_param(self.vit.blocks)  # カスタムの場合のみパラメータを固定
        #     self.space_transformer = self.vit.forward_blocks 
        #     self.merge_info = self.vit.merge_info
        # else:
        #     self.space_transformer = self.vit.blocks 

        

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        ##self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        ##self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # self.pos_embedding = self.vit.pos_embed
        # self.space_token =  self.vit.cls_token
        # self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        # self.space_transformer.load_state_dict(vit.blocks.state_dict())
        # self.space_transformer = self.vit.forward_blocks if custom else vit.blocks
        # self.space_transformer = create_model('vit_small_patch16_224.augreg_in21k', pretrained=True, num_classes=0)
        self.space_transformer = create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=True, num_classes=0)
        dim = self.space_transformer.embed_dim
        heads = dim // dim_head
        self.fix_param(self.space_transformer)
        self.space_transformer.cls_token.requires_grad = True
        self.space_token =  self.space_transformer.cls_token
        if custom:
            apply_custom(self.space_transformer)
        
        self.merge_info = self.space_transformer.merge_info if custom else {}

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_embedding = nn.Embedding(num_frames, dim)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        # self.temporal_transformer = vit.forward_blocks if custom else vit.blocks
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.linear_skeleton_data = nn.Linear(skeleton_dim, dim)
        # self.linear_skeleton_joint = nn.Linear(skeleton_dim*skeleton_joint, self.feature_size)
        self.joint_embedding = nn.Embedding(skeleton_joint, dim)
        
        self.module_names = ['space_transformer', 'temporal_token', 'temporal_embedding', 'temporal_transformer', 'mlp_head',]#HACK
        
    def forward(self, x):
        if type(x) is list:
            features = []
            for data in x:
                n, f, j, d = data.shape
                data = rearrange(data, 'n f j d -> (n f j) d')
                data = self.linear_skeleton_data(data)
                data = rearrange(data, '(n f j) d -> n f j d', n=n, f=f, j=j)
                data = torch.mean(data, dim=0)
                features.append(data)
            x = torch.stack(features)
            b, fs, j, d = x.shape
            x = x + self.joint_embedding(torch.arange(j)[None,None].repeat(b, fs, 1).to(x.device))
            cls_space_tokens = self.space_token[None].repeat(b,fs,1,1)
            x = torch.cat((cls_space_tokens, x), dim=-2)
            x = rearrange(x, 'b f j d -> (b f) j d')
            x = self.vit.blocks(x)
            x = rearrange(x[:, 0], '(b f) ... -> b f ...', b=b)
        else:
            print(x.shape)
            x = x.unsqueeze(0)
            b, f, _, _, _ = x.shape
            self.merge_info['batch_size'] = b
            x = rearrange(x, 'b f c h w -> (b f) c h w')
            x = self.space_transformer(x)
        # if self.merge_info.get('cls_idx', None) is not None:
        #     cls_idx = self.merge_info['cls_idx']
        #     x = x.gather(dim=-2, index=cls_idx[None,:,None].repeat(1,1,x.size(-1)))
        # else:
            x = rearrange(x, '(b t) ... -> b t ...', b=b)
        x = x + self.temporal_embedding(torch.arange(f)[None].repeat(b, 1).to(x.device))
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        
        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    
    def fix_param(self, model):
        for p in model.parameters():
            p.requires_grad = False
    def learnable(self, model):
        for p in model.parameters():
            p.requires_grad = True
    
        # ViViT クラス（MAADS.py）の中に以下を追加 #HACK
    def get_threshold_loss(self, target_ratio=0.8):
        if hasattr(self.space_transformer, "get_threshold_loss"):
            return self.space_transformer.get_threshold_loss(target_ratio=target_ratio)
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    model = ViViT(100, 16).cuda()
    
    out = model(img)
    print(out.shape)

    
    
    