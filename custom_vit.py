from typing import Tuple
import torch
import timm
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from einops import rearrange
import torch.nn.functional as F
import sys
import torch.nn as nn
class CustomBlock(Block):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        # if self.merge_info['cls_idx'] is not None:
        #     cls_idx_list = self.merge_info['cls_idx'].tolist() + [x.size(-2)]
        #     x_attn = []
        #     for s,e in zip(cls_idx_list[0:],cls_idx_list[1:]):
        #         x_f = x_norm[:, s:e]
        #         x_attn.append(self.attn(x_f))
        #     x_attn = torch.cat(x_attn, dim=-2)
        #     # x_attn = rearrange(x_attn,'() t d -> b t d', b=self.merge_info['batch_size'])
        x_attn = self.attn(x_norm, mask=self.merge_info['attn_mask'])
        # print(x_attn.shape)
        x = x + self.drop_path1(x_attn)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class CustomBlockCross(Block): 
    def _prepare_modules(self):
        import copy
        self.self_attn = self.attn
        embed_dim = self.attn.head_dim*self.attn.num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim, self.attn.num_heads)
        self.norm_post = nn.LayerNorm(embed_dim)
        # self.norm_post_2 = nn.LayerNorm(embed_dim)
        # self.block1 = copy.deepcopy(super())
        # self.block2 = copy.deepcopy(super())
    def forward_block(self, x: torch.Tensor):
        x = self.norm1(x)
        x_attn = self.self_attn(x)
        x = x + self.drop_path1(x_attn)
        x = x + self.drop_path2(self.mlp(x))
        return x
    def forward(self, x: list, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x1, x2, x3 = x
        
        x2, x3 = self.forward_block(x2), self.forward_block(x3)
        
        x1 = self.norm1(x1)
        x_attn = self.self_attn(x1)
        x1 = x1 + self.drop_path1(x_attn)
        
        x_cattn, _ = self.cross_attn(x1, x2, x2)
        x1 = self.norm_post(x1 + x_cattn)
        
        x_cattn, _ = self.cross_attn(x1, x3, x3)
        x1 = self.norm2(x1 + x_cattn)
        
        x1 = x1 + self.drop_path2(self.mlp(x1))
        return x


class CustomAttention(Attention):
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale # b, h, t, t
        # if self.merge_info['merge_mask'] is not None and not self.merge_info['inference']:
        #     attn_mask =  rearrange(self.merge_info['merge_mask'][:,0], 'b f d -> (b f) () d ()')
        #     attn = attn + attn_mask * (-1e9)
        if mask is not None:
            # mask = mask[None, None]
            # attn = attn + (1-mask) * (-1)*torch.inf #(-1e9) #HACK

            safe_large_neg = -1e9
            # mask を (B, 1, N, N) に合わせる（必要であれば device/dtype を一致）
            m = mask.to(attn.device)
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
            elif m.dim() == 3:
                m = m.unsqueeze(1)  # (B,1,N,N)
            attn = attn.masked_fill(m == 0, safe_large_neg)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def make_custom_vit_merge(transformer_class): #HACK
    class CustomVisionTransformerMerge(transformer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.threshold = nn.Parameter(torch.tensor(0.5))  # 学習可能しきい値
            self.merge_info = {
                'merge_mask': None,
                'cls_idx': None,
                'inference': True,
                "attn_mask": None,
                'batch_size': 1,
            }

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            return super().forward(*args, **kwdargs)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            for i, block in enumerate(self.blocks):
                if i == 5:
                    x = self.merge_tokens(x)
                    if self.merge_info['inference']:
                        x = self.reduce_tokens(x)
                x = block(x)
            x = self.norm(x)
            if self.merge_info.get('cls_idx', None) is not None:
                cls_idx = self.merge_info['cls_idx']
                x = x.gather(dim=-2, index=cls_idx[None,:,None].repeat(1,1,x.size(-1)))
            return x

        def reduce_tokens(self, x):
            x = rearrange(x, '(b f) t d -> (b f t) d', b=self.merge_info['batch_size'])
            x = x.gather(dim=-2, index=self.merge_info['keep_idx'].repeat(1, x.size(-1)))[None]
            return x

        def merge_tokens(self, x):
            if self.merge_info['inference']:
                assert self.merge_info['batch_size'] == 1
            x = rearrange(x, '(b f) t d -> b t f d', b=self.merge_info['batch_size'])
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)#HACK
            attn = x @ x.transpose(-1, -2)

        

            with torch.no_grad():
                threshold = self.threshold.clamp(0.0, 1.0)  # 学習可能パラメータを使用
                merge_mask = attn > threshold
                merge_mask[..., 0, :, :] = 0
                merge_mask_bool = merge_mask #HACK
                merge_mask = merge_mask.to(x.dtype).to(x.device)

            merge_tokens = merge_mask @ x

            # with torch.no_grad(): #HACK
            #     merge_tokens = merge_tokens / merge_mask.sum(-1, keepdim=True)
            #     merge_mask = merge_mask.triu_(1)
            #     merge_mask, top_row = self.convert_mask(merge_mask)

            with torch.no_grad():
                denom = merge_mask.sum(-1, keepdim=True)  # (b,t,f,1)
                # ゼロ除算防止
                denom_safe = denom.clamp(min=1.0)
                merge_tokens = merge_tokens / denom_safe

                merge_mask = merge_mask.triu_(1)
                merge_mask, top_row = self.convert_mask(merge_mask_bool)  # bool を渡すのが安全
                # ※ convert_mask が bool 想定ならそちらに合わせる

            if self.merge_info['inference']:
                top_row = rearrange((1 - top_row), 'b t f -> (b f) t')
                num_keep = top_row.sum(dim=-1)
                self.merge_info['attn_mask'] = self.create_attention_mask_2d_simple(num_keep)
                keep_idx = torch.nonzero(rearrange(top_row, 't f -> (t f)'))
                self.merge_info['keep_idx'] = keep_idx
                cls_idx = torch.cumsum(num_keep, dim=0)
                cls_idx[1:] = cls_idx[:-1].clone()
                cls_idx[0] = 0
                self.merge_info['cls_idx'] = cls_idx

            self.merge_info["merge_mask"] = merge_mask
            merge_tokens = rearrange(merge_tokens, 'b t f d -> (b f) t d')
            return merge_tokens

        def convert_mask(self, mask: torch.Tensor):
            top_row = mask[..., 0, :].to(int)
            mask1 = top_row.transpose(-1, -2).unsqueeze(-3).repeat(1, top_row.size(-1), 1, 1)
            mask2 = mask1.transpose(-2, -3)
            mask = mask1 | mask2
            return mask, top_row

        def create_attention_mask_2d_simple(self, lengths):
            total_len = lengths.sum()
            cumsum = torch.cat([torch.tensor([0]).to(lengths.device), lengths.cumsum(0)[:-1]])
            block_ids = torch.repeat_interleave(torch.arange(len(lengths)).to(lengths.device), lengths)
            mask = (block_ids.unsqueeze(1) == block_ids.unsqueeze(0))
            return mask.float()
        
        def get_threshold_loss(self, target_ratio=0.7):
            """
            Merge Mask から、マージされた割合を計算し、目標からのズレを罰する MSE ロスを返す。
            """
            if self.merge_info['merge_mask'] is None:
                return torch.tensor(0.0, device=self.threshold.device)

            merge_mask = self.merge_info['merge_mask']  # (b, t, f, f)
            B, T, F, _ = merge_mask.shape

            merge_ratio = merge_mask.sum() / (B * T * F * F)
            loss = (merge_ratio - target_ratio) ** 2
            return loss


    return CustomVisionTransformerMerge


def make_custom_vit_cross(transformer_class):
    class CustomVisionTransformerCross(VisionTransformer):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self.list_process = lambda x,f: list(map(lambda i: f(i), x))
            return super().forward(*args, **kwdargs)
        
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            x = self.list_process(x,self.patch_embed)
            x = self.list_process(x,self._pos_embed)
            x = self.list_process(x,self.patch_drop)
            x = self.list_process(x,self.norm_pre)
            for i, block in enumerate(self.blocks):
                x = block(x)
            x = self.list_process(x,self.norm)            
            return x[0]
    return CustomVisionTransformerCross
            
def apply_custom(model: VisionTransformer, method: str='merge'):
    print('APPLY CUSTOM')
    if method == 'merge':
        CustomVisionTransformer = make_custom_vit_merge(model.__class__)
        model.__class__ = CustomVisionTransformer
        
        model.threshold = nn.Parameter(torch.tensor(0.5))  # 学習可能しきい値
        
        model.merge_info = {
            # 'threshold' : 0.5,#HACK
            'merge_mask': None,
            'cls_idx': None,
            'inference': True,
            "attn_mask": None,
            }
        for module in model.modules():
            if isinstance(module, Attention):
                module.__class__ = CustomAttention
                module.merge_info = model.merge_info
            if isinstance(module, Block):
                module.__class__ = CustomBlock
                module.merge_info = model.merge_info
    elif method == 'cross':
        CustomVisionTransformer = make_custom_vit_cross(model.__class__)
        model.__class__ = CustomVisionTransformer
        for module in model.modules():
            if isinstance(module, Block):
                module.__class__ = CustomBlockCross
                module._prepare_modules()
    else:
        pass

if __name__ == '__main__':
    model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True)
    apply_custom(model, method='cross')
    # print(model)
    x = torch.randn(16, 3, 224, 224)
    # model.forward_blocks(x)
    # print(model.merge_info["merge_mask"].shape)
    out = model([x,x,x])
    print(out.shape)
    
