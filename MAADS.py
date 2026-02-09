import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm import create_model
from copy import deepcopy
from timm.models.vision_transformer import VisionTransformer, Block
from timm.models.vision_transformer import Attention as TimmAttention
from module import Attention, PreNorm, FeedForward
from scipy.stats import norm

# -----------------------------
# Multi-Modal Action Density Scoring Module (Soft-Attention)
#開発意図
#各モダリティの各フレームに対してMLPを通すことで「分類に対してどのフレームが重要化」を学習する。
#このスコアを用いてフレーム選択を行う。
#このスコアは、各モダリティのフレームごとに計算され、最終的に全モダリティのスコアを合計して、各フレームの重要度を示す。
# -----------------------------
class MultiModalActionDensity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.modal_attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1)
            )

    def forward(self, cls_tkn):
        scores = self.modal_attention(cls_tkn) # (f, d)
        weights = scores.sum(dim=-1, keepdim=True) # (f 1)
        return weights

# -----------------------------
# Transformer Encoder for Temporal Attention
# -----------------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for norm_1, attn, norm_2, ff in self.layers:
            x = attn(norm_1(x)) + x
            x = ff(norm_2(x)) + x
        return self.norm(x)

# -----------------------------
# Attention Masking Utility
# 開発意図
# モダリティごとにトークン数を指定し、
# それに基づいてマスクを生成する。
# マスクは、各モダリティのトークン間の相互作用を許可し、
# モダリティ間の相互作用を禁止する。
# さらに、最初のトークンは常にマスクされないように設定する。
# これにより、最初のトークンは常に他のトークンと相互作用できる。
# 具体的には、最初のトークンは、すべてのモダリティのトークンと相互作用できるように設定され、
# 各モダリティのトークンは、そのモダリティ内のトークンとのみ相互作用できるように設定される。
# これにより、モダリティ間の情報の流れを制御しつつ、
# 各モダリティ内の情報の流れを促進することができる。
# -----------------------------
def create_attention_mask(num_modalities, tokens_per_modality):
    total_tokens = num_modalities * tokens_per_modality + 1
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)
    mask[0, :] = True
    mask[:, 0] = True
    for i in range(num_modalities):
        start = 1 + i * tokens_per_modality
        end = start + tokens_per_modality
        mask[start:end, start:end] = True
    return ~mask

# -----------------------------
# ViViT Model with M-AADS and Frame Pruning
# 開発意図
# ViViTモデルにマルチモーダルアクション密度スコアリング（M-AADS）とフレームプルーニングを組み込む。
# M-AADSは、各モダリティのフレームごとに重要度スコアを計算し、
# そのスコアに基づいてフレームを選択する。
# フレームプルーニングは、重要度スコアが指定されたしきい値を超えるフレームのみを保持する。
# これにより、計算コストを削減し、重要なフレームに焦点を当てることができる。
# モデルは、RGB、深度、スケルトンの3つのモダリティを入力として受け取り、
# 各モダリティのフレームごとに重要度スコアを計算する。
# その後、重要度スコアに基づいてフレームを選択し、選択されたフレームを用いて時系列トランスフォーマーを適用する。
# 最終的に、プーリング方法に基づいて出力を生成し、MLPヘッドを通じて分類を行う。
# -----------------------------
class ViViT(nn.Module):
    def __init__(self, num_classes, num_frames, image_size=224, patch_size=16, dim=192, depth=4, heads=3, pool='cls', dim_head=64, dropout=0., emb_dropout=0.,
                 target_pruning_rate=0.5):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.space_transformer = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        dim = self.space_transformer.embed_dim
        heads = dim // dim_head
        self.fix_param(self.space_transformer)
        self.space_transformer.cls_token.requires_grad = True
        self.num_modalities = 3
        self.space_transformer = apply_space_transformer(self.space_transformer, num_modalities=self.num_modalities)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_embedding = nn.Embedding(num_frames, dim)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * 4, dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.frame_selector = MultiModalActionDensity(dim)

        # 学習可能なしきい値（pruning threshold）
        self.learnable_threshold = nn.Parameter(torch.tensor(0.1))
        self.sigmoid_slope = 6.5

        self.module_names = ['space_transformer', 'temporal_token', 'temporal_embedding', 'temporal_transformer', 'mlp_head', 'frame_selector', 'learnable_threshold']
        
        self.prune_info = {'frame': 0, 
                           'topk': 0, 
                           'target_pruning_rate': target_pruning_rate, 
                           'target_z': norm.ppf(target_pruning_rate),
                           'scores': None,
                           }
        

    def fix_param(self, model):
        for p in model.parameters():
            p.requires_grad = False
    
    def get_threshold_loss(self, lambda_loss=0.07):
        scores = self.prune_info['scores']
        mean_score = scores.mean()
        std_score = scores.var() ** 0.5
        target_z = mean_score + self.prune_info['target_z'] * std_score
        return (self.learnable_threshold - target_z).abs() * lambda_loss
    
    def forward(self, rgb, depth, skeleton):
        b, f, c, h, w = rgb.shape
        self.prune_info['frame'] = f
        modalities = torch.stack([rgb, depth, skeleton])
        attn_mask = create_attention_mask(num_modalities=self.num_modalities, tokens_per_modality=self.num_patches)

        spatial_tokens = self.space_transformer(modalities, attn_mask)
        cls_tokens = spatial_tokens[:,0] # (f, d)

        scores = self.frame_selector(cls_tokens) # (f, 1)
        
        self.prune_info['scores'] = scores

        # Soft pruning via sigmoid attention score masking
        soft_mask = torch.sigmoid(self.sigmoid_slope * (scores - self.learnable_threshold))  # (f, 1)
        
        if self.training:
            masked_tokens = cls_tokens * soft_mask # (f, d)
        else:
            assert b == 1, "Batch size should be 1 for inference"
            sorted_idx = soft_mask.argsort(dim=0, descending=True) # (f, 1)
            topk = (soft_mask > self.learnable_threshold).sum()
            idxs = sorted_idx[:topk]
            self.prune_info['topk'] = topk.item()
            masked_tokens = cls_tokens.gather(-2, idxs.repeat(1, cls_tokens.shape[-1]))
        
        masked_tokens += self.temporal_embedding(torch.arange(masked_tokens.shape[-2], device=masked_tokens.device)) # (f, d)
        temporal_cls_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        temporal_tokens = torch.cat((temporal_cls_tokens, masked_tokens[None]), dim=1)

        temporal_tokens = self.temporal_transformer(temporal_tokens)

        if self.pool == 'mean':
            out = temporal_tokens.mean(dim=1)
        else:
            out = temporal_tokens[:, 0]

        return self.mlp_head(out)

# -----------------------------
# Space Transformer Modules (Unchanged)
# 開発意図
# ViViTの空間トランスフォーマーを拡張して、
# モダリティごとのトークンを処理できるようにする。
# これにより、各モダリティの特徴を個別に学習し、
# モダリティ間の相互作用を制御することができる。
# モダリティごとのトークンは、空間トランスフォーマーのパッチ埋め込み層を通じて処理され、    
# 位置埋め込みが適用される。
# その後、各モダリティのトークンは、モダリティ埋め込みを通じて拡張され、    
# トークンの次元が統一される。
# 最後に、トークンはパッチドロップ層を通じてドロップアウトされ、    
# 正規化層を通じて正規化される。
# その後、各ブロックが順番に適用され、最終的な特徴が得られる。
# -----------------------------
class SpaceTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor, attn_mask: torch.Tensor):
        n, b, f, c, h, w = x.shape
        x = rearrange(x, 'n b f c h w -> (b n f) c h w')
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        cls_tkn, x = x[0:1,0:1], x[:,1:]
        x = rearrange(x, '(b n f) t d -> (b f) n t d', b=b, n=n, f=f)
        x = x + self.modality_embedding(torch.arange(n, device=x.device)[None,:,None])
        x = rearrange(x, 'b_f n t d -> b_f (n t) d')
        x = torch.cat([cls_tkn.repeat(b*f, 1, 1), x], dim=-2)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, attn_mask.to(x.device))
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.forward_features(x, attn_mask)

class SpaceTransformerAttention(TimmAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask[None, None].to(attn.dtype) * -1e9
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpaceTransformerBlock(Block):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def apply_space_transformer(space_transformer, num_modalities):
    space_transformer.__class__ = SpaceTransformer
    space_transformer.modality_embedding = nn.Embedding(num_modalities, space_transformer.embed_dim)
    for module in space_transformer.modules():
        if isinstance(module, Block):
            module.__class__ = SpaceTransformerBlock
        if isinstance(module, TimmAttention):
            module.__class__ = SpaceTransformerAttention
    return space_transformer


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from timm import create_model
# from copy import deepcopy
# from timm.models.vision_transformer import VisionTransformer, Block
# from timm.models.vision_transformer import Attention as TimmAttention
# from module import Attention, PreNorm, FeedForward

# # -----------------------------
# # Multi-Modal Action Density Scoring Module (Soft-Attention)
# # -----------------------------
# class MultiModalActionDensity(nn.Module):
#     def __init__(self, dim, num_modalities=3):
#         super().__init__()
#         self.modal_attention = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, dim),
#                 nn.ReLU(),
#                 nn.Linear(dim, 1)
#             ) for _ in range(num_modalities)
#         ])
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, rgb_tokens, depth_tokens, skeleton_tokens):
#         modalities = [rgb_tokens, depth_tokens, skeleton_tokens]  # (b, f, d)
#         scores = [self.modal_attention[i](mod) for i, mod in enumerate(modalities)]  # list of (b, f, 1)
#         total_score = sum(scores)  # (b, f, 1)
#         weights = self.softmax(total_score)  # soft attention weights over time
#         return weights  # (b, f, 1)

# # -----------------------------
# # Transformer Encoder for Temporal Attention
# # -----------------------------
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.norm = nn.LayerNorm(dim)
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 nn.LayerNorm(dim),
#                 Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
#                 nn.LayerNorm(dim),
#                 FeedForward(dim, mlp_dim, dropout=dropout)
#             ]))

#     def forward(self, x):
#         for norm_1, attn, norm_2, ff in self.layers:
#             x = attn(norm_1(x)) + x
#             x = ff(norm_2(x)) + x
#         return self.norm(x)

# # -----------------------------
# # Attention Masking Utility
# # -----------------------------
# def create_attention_mask(num_modalities, tokens_per_modality):
#     total_tokens = num_modalities * tokens_per_modality + 1
#     mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool)
#     mask[0, :] = True
#     mask[:, 0] = True
#     for i in range(num_modalities):
#         start = 1 + i * tokens_per_modality
#         end = start + tokens_per_modality
#         mask[start:end, start:end] = True
#     return ~mask

# # -----------------------------
# # ViViT Model with M-AADS and Frame Pruning
# # -----------------------------
# class ViViT(nn.Module):
#     def __init__(self, num_classes, num_frames, image_size=224, patch_size=16, dim=192, depth=4, heads=3, pool='cls', dim_head=64, dropout=0., emb_dropout=0., pruning_threshold=0.01):
#         super().__init__()
#         self.num_patches = (image_size // patch_size) ** 2

#         self.space_transformer = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
#         dim = self.space_transformer.embed_dim
#         heads = dim // dim_head
#         self.fix_param(self.space_transformer)
#         self.space_transformer.cls_token.requires_grad = True
#         self.num_modalities = 3
#         self.space_transformer = apply_space_transformer(self.space_transformer, num_modalities=self.num_modalities)

#         self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.temporal_embedding = nn.Embedding(num_frames, dim)
#         self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * 4, dropout)
#         self.dropout = nn.Dropout(emb_dropout)
#         self.pool = pool
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.frame_selector = MultiModalActionDensity(dim)
#         self.pruning_threshold = pruning_threshold

#     def fix_param(self, model):
#         for p in model.parameters():
#             p.requires_grad = False

#     def forward(self, rgb, depth, skeleton):
#         b, f, c, h, w = rgb.shape
#         modalities = torch.stack([rgb, depth, skeleton])
#         attn_mask = create_attention_mask(num_modalities=self.num_modalities, tokens_per_modality=self.num_patches)

#         spatial_tokens = self.space_transformer(modalities, attn_mask)  # (b * f * m, tokens+1, d)
#         cls_tokens = spatial_tokens[:, 0]  # (b*f*3, d)
#         cls_tokens = cls_tokens.view(self.num_modalities, b, f, -1)  # (3, b, f, d)

#         rgb_cls, depth_cls, skeleton_cls = cls_tokens[0], cls_tokens[1], cls_tokens[2]  # (b, f, d)
#         frame_weights = self.frame_selector(rgb_cls, depth_cls, skeleton_cls)  # (b, f, 1)

#         # Apply pruning: keep only frames with score > threshold
#         mask = frame_weights.squeeze(-1) > self.pruning_threshold  # (b, f)

#         # Use boolean mask to filter frames
#         cls_spatial_tokens = rearrange(spatial_tokens[:, 0], '(b f m) d -> b f d', b=b, f=f, m=self.num_modalities)
#         kept_tokens = []
#         for i in range(b):
#             kept_tokens.append(cls_spatial_tokens[i][mask[i]])  # list of (f_kept, d)
#         max_len = max(t.shape[0] for t in kept_tokens)
#         padded_tokens = torch.zeros(b, max_len, cls_spatial_tokens.shape[-1], device=cls_spatial_tokens.device)
#         for i, t in enumerate(kept_tokens):
#             padded_tokens[i, :t.shape[0]] = t

#         kept_tokens = padded_tokens  # (b, kept_f, d)
#         kept_f = kept_tokens.shape[1]
#         kept_tokens += self.temporal_embedding(torch.arange(kept_f, device=kept_tokens.device)).unsqueeze(0)
#         temporal_cls_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
#         temporal_tokens = torch.cat((temporal_cls_tokens, kept_tokens), dim=1)

#         temporal_tokens = self.temporal_transformer(temporal_tokens)

#         if self.pool == 'mean':
#             out = temporal_tokens.mean(dim=1)
#         else:
#             out = temporal_tokens[:, 0]

#         return self.mlp_head(out)

# # -----------------------------
# # Space Transformer Modules (Unchanged)
# # -----------------------------
# class SpaceTransformer(VisionTransformer):
#     def forward_features(self, x: torch.Tensor, attn_mask: torch.Tensor):
#         n, b, f, c, h, w = x.shape
#         x = rearrange(x, 'n b f c h w -> (b n f) c h w')
#         x = self.patch_embed(x)
#         x = self._pos_embed(x)
#         cls_tkn, x = x[0:1,0:1], x[:,1:]
#         x = rearrange(x, '(b n f) t d -> (b f) n t d', b=b, n=n, f=f)
#         x = x + self.modality_embedding(torch.arange(n, device=x.device)[None,:,None])
#         x = rearrange(x, 'b_f n t d -> b_f (n t) d')
#         x = torch.cat([cls_tkn.repeat(b*f, 1, 1), x], dim=-2)
#         x = self.patch_drop(x)
#         x = self.norm_pre(x)
#         for blk in self.blocks:
#             x = blk(x, attn_mask.to(x.device))
#         x = self.norm(x)
#         return x

#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
#         return self.forward_features(x, attn_mask)

# class SpaceTransformerAttention(TimmAttention):
#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         if attn_mask is not None:
#             attn = attn + attn_mask[None, None].to(attn.dtype) * -1e9
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = attn @ v
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class SpaceTransformerBlock(Block):
#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x

# def apply_space_transformer(space_transformer, num_modalities):
#     space_transformer.__class__ = SpaceTransformer
#     space_transformer.modality_embedding = nn.Embedding(num_modalities, space_transformer.embed_dim)
#     for module in space_transformer.modules():
#         if isinstance(module, Block):
#             module.__class__ = SpaceTransformerBlock
#         if isinstance(module, TimmAttention):
#             module.__class__ = SpaceTransformerAttention
#     return space_transformer
