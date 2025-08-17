from typing import Optional, Type, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial

from timm.layers import DropPath, use_fused_attn
from timm.layers.helpers import to_2tuple
from rotary_embedding_torch import RotaryEmbedding

class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for efficient fine-tuning.
    
    Args:
        in_features: Input dimension of the layer.
        out_features: Output dimension of the layer.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA adaptation.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Apply LoRA adaptation to the input."""
        return (x @ self.lora_A @ self.lora_B) * self.scale

class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) used in Vision Transformers.
    
    Supports LoRA adaptation and optional convolutional layers.
    
    Args:
        in_features: Input dimension.
        hidden_features: Hidden layer dimension (defaults to in_features).
        out_features: Output dimension (defaults to in_features).
        act_layer: Activation function (default: nn.GELU).
        norm_layer: Normalization layer (default: None).
        bias: Whether to use bias in linear/conv layers.
        drop: Dropout probability.
        use_conv: Whether to use 1x1 convolutions instead of linear layers.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        # Main MLP layers
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        # LoRA adaptation layers
        self.lora1_glo = LoRALayer(in_features, hidden_features)
        self.lora2_glo = LoRALayer(hidden_features, out_features)
        self.lora1_loc = LoRALayer(in_features, hidden_features)
        self.lora2_loc = LoRALayer(hidden_features, out_features)

    def forward(self, x: Tensor, use_lora: int = 0) -> Tensor:
        """Forward pass with optional LoRA adaptation.
        
        Args:
            x: Input tensor of shape (B, N, C) or (B, C, H, W) if use_conv=True.
            use_lora: LoRA mode (0: none, 1: global, 3: local).
        
        Returns:
            Output tensor with the same shape as input.
        """
        if use_lora == 1:
            x = self.fc1(x) + self.lora1_glo(x)
        elif use_lora == 3:
            x = self.fc1(x) + self.lora1_loc(x)
        else:
            x = self.fc1(x) + self.lora1_loc(x)
        
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        
        if use_lora == 1:
            x = self.fc2(x) + self.lora2_glo(x)
        elif use_lora == 3:
            x = self.fc2(x) + self.lora2_loc(x)
        else:
            x = self.fc2(x) + self.lora2_loc(x)
        
        x = self.drop2(x)
        return x

def build_4d_position_ids(T: int, valid_indices: list, grid_size: int, tpp: int, device: str = 'cuda') -> Tensor:
    """Build 4D position IDs for RoPE.
    
    Args:
        T: Number of temporal frames.
        valid_indices: List of valid spatial indices.
        grid_size: Size of the spatial grid.
        tpp: Tokens per patch.
        device: Device for tensor operations.
    
    Returns:
        Tensor of shape [4, 1, T * np * tpp] containing temporal, token, and spatial position IDs.
    """
    np = len(valid_indices)
    total_tokens = T * np * tpp

    # Spatial coordinates
    r_coords = torch.tensor([i // grid_size for i in valid_indices], device=device)
    c_coords = torch.tensor([i % grid_size for i in valid_indices], device=device)
    h_ids = r_coords.repeat_interleave(tpp).repeat(T).unsqueeze(0)
    w_ids = c_coords.repeat_interleave(tpp).repeat(T).unsqueeze(0)

    # Temporal and token positions
    t_ids = torch.arange(T, device=device).repeat_interleave(np * tpp).unsqueeze(0)
    tpp_ids = torch.arange(tpp, device=device).repeat(np).repeat(T).unsqueeze(0)

    return torch.stack([t_ids, tpp_ids, h_ids, w_ids], dim=0)

def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the tensor for RoPE (sine on odd, cosine on even indices)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: Tensor, pos_ids: Tensor, rotary_dim: int) -> Tensor:
    """Apply Rotary Position Embedding (RoPE) to the input tensor.
    
    Args:
        x: Input tensor of shape [B, L, H, D].
        pos_ids: Position IDs of shape [B, L].
        rotary_dim: Number of dimensions to apply rotation to.
    
    Returns:
        Tensor with RoPE applied to the first rotary_dim dimensions.
    """
    sin_cos_dim = rotary_dim // 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Compute rotation angles
    pos = pos_ids.unsqueeze(-1).float()
    freqs = torch.arange(0, sin_cos_dim, dtype=torch.float32, device=x.device)
    freqs = 1.0 / (10000 ** (freqs / sin_cos_dim))
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    theta = pos * freqs

    sin = torch.sin(theta)
    cos = torch.cos(theta)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)

    if x_rot.ndim == 4:
        sin = sin.unsqueeze(-2)
        cos = cos.unsqueeze(-2)

    x_rotated = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rotated, x_pass], dim=-1)

def apply_2d_rope(x: Tensor, h_ids: Tensor, w_ids: Tensor, rotary_dim: int = 16) -> Tensor:
    """Apply 2D Rotary Position Embedding.
    
    Args:
        x: Input tensor of shape [B, L, H, D].
        h_ids: Height position IDs of shape [B, L].
        w_ids: Width position IDs of shape [B, L].
        rotary_dim: Number of dimensions for rotation.
    
    Returns:
        Tensor with 2D RoPE applied.
    """
    x_h, x_w, x_remain = torch.split(x, [rotary_dim, rotary_dim, x.shape[-1] - 2 * rotary_dim], dim=-1)
    x_h = apply_rope(x_h, h_ids, rotary_dim)
    x_w = apply_rope(x_w, w_ids, rotary_dim)
    return torch.cat([x_h, x_w, x_remain], dim=-1)

def apply_4d_rope(x: Tensor, t_ids: Tensor, tpp_ids: Tensor, h_ids: Tensor, w_ids: Tensor, 
                  t_rotary_dim: int = 8, s_rotary_dim: int = 8) -> Tensor:
    """Apply 4D Rotary Position Embedding.
    
    Args:
        x: Input tensor of shape [B, L, H, D].
        t_ids: Temporal position IDs of shape [B, L].
        tpp_ids: Token-per-patch position IDs of shape [B, L].
        h_ids: Height position IDs of shape [B, L].
        w_ids: Width position IDs of shape [B, L].
        t_rotary_dim: Rotary dimension for temporal and token-per-patch.
        s_rotary_dim: Rotary dimension for spatial dimensions.
    
    Returns:
        Tensor with 4D RoPE applied.
    """
    x_t, x_tpp, x_h, x_w, x_remain = torch.split(
        x, [t_rotary_dim, t_rotary_dim, s_rotary_dim, s_rotary_dim, 
            x.shape[-1] - (2 * t_rotary_dim + 2 * s_rotary_dim)], dim=-1
    )
    x_t = apply_rope(x_t, t_ids, t_rotary_dim)
    x_tpp = apply_rope(x_tpp, tpp_ids, t_rotary_dim)
    x_h = apply_rope(x_h, h_ids, s_rotary_dim)
    x_w = apply_rope(x_w, w_ids, s_rotary_dim)
    return torch.cat([x_t, x_tpp, x_h, x_w, x_remain], dim=-1)

class DecAttention(nn.Module):
    """Decoder Attention module with separate Q, K, V projections and RoPE.
    
    Args:
        dim: Input and output dimension.
        token_per_patch: Number of tokens per patch.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
        qk_norm: Whether to apply normalization to Q and K.
        proj_bias: Whether to use bias in the output projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        norm_layer: Normalization layer (default: nn.LayerNorm).
        use_lora: Whether to include LoRA layers.
        lora_rank: Rank for LoRA adaptation.
        lora_alpha: Scaling factor for LoRA.
    """
    fused_attn: bool

    def __init__(
        self,
        dim: int,
        token_per_patch: int = 2,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16
    ):
        super().__init__()
        assert dim % num_heads == 0, 'Dimension must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.tpp = token_per_patch

        # Projection layers
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Rotary Embedding for tokens
        self.token_rotary_emb = RotaryEmbedding(self.head_dim, theta=10)

    def apply_rope_q(self, x: Tensor) -> Tensor:
        """Apply RoPE to queries or keys."""
        return self.token_rotary_emb.rotate_queries_or_keys(x)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor], cross: bool = False, 
                num_patches: Optional[int] = None, H: Optional[int] = None, 
                W: Optional[int] = None) -> Tensor:
        """Forward pass for decoder attention.
        
        Args:
            x: Tuple of (query, key, value) tensors, each of shape (B, N, C).
            cross: Whether to apply cross-attention (unused in this implementation).
            num_patches: Number of patches (unused).
            H: Height of the input grid (unused).
            W: Width of the input grid (unused).
        
        Returns:
            Output tensor of shape (B, N, C).
        """
        q, k, v = x
        B, N, C = q.shape
        B, T, C = k.shape

        # Project and reshape queries, keys, and values
        q = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply normalization and RoPE
        q, k = self.q_norm(q), self.k_norm(k)
        q = self.apply_rope_q(q)
        k = self.apply_rope_q(k)

        # Compute attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # Project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    """Self-Attention module with LoRA and 4D RoPE.
    
    Args:
        dim: Input and output dimension.
        token_per_patch: Number of tokens per patch.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projection.
        qk_norm: Whether to apply normalization to Q and K.
        proj_bias: Whether to use bias in the output projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
        norm_layer: Normalization layer (default: nn.LayerNorm).
        use_lora: Whether to include LoRA layers.
        lora_rank: Rank for LoRA adaptation.
        lora_alpha: Scaling factor for LoRA.
    """
    fused_attn: bool

    def __init__(
        self,
        dim: int,
        token_per_patch: int = 2,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16
    ):
        super().__init__()
        assert dim % num_heads == 0, 'Dimension must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.use_lora = use_lora
        self.tpp = token_per_patch

        # Main projection layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # LoRA layers
        if use_lora:
            self.qkv_lora_glo = LoRALayer(dim, dim * 3, rank=lora_rank, alpha=lora_alpha)
            self.proj_lora_glo = LoRALayer(dim, dim, rank=lora_rank, alpha=lora_alpha)
            self.qkv_lora_loc = LoRALayer(dim, dim * 3, rank=lora_rank, alpha=lora_alpha)
            self.proj_lora_loc = LoRALayer(dim, dim, rank=lora_rank, alpha=lora_alpha)

    def apply_rope(self, x: Tensor, valid: Optional[Tensor], grid_size: Optional[int], clip_len: Optional[int]) -> Tensor:
        """Apply 4D RoPE to the input tensor.
        
        Args:
            x: Input tensor of shape [B, H, S, D].
            valid: Valid indices for spatial positions.
            grid_size: Size of the spatial grid.
            clip_len: Number of temporal frames.
        
        Returns:
            Tensor with 4D RoPE applied.
        """
        position_ids = build_4d_position_ids(clip_len, valid, grid_size, self.tpp, device=x.device)
        return apply_4d_rope(
            x.transpose(1, 2), position_ids[0], position_ids[1], position_ids[2], position_ids[3],
            t_rotary_dim=8, s_rotary_dim=8
        ).transpose(1, 2)

    def forward(self, x: Tensor, use_lora: int = 0, valid: Optional[Tensor] = None, 
                grid_size: Optional[int] = None, clip_len: Optional[int] = None) -> Tensor:
        """Forward pass with optional LoRA and RoPE.
        
        Args:
            x: Input tensor of shape (B, N, C).
            use_lora: LoRA mode (0: none, 1: global, 3: local).
            valid: Valid indices for spatial positions.
            grid_size: Size of the spatial grid.
            clip_len: Number of temporal frames.
        
        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape

        # Compute Q, K, V with optional LoRA
        if use_lora == 1:
            qkv = self.qkv(x) + self.qkv_lora_glo(x)
        elif use_lora == 3:
            qkv = self.qkv(x) + self.qkv_lora_loc(x)
        else:
            qkv = self.qkv(x) + self.qkv_lora_loc(x)

        # Split and reshape Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if specified
        if use_lora == 1:
            q = self.apply_rope(q, valid, grid_size, clip_len)
            k = self.apply_rope(k, valid, grid_size, clip_len)

        # Compute attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # Project output with optional LoRA
        x = x.transpose(1, 2).reshape(B, N, C)
        if use_lora == 1:
            x = self.proj(x) + self.proj_lora_glo(x)
        elif use_lora == 3:
            x = self.proj(x) + self.proj_lora_loc(x)
        else:
            x = self.proj(x) + self.proj_lora_loc(x)

        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    """LayerScale module for scaling residual connections.
    
    Args:
        dim: Input dimension.
        init_values: Initial scaling factor.
        inplace: Whether to perform scaling in-place.
    """
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply LayerScale to the input."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    """Vision Transformer block with attention and MLP.
    
    Args:
        dim: Input and output dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to input dimension.
        qkv_bias: Whether to use bias in QKV projections.
        qk_norm: Whether to apply normalization to Q and K.
        proj_bias: Whether to use bias in projections.
        proj_drop: Dropout rate for output projection.
        attn_drop: Dropout rate for attention weights.
        init_values: Initial value for LayerScale.
        drop_path: Drop path probability.
        act_layer: Activation function for MLP.
        norm_layer: Normalization layer.
        mlp_layer: MLP layer class.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer,
            bias=proj_bias, drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, use_lora: int = 0, valid: Optional[Tensor] = None, 
                grid_size: Optional[int] = None, clip_len: Optional[int] = None) -> Tensor:
        """Forward pass through the block.
        
        Args:
            x: Input tensor of shape (B, N, C).
            use_lora: LoRA mode (0: none, 1: global, 3: local).
            valid: Valid indices for spatial positions.
            grid_size: Size of the spatial grid.
            clip_len: Number of temporal frames.
        
        Returns:
            Output tensor of shape (B, N, C).
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), use_lora, valid, grid_size, clip_len)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), use_lora)))
        return x



def replace_vit_modules(model, dropout=0., drop_path_rate=0.1, custom_block_class=Block, depth=12):
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

    for i, block in enumerate(model.blocks):
        dim = block.attn.qkv.in_features
        num_heads = block.attn.num_heads
        mlp_ratio = 4.0
        qkv_bias = block.attn.qkv.bias is not None
        attn_drop = block.attn.attn_drop.p
        proj_drop = block.attn.proj_drop.p
        drop_path = dpr[i]
        drop = dropout
        act_layer = type(block.mlp.act)
        norm_layer = type(block.norm1)

        new_block = custom_block_class(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        model.blocks[i] = new_block

    return model
