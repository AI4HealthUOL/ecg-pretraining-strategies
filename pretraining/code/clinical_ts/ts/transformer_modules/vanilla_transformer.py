__all__ = ['TransformerModel']

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'"""
    
    def __init__(self, d_model, max_len=8192, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Handle odd d_model
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) from RoFormer"""
    
    def __init__(self, d_model, max_len=8192, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max_len positions
        self._update_cos_sin_cache(max_len)
    
    def _update_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, d_model)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)
    
    def forward(self, x, seq_len=None):
        """
        Returns cos and sin for RoPE application
        Args:
            x: tensor to get device/dtype from
            seq_len: sequence length
        Returns:
            cos, sin: (1, seq_len, d_model)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :]
        )


def rotate_half(x):
    """Rotate half the hidden dims of the input for RoPE"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors"""
    # q, k: (B, n_heads, L, head_dim)
    # cos, sin: (1, L, head_dim) -> need to reshape for heads
    cos = cos.unsqueeze(1)  # (1, 1, L, head_dim)
    sin = sin.unsqueeze(1)  # (1, 1, L, head_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE support"""
    
    def __init__(self, d_model, n_heads, dropout=0.1, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, attn_mask=None, rope_cos=None, rope_sin=None):
        """
        Args:
            x: (B, L, D)
            attn_mask: (L, L) or (B, L, L), True means mask out
            rope_cos, rope_sin: (1, L, D) for RoPE
        Returns:
            (B, L, D)
        """
        B, L, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope and rope_cos is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, n_heads, L, L)
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L, L)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, D)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, prenorm=True, 
                 use_rope=False, activation='gelu'):
        super().__init__()
        
        self.prenorm = prenorm
        self.use_rope = use_rope
        
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, rope_cos=None, rope_sin=None):
        """
        Args:
            x: (B, L, D)
            attn_mask: optional attention mask
            rope_cos, rope_sin: for RoPE
        Returns:
            (B, L, D)
        """
        if self.prenorm:
            # Pre-norm architecture (GPT-2 style)
            x = x + self.dropout1(self.attn(self.norm1(x), attn_mask, rope_cos, rope_sin))
            x = x + self.dropout2(self.ff(self.norm2(x)))
        else:
            # Post-norm architecture (original Transformer)
            x = self.norm1(x + self.dropout1(self.attn(x, attn_mask, rope_cos, rope_sin)))
            x = self.norm2(x + self.dropout2(self.ff(x)))
        
        return x


class TransformerModel(nn.Module):
    """
    Transformer encoder model for sequence modeling.
    
    Supports:
    - Sinusoidal positional encoding
    - Rotary Position Embedding (RoPE)
    - Causal (unidirectional) and bidirectional attention
    - Pre-norm and post-norm architectures
    """
    
    def __init__(
        self,
        d_input,           # Input dimension (None to disable encoder)
        d_output,          # Output dimension (None to disable decoder)
        d_model=512,       # Model hidden dimension
        n_heads=8,         # Number of attention heads
        n_layers=6,        # Number of transformer blocks
        d_ff=None,         # Feed-forward dimension (default: 4 * d_model)
        dropout=0.1,       # Dropout rate
        prenorm=True,      # Use pre-norm (True) or post-norm (False)
        max_len=8192,      # Maximum sequence length
        causal=False,      # Use causal (unidirectional) attention
        pos_encoding='sinusoidal',  # 'sinusoidal' or 'rope'
        activation='gelu', # Activation function: 'gelu', 'relu', 'silu'
        pooling=False,     # Global average pooling over sequence
    ):
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        self.pooling = pooling
        self.pos_encoding_type = pos_encoding
        self.prenorm = prenorm
        
        # Input projection
        if d_input is None or d_input == d_model:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Linear(d_input, d_model)
        
        # Positional encoding
        use_rope = (pos_encoding == 'rope')
        if use_rope:
            self.pos_encoding = RotaryPositionalEncoding(d_model // n_heads, max_len)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                prenorm=prenorm,
                use_rope=use_rope,
                activation=activation
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(d_model) if prenorm else nn.Identity()
        
        # Output projection
        if d_output is None:
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def forward(self, x):
        """
        Args:
            x: (B, L, d_input) input tensor
        Returns:
            (B, L, d_model) or (B, d_model) if pooling
        """
        B, L, _ = x.shape
        
        # Input projection
        x = self.encoder(x)
        
        # Prepare positional encoding and attention mask
        rope_cos, rope_sin = None, None
        
        if self.pos_encoding_type == 'rope':
            rope_cos, rope_sin = self.pos_encoding(x, L)
        else:
            x = self.pos_encoding(x)
        
        # Causal mask for unidirectional attention
        attn_mask = None
        if self.causal:
            attn_mask = self._generate_causal_mask(L, x.device)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, attn_mask, rope_cos, rope_sin)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Pooling
        if self.pooling:
            x = x.mean(dim=1)  # (B, d_model)
        
        # Output projection
        if self.decoder is not None:
            x = self.decoder(x)
        
        return x