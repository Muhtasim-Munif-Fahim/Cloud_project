"""
Configurable Transformer Model for Architecture Scaling Laws Study

This module implements a flexible transformer architecture where key
hyperparameters (width, depth, attention ratio) can be varied systematically.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
import dataclasses


@dataclasses.dataclass
class TransformerConfig:
    """Configuration for transformer architecture experiments."""
    vocab_size: int = 50257
    max_seq_len: int = 512
    
    # Architecture parameters to vary
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048  # MLP hidden dimension
    
    # Derived ratios (for analysis)
    @property
    def width_depth_ratio(self) -> float:
        return self.d_model / self.n_layers
    
    @property
    def ff_model_ratio(self) -> float:
        return self.d_ff / self.d_model
    
    @property
    def n_params(self) -> int:
        """Approximate parameter count."""
        embed = self.vocab_size * self.d_model
        attn = self.n_layers * 4 * self.d_model * self.d_model  # Q, K, V, O
        ff = self.n_layers * 2 * self.d_model * self.d_ff
        return embed + attn + ff
    
    dropout: float = 0.0  # Disabled for scaling law experiments
    dtype: jnp.dtype = jnp.bfloat16


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with configurable dimensions."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        cfg = self.config
        head_dim = cfg.d_model // cfg.n_heads
        
        # QKV projections
        qkv = nn.Dense(3 * cfg.d_model, dtype=cfg.dtype, name='qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        batch, seq_len, _ = x.shape
        q = q.reshape(batch, seq_len, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # Use float32 for attention weights for numerical stability
        scale = jnp.sqrt(head_dim).astype(jnp.float32)
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        
        attn_weights = jnp.matmul(q_f32, k_f32.transpose(0, 1, 3, 2)) / scale
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(cfg.dtype)
        attn_out = jnp.matmul(attn_weights, v)
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, cfg.d_model)
        
        # Output projection
        return nn.Dense(cfg.d_model, dtype=cfg.dtype, name='out')(attn_out)


class FeedForward(nn.Module):
    """Feed-forward network with configurable expansion ratio."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        x = nn.Dense(cfg.d_ff, dtype=cfg.dtype)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(cfg.d_model, dtype=cfg.dtype)(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        cfg = self.config
        
        # Self-attention with residual
        x = x + MultiHeadAttention(cfg)(nn.LayerNorm(dtype=cfg.dtype)(x), mask)
        
        # Feed-forward with residual
        x = x + FeedForward(cfg)(nn.LayerNorm(dtype=cfg.dtype)(x))
        
        return x


class Transformer(nn.Module):
    """Full transformer for language modeling."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        batch, seq_len = input_ids.shape
        
        # Token + positional embeddings
        x = nn.Embed(cfg.vocab_size, cfg.d_model, dtype=cfg.dtype)(input_ids)
        pos_embed = self.param('pos_embed', 
                               nn.initializers.normal(0.02),
                               (cfg.max_seq_len, cfg.d_model))
        x = x + pos_embed[:seq_len].astype(cfg.dtype)
        
        # Causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        
        # Transformer blocks
        for _ in range(cfg.n_layers):
            x = TransformerBlock(cfg)(x, mask)
        
        # Final layer norm
        x = nn.LayerNorm(dtype=cfg.dtype)(x).astype(jnp.float32)
        
        # LM head (projections to float32 for softmax stability)
        logits = nn.Dense(cfg.vocab_size, dtype=jnp.float32)(x)
        
        return logits


def create_model(config: TransformerConfig) -> Tuple[Transformer, dict]:
    """Initialize model and return with parameter count."""
    model = Transformer(config)
    
    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    
    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
    
    return model, {'n_params': n_params, 'config': config}


# Architecture grid for experiments
def generate_architecture_grid():
    """
    Generate systematic grid of ~500 architectures to study.
    
    Covers:
    - Layers: 2, 4, 6, 8, 12, 16, 24, 32 (8 values) - tests "curse of depth"
    - Width: 128, 256, 384, 512, 768, 1024, 1536 (7 values) - covers small to medium
    - FF Ratio: 2, 4, 6, 8 (4 values) - standard to wide MLPs
    - Head Dim: 32, 64, 128 (3 values) - attention granularity
    """
    configs = []
    
    # Systematic grid
    layer_values = [2, 4, 6, 8, 12, 16, 24, 32]
    width_values = [128, 256, 384, 512, 768, 1024, 1536]
    ff_ratios = [2, 4, 6, 8]
    head_dims = [32, 64, 128]
    
    for n_layers in layer_values:
        for d_model in width_values:
            for ff_ratio in ff_ratios:
                for head_dim in head_dims:
                    # Ensure d_model is divisible by head_dim
                    if d_model % head_dim != 0:
                        continue
                    
                    n_heads = d_model // head_dim
                    
                    cfg = TransformerConfig(
                        n_layers=n_layers,
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_model * ff_ratio,
                    )
                    
                    # Filter to reasonable parameter range: 5M - 500M
                    if 5_000_000 <= cfg.n_params <= 500_000_000:
                        configs.append(cfg)
    
    # Sort by parameter count for systematic training
    configs.sort(key=lambda c: c.n_params)
    
    return configs


if __name__ == '__main__':
    # Test model creation
    cfg = TransformerConfig(n_layers=6, d_model=512, d_ff=2048)
    model, info = create_model(cfg)
    print(f"Created model with {info['n_params']:,} parameters")
    print(f"Width/Depth ratio: {cfg.width_depth_ratio:.2f}")
    print(f"FF/Model ratio: {cfg.ff_model_ratio:.2f}")
    
    # Generate architecture grid
    grid = generate_architecture_grid()
    print(f"\nGenerated {len(grid)} architecture configurations")
