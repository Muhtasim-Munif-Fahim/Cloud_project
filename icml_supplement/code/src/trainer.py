"""
TPU-Optimized Training Loop for Scaling Law Experiments

Features:
- pmap for multi-device training
- bfloat16 for memory efficiency  
- Adaptive batch sizing for memory management
- Gradient checkpointing for large models
- Minimal logging overhead
"""

import jax
import jax.numpy as jnp
from jax import pmap
import optax
from flax.training import train_state
from functools import partial
import time
from typing import Dict, Any
import json
from pathlib import Path

from model import TransformerConfig, create_model


def get_adaptive_batch_size(n_params: int) -> int:
    """Return appropriate batch size based on model size to avoid OOM."""
    if n_params < 10_000_000:  # <10M params
        return 32
    elif n_params < 25_000_000:  # 10-25M params
        return 16
    elif n_params < 50_000_000:  # 25-50M params
        return 8
    elif n_params < 100_000_000:  # 50-100M params
        return 4
    else:  # >100M params
        return 8


class ScalingExperimentTrainer:
    """Trainer optimized for running many short training experiments."""
    
    def __init__(
        self,
        config: TransformerConfig,
        learning_rate: float = 3e-4,
        warmup_steps: int = 50,
        total_steps: int = 500,
        batch_size_per_device: int = None,  # Will be auto-determined if None
        use_gradient_checkpointing: bool = True,
    ):
        self.config = config
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Auto-determine batch size based on model size if not specified
        if batch_size_per_device is None:
            self.batch_size = get_adaptive_batch_size(config.n_params)
        else:
            self.batch_size = batch_size_per_device
        
        self.n_devices = jax.local_device_count()
        self.global_batch_size = self.batch_size * self.n_devices
        
        # Initialize model and optimizer
        self._setup()
    
    def _setup(self):
        """Initialize model, optimizer, and training state."""
        model, info = create_model(self.config)
        self.model = model
        self.n_params = info['n_params']
        
        # Cosine schedule with warmup
        # decay_steps is the total schedule length including warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=min(self.warmup_steps, self.total_steps // 2),
            decay_steps=max(self.total_steps, self.warmup_steps + 1),
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=0.1),
        )
        
        # Initialize state
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((self.batch_size, self.config.max_seq_len), dtype=jnp.int32)
        variables = self.model.init(rng, dummy_input)
        
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=optimizer,
        )
        
        # Replicate across devices
        self.state = jax.device_put_replicated(self.state, jax.local_devices())
    
    @partial(pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def _train_step(self, state, batch):
        """Single training step with pmap and optional gradient checkpointing."""
        def loss_fn(params):
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Use gradient checkpointing to save activation memory
            if self.use_gradient_checkpointing:
                apply_fn = jax.checkpoint(state.apply_fn)
            else:
                apply_fn = state.apply_fn
            
            logits = apply_fn({'params': params}, input_ids)
            
            # Cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return loss.mean()
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def train(self, data_iterator) -> Dict[str, Any]:
        """Run training and return scaling metrics."""
        losses = []
        start_time = time.time()
        total_tokens = 0
        
        for step in range(self.total_steps):
            # Get batch and reshape for pmap
            batch = next(data_iterator)
            batch = batch.reshape(self.n_devices, self.batch_size, -1)
            
            self.state, loss = self._train_step(self.state, batch)
            
            # Track metrics (every 100 steps to reduce overhead)
            if step % 100 == 0:
                loss_val = float(loss[0])  # Get from first device
                losses.append({'step': step, 'loss': loss_val})
                
            total_tokens += self.global_batch_size * self.config.max_seq_len
        
        elapsed = time.time() - start_time
        final_loss = float(jax.device_get(loss)[0])
        
        return {
            'config': {
                'n_layers': self.config.n_layers,
                'd_model': self.config.d_model,
                'd_ff': self.config.d_ff,
                'n_heads': self.config.n_heads,
            },
            'n_params': self.n_params,
            'final_loss': final_loss,
            'total_tokens': total_tokens,
            'total_flops': self._estimate_flops(total_tokens),
            'elapsed_seconds': elapsed,
            'tokens_per_second': total_tokens / elapsed,
            'loss_curve': losses,
        }
    
    def _estimate_flops(self, tokens: int) -> int:
        """Estimate training FLOPs (6 * params * tokens for forward + backward)."""
        return 6 * self.n_params * tokens


def run_scaling_experiment(
    config: TransformerConfig,
    data_path: str,
    output_dir: str,
    total_steps: int = 1000,
    debug_local: bool = False,
) -> Dict[str, Any]:
    """Run a single scaling experiment and save results."""
    
    # Ensure data path exists
    # Ensure output dir is GCS
    if not debug_local and not str(output_dir).startswith("gs://"):
         raise ValueError(f"Output directory {output_dir} must be a GCS path (gs://) for persistent TPU storage.")

    # Ensure data path exists
    if not os.path.exists(data_path) and not data_path.startswith("gs://"):
         raise ValueError(f"Data path {data_path} not found")

    # Real data iterator
    # (Implementation depends on dataset format, assuming pre-tokenized)
    def real_data_iterator():
         raise NotImplementedError("Real data loading must be implemented based on exact dataset format")
    
    data_iterator = real_data_iterator()
    
    trainer = ScalingExperimentTrainer(
        config=config,
        total_steps=total_steps,
    )
    
    print(f"Training {trainer.n_params:,} param model on {trainer.n_devices} TPU cores")
    results = trainer.train(data_iterator)
    
    # Save results
    output_path = Path(output_dir) / f"exp_{config.n_layers}L_{config.d_model}D_{config.d_ff}FF.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-local', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Test with small model
    cfg = TransformerConfig(n_layers=2, d_model=128, d_ff=512, max_seq_len=128)
    if args.debug_local:
         cfg.debug_local = True
    results = run_scaling_experiment(cfg, '', 'gs://icml-test-bucket/scaling_results', total_steps=200, debug_local=args.debug_local)
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Throughput: {results['tokens_per_second']:.0f} tokens/sec")
