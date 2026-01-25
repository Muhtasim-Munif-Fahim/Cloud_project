# Helper to load tokenized data from GCS or local disk.
# Handles the .npy chunk format used by the preprocessing script.

import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, List
from functools import partial
import json
from pathlib import Path
import io


# HuggingFace streaming support
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

# TF support for GCS
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def load_npy_from_gcs(gcs_path: str) -> np.ndarray:
    """Load .npy file directly from GCS using tf.io.gfile."""
    if HAS_TF:
        with tf.io.gfile.GFile(gcs_path, 'rb') as f:
            return np.load(io.BytesIO(f.read()))
    else:
        raise RuntimeError("TensorFlow required for GCS streaming")


def list_gcs_chunks(gcs_prefix: str) -> List[str]:
    """List all .npy chunks in a GCS bucket prefix."""
    if HAS_TF:
        pattern = gcs_prefix.rstrip('/') + '/chunk_*.npy'
        return sorted(tf.io.gfile.glob(pattern))
    else:
        return []


class SlimPajamaLoader:
    """
    Efficient data loader for SlimPajama on TPU.
    
    Supports:
    1. GCS streaming from pre-tokenized .npy chunks
    2. Direct HuggingFace streaming (on-the-fly tokenization)
    3. Local disk loading
    """
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 512,
        batch_size_per_device: int = 32,
        vocab_size: int = 50257,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.batch_size = batch_size_per_device
        self.vocab_size = vocab_size
        self.n_devices = jax.local_device_count()
        self.global_batch_size = self.batch_size * self.n_devices
        self.rng = np.random.default_rng(seed)
        
        # Load chunk index or setup HF
        self._load_index()
    
    def _load_index(self):
        """Load list of available data chunks or setup streaming."""
        self.is_gcs = self.data_path.startswith('gs://')
        self.is_hf = self.data_path.startswith('hf://') or 'slimpajama' in self.data_path.lower()
        
        if self.is_hf:
            self.n_chunks = 1  # Virtual infinite stream
            print(f"Configured direct HuggingFace streaming from {self.data_path}")
        elif self.is_gcs:
            self.chunks = list_gcs_chunks(self.data_path)
            self.n_chunks = len(self.chunks)
            print(f"Loaded {self.n_chunks} data chunks (GCS streaming)")
        else:
            # Local mode
            index_path = Path(self.data_path) / "index.json"
            if index_path.exists():
                with open(index_path) as f:
                    self.chunks = json.load(f)['chunks']
            else:
                self.chunks = sorted([str(p) for p in Path(self.data_path).glob("*.npy")])
            self.n_chunks = len(self.chunks)
            print(f"Loaded {self.n_chunks} data chunks (local)")
    
    def __iter__(self) -> Iterator[jnp.ndarray]:
        """Iterate over batches."""
        if self.is_hf:
            yield from self._huggingface_iterator()
            return

        while True:
            # Shuffle chunks each epoch
            chunk_order = self.rng.permutation(self.n_chunks) if self.n_chunks > 0 else []
            
            for chunk_idx in chunk_order:
                try:
                    # Load chunk (GCS or local)
                    chunk_path = self.chunks[chunk_idx]
                    if self.is_gcs:
                        data = load_npy_from_gcs(chunk_path)
                    else:
                        data = np.load(chunk_path)
                    
                    # Shuffle within chunk
                    self.rng.shuffle(data)
                    
                    # Yield batches
                    for i in range(0, len(data) - self.global_batch_size + 1, self.global_batch_size):
                        batch = data[i:i + self.global_batch_size]
                        yield jnp.array(batch)
                        
                except Exception as e:
                    print(f"Error loading chunk {chunk_idx}: {e}")
                    continue
            
            # Ensure we have data
            if self.n_chunks == 0:
                raise RuntimeError("No data chunks found and HuggingFace streaming not configured. Cannot train.")
    
    def _huggingface_iterator(self) -> Iterator[jnp.ndarray]:
        """Stream and tokenize directly from HuggingFace."""
        if not HAS_HF:
            raise RuntimeError("HuggingFace datasets/transformers required for direct streaming")
            
        print("Initializing HuggingFace stream...")
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        buffer = []
        for sample in dataset:
            text = sample['text']
            tokens = tokenizer(text, truncation=False, return_tensors='np')['input_ids'][0]
            buffer.extend(tokens)
            
            # Yield full batches
            while len(buffer) >= self.global_batch_size * (self.seq_len + 1):
                flat_batch = np.array(buffer[:self.global_batch_size * (self.seq_len + 1)])
                buffer = buffer[self.global_batch_size * (self.seq_len + 1):]
                
                batch = flat_batch.reshape(self.global_batch_size, self.seq_len + 1)
                yield jnp.array(batch)


    
    def get_batch_for_pmap(self) -> jnp.ndarray:
        """Get a single batch reshaped for pmap [n_devices, batch_per_device, seq_len]."""
        batch = next(iter(self))
        return batch.reshape(self.n_devices, self.batch_size, -1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/scaling_data', help='Data directory')
    parser.add_argument('--test-loader', action='store_true', help='Test data loader')
    args = parser.parse_args()
    
    if args.test_loader:
        loader = SlimPajamaLoader(args.output_dir)
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")
        print(f"Sample tokens: {batch[0, :10]}")
