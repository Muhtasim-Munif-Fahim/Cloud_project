import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import argparse

def preprocess(output_dir, num_shards=1024):
    """Convert SlimPajama into shuffled .npy chunks for TPU streaming."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Loading dataset...")
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    
    buffer = []
    chunk_idx = 0
    
    for i, sample in enumerate(dataset):
        tokens = tokenizer(sample['text'])['input_ids'] + [tokenizer.eos_token_id]
        buffer.extend(tokens)
        
        # Save chunk every 100M tokens
        if len(buffer) > 100_000_000:
            arr = np.array(buffer[:100_000_000], dtype=np.uint16)
            buffer = buffer[100_000_000:]
            
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.npy")
            np.save(chunk_path, arr)
            print(f"Saved {chunk_path}")
            chunk_idx += 1
            
            if chunk_idx >= num_shards:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shards", type=int, default=100)
    args = parser.parse_args()
    preprocess(args.output_dir, args.shards)
