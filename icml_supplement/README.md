# The Depth Delusion: Code and Results Supplement

This supplement contains the anonymized source code and training results for the paper **"The Depth Delusion: Why Transformers Should Be Wider, Not Deeper"**.

## Directory Structure

- `code/`: Contains the JAX/Flax implementation of the Transformer architectures, training loops, and analysis scripts.
    - `src/`: Core implementation files (`model.py`, `trainer.py`, `rigorous_scaling_analysis.py`).
    - `scripts/`: Utility scripts for data preprocessing, plotting, and result unification.
    - `configs/`: YAML configurations for the experimental sweeps.
- `results/`: Contains the raw JSON result files for all 29 models (baseline and large-scale validation).
    - `baseline/`: Results for the 17 baseline models used for scaling law fitting.
    - `1b/`, `3b/`, `7b/`: Results for the validation models at respective scales.
    - `all_model_results.json`: Unified summary of all training metrics and architectural parameters.

## Reproducibility

### 1. Requirements
Ensure you have JAX and the necessary dependencies installed:
```bash
pip install -r code/requirements.txt
```

### 2. Result Analysis
To reproduce the scaling law plots and statistical findings reported in the paper:
```bash
# Navigate to the code directory
cd code
# Generate the scaling law plots and U-curves
python scripts/generate_plots.py
# Run the rigorous scaling analysis
python src/rigorous_scaling_analysis.py
```

### 3. Training
The training scripts require Cloud TPU access and the SlimPajama dataset. Data preprocessing steps are provided in `code/scripts/preprocess_slimpajama.py`. 

To launch a sweep, use the provided shell scripts:
```bash
bash code/scripts/run_experiments.sh
```

## Abstract Summary

We identify a critical failure mode in deep Transformer architectures: **The Depth Delusion**. Our theoretical framework, validated by 29 architectures spanning 17M to 7B parameters, shows that:
1. Optimal depth scales as $D^* \propto C^{0.12}$ while optimal width scales as $W^* \propto C^{0.34}$.
2. Beyond a critical depth $D_{crit} \propto W^{0.44}$, adding layers increases loss despite adding parameters.
3. At the 7B scale, a 32-layer model (6.9B params) outperforms a 64-layer model (7.1B params).
