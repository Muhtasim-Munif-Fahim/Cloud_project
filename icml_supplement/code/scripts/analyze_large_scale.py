#!/usr/bin/env python3
"""
Analyze Large-Scale Experiment Results

This script loads results from all completed experiments and:
1. Plots U-curves at each scale (1B, 3B, 7B)
2. Compares target results vs experimental results
3. Validates success criteria
4. Generates figures for paper submission
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import yaml

def load_submission_results(config_path: str) -> Dict[str, Dict]:
    """Load target results from YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results_map = {}
    for scale in ['baseline', '1b', '3b', '7b']:
        scale_key = f"{scale}_experiments"
        if scale_key in config:
            results_map[scale] = {}
            for model in config[scale_key]['models']:
                results_map[scale][model['name']] = {
                    'layers': model['layers'] if 'layers' in model else model.get('n_layers'),
                    'loss': model.get('loss', None)
                }
    return results_map


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result JSON files from directory."""
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob('**/*_result.json'):
        with open(json_file) as f:
            data = json.load(f)
            results[data['model_name']] = data
    
    return results


def validate_success_criteria(results: Dict) -> Dict[str, bool]:
    """Check if success criteria are met."""
    criteria = {}
    
    # 1B criteria
    if '1B_medium' in results and '1B_vdeep' in results:
        medium_loss = results['1B_medium']['final_loss']
        vdeep_loss = results['1B_vdeep']['final_loss']
        criteria['1B_medium_optimal'] = all(
            results[m]['final_loss'] >= medium_loss
            for m in results if m.startswith('1B_')
        )
        criteria['1B_vdeep_degradation'] = (vdeep_loss - medium_loss) >= 0.05
    
    # 3B criteria
    if '3B_medium' in results and '3B_extreme' in results:
        medium_loss = results['3B_medium']['final_loss']
        extreme_loss = results['3B_extreme']['final_loss']
        shallow_loss = results.get('3B_shallow', {}).get('final_loss', float('inf'))
        criteria['3B_medium_optimal'] = medium_loss <= shallow_loss
        criteria['3B_extreme_degradation'] = extreme_loss > shallow_loss
    
    # 7B criteria (HEADLINE)
    if '7B_optimal' in results and '7B_deep' in results:
        optimal_loss = results['7B_optimal']['final_loss']
        deep_loss = results['7B_deep']['final_loss']
        criteria['7B_depth_delusion'] = deep_loss > optimal_loss
        criteria['7B_significant_gap'] = (deep_loss - optimal_loss) >= 0.10
    
    return criteria


def plot_u_curves(results: Dict, output_dir: str, submission_results: Dict):
    """Generate U-curve plots for each scale."""
    scales = ['baseline', '1b', '3b', '7b']
    titles = ['Baseline Sweep Result', '1B Scale Result', '3B Scale Result', '7B Scale Result (Headline)']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for ax, scale, title in zip(axes, scales, titles):
        # Get results for this scale
        scale_results = {k: v for k, v in results.items() if k.lower().startswith(scale.replace('b', 'b_'))}
        
        if not scale_results:
            ax.set_title(f'{title} (No data)')
            continue
        
        # Plot
        layers = []
        losses = []
        target_losses = []
        
        for name, data in sorted(scale_results.items(), key=lambda x: x[1]['config']['n_layers']):
            n_layers = data['config']['n_layers']
            layers.append(n_layers)
            losses.append(data['final_loss'])
            
            if name in submission_results.get(scale, {}):
                val = submission_results[scale][name]['loss']
                target_losses.append(val if val is not None else None)
            else:
                target_losses.append(None)
        
        ax.plot(layers, losses, 'o-', label='Experimental Result', linewidth=2, markersize=10)
        
        # Plot target
        target_layers = [l for l, e in zip(layers, target_losses) if e is not None]
        target_plot_losses = [e for e in target_losses if e is not None]
        if target_plot_losses:
            ax.plot(target_layers, target_plot_losses, 's--', label='Target Result', alpha=0.7)
        
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Result Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate optimal
        if losses:
            min_idx = np.argmin(losses)
            ax.annotate(
                f'Optimal: {layers[min_idx]}L',
                xy=(layers[min_idx], losses[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='green'
            )
    
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / 'large_scale_u_curves.pdf', bbox_inches='tight')
    fig.savefig(output_path / 'large_scale_u_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved U-curve plots to {output_path}")


def generate_summary_table(results: Dict, submission_results: Dict) -> str:
    """Generate markdown summary table."""
    lines = [
        "# Large-Scale Experiment Results",
        "",
        "| Model | Layers | Width | Result Params | Target Result | Experimental Result | Δ |",
        "|-------|--------|-------|---------------|---------------|---------------------|---|",
    ]
    
    for scale in ['baseline', '1b', '3b', '7b']:
        scale_prefix = scale.upper() + '_'
        scale_results = {k: v for k, v in results.items() if k.startswith(scale_prefix)}
        
        for name, data in sorted(scale_results.items(), key=lambda x: x[1]['config']['n_layers']):
            n_layers = data['config']['n_layers']
            d_model = data['config']['d_model']
            n_params = data['n_params']
            actual_loss = data['final_loss']
            
            target = submission_results.get(scale, {}).get(name, {}).get('loss', None)
            if target is not None:
                delta = f"{actual_loss - target:+.3f}"
            else:
                delta = '-'
            
            lines.append(f"| {name} | {n_layers} | {d_model} | {n_params/1e9:.2f}B | {target} | {actual_loss:.3f} | {delta} |")
    
    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--config-path', type=str, default='configs/large_scale_experiments.yaml')
    parser.add_argument('--output-dir', type=str, default='./analysis_output')
    args = parser.parse_args()
    
    print("Loading submission target results...")
    submission_results = load_submission_results(args.config_path)
    
    print("Loading experimental results...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} model results")
    
    print("\nValidating success criteria...")
    criteria = validate_success_criteria(results)
    for name, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print("\nGenerating plots...")
    plot_u_curves(results, args.output_dir, submission_results)
    
    print("\nGenerating summary table...")
    summary = generate_summary_table(results, submission_results)
    summary_path = Path(args.output_dir) / 'results_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved summary to {summary_path}")
    

if __name__ == '__main__':
    main()
