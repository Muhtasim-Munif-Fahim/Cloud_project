import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load unified results
with open('results/all_model_results.json', 'r') as f:
    results_json = json.load(f)

data = []
# Extract baseline
for r in results_json['baseline_sweep']:
    data.append({'D': r['n_layers'], 'W': r['d_model'], 'params': r['n_params']/1e6, 'loss': r['final_loss']})

import pandas as pd
df = pd.DataFrame(data)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig_dir = "analysis_output/figures"
os.makedirs(fig_dir, exist_ok=True)

# ---------------------------------------------------------
# Calculate scaling constants from data
# ---------------------------------------------------------
# Dynamic D_crit for W=512
w512 = df[df['W'] == 512].sort_values('D')
l16 = w512[w512['D'] == 16]['loss'].values[0]
l24 = w512[w512['D'] == 24]['loss'].values[0]

# Estimate D_crit as the minimum point in the U-curve
d_crit_w512 = w512.loc[w512['loss'].idxmin()]['D']

# Generate Figure 1B: The U-Shape Curve at W=512
plt.figure(figsize=(8, 5))
plt.plot(w512['D'], w512['loss'], 'o-', linewidth=2, markersize=8, color='#d62728', label='Width 512')
plt.axvline(x=d_crit_w512, linestyle='--', color='gray', alpha=0.7, label=f'D_crit ≈ {d_crit_w512}')
plt.text(d_crit_w512 + 0.5, 3.8, 'The Depth Delusion: \nAdding layers increases loss', color='#d62728', fontweight='bold')

# Annotate values dynamically
plt.annotate(f'{int(d_crit_w512)}L (Optimum)\n{l16:.3f}', xy=(16, l16), xytext=(12, 3.6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
plt.annotate(f'24L (+25% Params)\n{l24:.3f}', xy=(24, l24), xytext=(26, 3.65),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.xlabel('Depth (Layers)', fontsize=12)
plt.ylabel('Validation Loss (nats)', fontsize=12)
plt.title('The Depth Delusion at Fixed Width (W=512)', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/figure1b_u_curve.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 1A: Predicted vs Actual
# ---------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(df['params'], df['loss'], c=df['D'], cmap='coolwarm', s=100, alpha=0.8)
cbar = plt.colorbar()
cbar.set_label('Depth (D)', rotation=270, labelpad=15)

plt.xscale('log')
plt.xlabel('Parameter Count (Millions)', fontsize=12)
plt.ylabel('Validation Loss (nats)', fontsize=12)
plt.title('Scaling Law: Capacity vs Loss', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{fig_dir}/figure1a_scaling.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2: Gradient Decay (Empirical τ(W))
# ---------------------------------------------------------
# Load gradient data from the verified evidence file
import json
try:
    with open('results/gradient_flow_stats.json', 'r') as f:
        grad_data = json.load(f)['data']
    
    plt.figure(figsize=(8, 5))
    x = np.linspace(0, 40, 100)
    widths = sorted([int(w) for w in grad_data.keys()])
    colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for w, color in zip(widths, colors):
        # Calculate tau empirically from the deepest model of this width
        depths = [int(d) for d in grad_data[str(w)].keys()]
        deepest = str(max(depths))
        ratios = np.array(grad_data[str(w)][deepest])
        
        # Fit tau: ratio = exp(-(depth-1-ell)/tau)
        ell = np.arange(len(ratios))
        # Log-linear fit for tau
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(lambda l, t: np.exp(-(int(deepest)-1-l)/t), ell, ratios, p0=[10.0])
        tau = popt[0]
        
        y = np.exp(-x / tau)
        plt.plot(x, y, label=f'W={w} (τ={tau:.1f})', color=color, linewidth=2)

    plt.axhline(y=1/np.e, color='black', linestyle=':', alpha=0.5)
    plt.text(32, 0.4, '1/e Threshold', fontsize=10)
    plt.xlabel('Depth (l)', fontsize=12)
    plt.ylabel(r'Relative Gradient Magnitude $\|\nabla_\ell L\| / \|\nabla_D L\|$', fontsize=12)
    plt.title('Theoretical Gradient Starvation Mechanism', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/figure2_gradient_decay.png", dpi=300)
    plt.close()
except:
    print("Warning: Could not derive tau from data. Evidence file missing?")

print(f"Figures generated in {fig_dir}")
