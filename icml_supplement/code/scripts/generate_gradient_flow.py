import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    256: '#2ecc71',   # Green
    512: '#3498db',   # Blue
    1024: '#9b59b6',  # Purple
    1536: '#e74c3c',  # Red
}

DATA_FILE = "../../results/gradient_flow_stats.json"

def load_gradient_data():
    # Determine base path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, DATA_FILE))
    
    if not os.path.exists(data_path):
        print(f"Warning: Could not find {data_path}. Falling back to default data.")
        return {}
        
    try:
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert keys from strings to ints (JSON keys are always strings)
        formatted_data = {}
        for w_str, depths in raw_data['data'].items():
            w = int(w_str)
            formatted_data[w] = {int(d): vals for d, vals in depths.items()}
        return formatted_data
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return {}

GRADIENT_DATA = load_gradient_data()


def exponential_decay(ell, tau):
    """Gradient decay: ||∇_ℓ|| / ||∇_D|| = exp(-(D-ℓ)/τ)"""
    D = len(ell)
    return np.exp(-(D - 1 - ell) / tau)


def fit_tau_for_width(width_data):
    """Fit τ from gradient decay curves for a given width."""
    all_layers = []
    all_ratios = []
    
    for depth, ratios in width_data.items():
        layers = np.arange(depth)
        all_layers.extend(layers)
        all_ratios.extend(ratios)
    
    # Use the deepest model for best τ estimate
    deepest = max(width_data.keys())
    layers = np.arange(deepest)
    ratios = np.array(width_data[deepest])
    
    # Fit exponential decay
    try:
        popt, _ = curve_fit(
            lambda ell, tau: np.exp(-(deepest - 1 - ell) / tau),
            layers, ratios,
            p0=[10.0],
            bounds=([1.0], [50.0])
        )
        return popt[0]
    except:
        return None


def fit_scaling_law(widths, taus):
    """Fit τ(W) = c * f(W) for different functional forms."""
    W = np.array(widths)
    T = np.array(taus)
    
    results = {}
    
    # Form 1: τ = c * log(W)
    def log_form(w, c):
        return c * np.log(w)
    
    try:
        popt, _ = curve_fit(log_form, W, T)
        pred = log_form(W, *popt)
        r2 = 1 - np.sum((T - pred)**2) / np.sum((T - np.mean(T))**2)
        results['log W'] = {'c': popt[0], 'r2': r2, 'pred': pred}
    except:
        pass
    
    # Form 2: τ = c * W^a (power law)
    def power_form(w, c, a):
        return c * np.power(w, a)
    
    try:
        popt, _ = curve_fit(power_form, W, T, p0=[1.0, 0.2], bounds=([0, 0], [100, 1]))
        pred = power_form(W, *popt)
        r2 = 1 - np.sum((T - pred)**2) / np.sum((T - np.mean(T))**2)
        results['W^a'] = {'c': popt[0], 'a': popt[1], 'r2': r2, 'pred': pred}
    except:
        pass
    
    # Form 3: τ = c * sqrt(W) * log(W)
    def sqrt_log_form(w, c):
        return c * np.sqrt(w) * np.log(w)
    
    try:
        popt, _ = curve_fit(sqrt_log_form, W, T)
        pred = sqrt_log_form(W, *popt)
        r2 = 1 - np.sum((T - pred)**2) / np.sum((T - np.mean(T))**2)
        results['sqrt(W)*log(W)'] = {'c': popt[0], 'r2': r2, 'pred': pred}
    except:
        pass
    
    return results


def generate_figure():
    """Generate Figure 3: Gradient Flow Validation."""
    
    fig = plt.figure(figsize=(10, 4))
    
    # ========== Panel A: Gradient Decay Curves ==========
    ax1 = fig.add_subplot(1, 2, 1)
    
    for width in [256, 512, 1024, 1536]:
        # Use 16-layer model (or deepest available)
        depths = list(GRADIENT_DATA[width].keys())
        deepest = max(depths)
        ratios = GRADIENT_DATA[width][deepest]
        
        layers = np.arange(deepest)
        ax1.plot(layers, ratios, 'o-', color=COLORS[width], 
                 label=f'W={width}', markersize=4, linewidth=1.5)
        
        # Fit and plot exponential
        tau = fit_tau_for_width(GRADIENT_DATA[width])
        if tau:
            x_fit = np.linspace(0, deepest-1, 100)
            y_fit = np.exp(-(deepest - 1 - x_fit) / tau)
            ax1.plot(x_fit, y_fit, '--', color=COLORS[width], alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Layer $\\ell$ (from input)')
    ax1.set_ylabel('Relative Gradient $\\|\\nabla_\\ell L\\| / \\|\\nabla_D L\\|$')
    ax1.set_title('(a) Gradient Decay Across Layers')
    ax1.legend(loc='lower left')
    ax1.set_ylim([0.2, 1.05])
    ax1.grid(True, alpha=0.3)
    
    # Add 1/e threshold line
    ax1.axhline(y=1/np.e, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.text(0.5, 1/np.e + 0.03, '$1/e$ threshold', fontsize=8, color='gray')
    
    # ========== Panel B: τ(W) Scaling ==========
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Fit τ for each width
    widths = []
    taus = []
    for width in sorted(GRADIENT_DATA.keys()):
        tau = fit_tau_for_width(GRADIENT_DATA[width])
        if tau:
            widths.append(width)
            taus.append(tau)
    
    # Plot measured τ values
    ax2.scatter(widths, taus, s=100, c=[COLORS[w] for w in widths], 
                edgecolors='black', linewidths=1, zorder=5)
    
    # Fit different functional forms
    results = fit_scaling_law(widths, taus)
    
    W_dense = np.linspace(200, 1700, 100)
    
    # Plot log W fit (best)
    if 'log W' in results:
        c = results['log W']['c']
        ax2.plot(W_dense, c * np.log(W_dense), 'b-', linewidth=2, 
                 label=f'$\\tau = {c:.2f} \\log W$ ($R^2$={results["log W"]["r2"]:.3f})')
    
    # Plot power law fit
    if 'W^a' in results:
        c, a = results['W^a']['c'], results['W^a']['a']
        ax2.plot(W_dense, c * np.power(W_dense, a), 'g--', linewidth=1.5, alpha=0.7,
                 label=f'$\\tau = {c:.2f} W^{{{a:.2f}}}$ ($R^2$={results["W^a"]["r2"]:.3f})')
    
    ax2.set_xlabel('Width $W$')
    ax2.set_ylabel('Gradient Persistence $\\tau(W)$')
    ax2.set_title('(b) Persistence Length Scaling')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for w, t in zip(widths, taus):
        ax2.annotate(f'W={w}', (w, t), xytext=(5, 5), textcoords='offset points', 
                     fontsize=8, color=COLORS[w])
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "analysis_output/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure3_gradient_flow.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Print summary statistics
    print("\n=== Gradient Flow Validation Summary ===")
    print(f"Widths analyzed: {widths}")
    print(f"Fitted τ values: {[f'{t:.2f}' for t in taus]}")
    print("\nFunctional form comparison:")
    for form, res in results.items():
        print(f"  {form}: R² = {res['r2']:.4f}")
    
    best_form = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest fit: {best_form[0]} with R² = {best_form[1]['r2']:.4f}")
    
    return results


if __name__ == '__main__':
    results = generate_figure()
