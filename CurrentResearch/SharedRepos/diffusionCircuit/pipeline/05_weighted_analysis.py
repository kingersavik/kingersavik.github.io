#!/usr/bin/env python3
"""
SCRIPT 05: Weighted Connectome Analysis
=======================================

Extracts coupling weights from SBTG mu_hat matrices and compares to
structural connectome weights (synapse counts).

Key analyses:
- Uses |mu_hat[i,j]| as functional connection strength
- Correlates with structural synapse counts
- Analyzes weight distributions and relationships
- Computes Spearman/Pearson correlations on shared edges

USAGE:
    python pipeline/05_weighted_analysis.py

INPUTS:
    - results/comprehensive_sbtg/models/{stim}/{config}/result.npz
    - results/intermediate/connectome/A_struct.npy

OUTPUTS:
    - results/comprehensive_sbtg/weighted_analysis/weight_correlations.csv
    - results/comprehensive_sbtg/weighted_analysis/weighted_edge_details.json
    - results/comprehensive_sbtg/weighted_analysis/figures/*.png

Expected runtime: ~2-3 minutes
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.io import loadmat

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_sbtg"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
OUTPUT_DIR = RESULTS_DIR / "weighted_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# WEIGHT EXTRACTION
# ============================================================================

def extract_wparam_weights(result_file: Path) -> np.ndarray:
    """
    Extract W_param/mu_hat coupling matrix from SBTG result file.
    
    Args:
        result_file: Path to result.npz
    
    Returns:
        mu_hat: Coupling weight matrix (may be None if not available)
    """
    data = np.load(result_file)
    
    # Check for mu_hat (SBTG coupling estimates)
    if 'mu_hat' in data:
        return data['mu_hat']
    
    # Check for W_param in saved arrays
    if 'W_param' in data:
        return data['W_param']
    
    # Try alternative names
    for key in ['W', 'weights', 'coupling_matrix']:
        if key in data:
            return data[key]
    
    print(f"  ⚠️  No mu_hat/W_param found in {result_file.name}")
    print(f"      Available keys: {list(data.keys())}")
    return None

def compute_weight_matrix(
    W_param: np.ndarray,
    sign_adj: np.ndarray,
    method: str = "absolute"
) -> np.ndarray:
    """
    Compute functional weight matrix from W_param.
    
    Args:
        W_param: Raw coupling parameter matrix
        sign_adj: Binary sign adjacency matrix
        method: "absolute" (|W|), "masked" (|W| where sign_adj > 0), or "signed" (W)
    
    Returns:
        Weight matrix
    """
    if method == "absolute":
        return np.abs(W_param)
    elif method == "masked":
        weights = np.abs(W_param)
        weights[sign_adj == 0] = 0
        return weights
    elif method == "signed":
        return W_param
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================================
# STRUCTURAL WEIGHTS
# ============================================================================

def load_structural_weights() -> Tuple[np.ndarray, List[str]]:
    """
    Load structural connectome with synapse counts.
    
    Returns:
        W_struct: Weight matrix (synapse counts)
        neurons: Neuron names
    """
    struct_file = CONNECTOME_DIR / "A_struct.npy"
    nodes_file = CONNECTOME_DIR / "nodes.json"
    
    if not struct_file.exists() or not nodes_file.exists():
        print("⚠️  Structural connectome not found")
        return None, None
    
    A_struct = np.load(struct_file)
    with open(nodes_file, 'r') as f:
        neurons = json.load(f)
    
    # Check if weighted
    unique_vals = np.unique(A_struct[A_struct > 0])
    if len(unique_vals) == 1:
        print(f"  Structural connectome is BINARY (all edges = {unique_vals[0]})")
        print(f"  Will use edge presence/absence for comparison")
        return A_struct, neurons
    else:
        print(f"  Structural connectome has WEIGHTS (range: {A_struct[A_struct > 0].min():.1f} - {A_struct[A_struct > 0].max():.1f})")
        print(f"  Unique weight values: {len(unique_vals)}")
        return A_struct, neurons

# ============================================================================
# WEIGHT CORRELATION ANALYSIS
# ============================================================================

def analyze_weight_correlation(
    W_functional: np.ndarray,
    W_structural: np.ndarray,
    neuron_names: List[str]
) -> Dict:
    """
    Compute detailed correlation analysis between functional and structural weights.
    
    Args:
        W_functional: Functional weight matrix
        W_structural: Structural weight matrix (synapse counts)
        neuron_names: Neuron names for both matrices
    
    Returns:
        Dict with correlation statistics and edge lists
    """
    # Find edges present in both networks
    both_present = (W_functional > 0) & (W_structural > 0)
    n_shared = int(np.sum(both_present))
    
    if n_shared == 0:
        return {
            "n_shared_edges": 0,
            "correlation_possible": False
        }
    
    # Extract weights for shared edges
    func_weights = W_functional[both_present]
    struct_weights = W_structural[both_present]
    
    # Check for variance
    func_var = np.var(func_weights)
    struct_var = np.var(struct_weights)
    
    results = {
        "n_shared_edges": n_shared,
        "correlation_possible": func_var > 1e-10 and struct_var > 1e-10,
        "func_weights_mean": float(np.mean(func_weights)),
        "func_weights_std": float(np.std(func_weights)),
        "func_weights_median": float(np.median(func_weights)),
        "func_weights_min": float(np.min(func_weights)),
        "func_weights_max": float(np.max(func_weights)),
        "struct_weights_mean": float(np.mean(struct_weights)),
        "struct_weights_std": float(np.std(struct_weights)),
        "struct_weights_median": float(np.median(struct_weights)),
        "struct_weights_min": float(np.min(struct_weights)),
        "struct_weights_max": float(np.max(struct_weights)),
        "func_weight_variance": float(func_var),
        "struct_weight_variance": float(struct_var)
    }
    
    # Compute correlations if possible
    if results["correlation_possible"]:
        try:
            pearson_r, pearson_p = pearsonr(func_weights, struct_weights)
            spearman_r, spearman_p = spearmanr(func_weights, struct_weights)
            
            results.update({
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p)
            })
        except Exception as e:
            print(f"  ⚠️  Correlation computation failed: {e}")
            results["correlation_possible"] = False
    
    # Store edge information
    edge_indices = np.where(both_present)
    results["shared_edges"] = [
        {
            "source": neuron_names[i],
            "target": neuron_names[j],
            "func_weight": float(W_functional[i, j]),
            "struct_weight": float(W_structural[i, j])
        }
        for i, j in zip(edge_indices[0], edge_indices[1])
    ]
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_weight_distributions(
    W_functional: np.ndarray,
    W_structural: np.ndarray,
    config_name: str,
    output_path: Path
):
    """Plot weight distributions for functional and structural networks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Weight Distributions: {config_name}", fontsize=14, fontweight='bold')
    
    # Functional weights (all edges)
    ax = axes[0, 0]
    func_weights = W_functional[W_functional > 0]
    if len(func_weights) > 0:
        ax.hist(func_weights, bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
        ax.set_xlabel("Functional Weight (|W_param|)")
        ax.set_ylabel("Count")
        ax.set_title(f"Functional Edge Weights (n={len(func_weights)})")
        ax.axvline(np.median(func_weights), color='r', linestyle='--', 
                   label=f'Median={np.median(func_weights):.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Structural weights (all edges)
    ax = axes[0, 1]
    struct_weights = W_structural[W_structural > 0]
    if len(struct_weights) > 0:
        ax.hist(struct_weights, bins=50, edgecolor='black', alpha=0.7, color='#ff7f0e')
        ax.set_xlabel("Structural Weight (Synapse Count)")
        ax.set_ylabel("Count")
        ax.set_title(f"Structural Edge Weights (n={len(struct_weights)})")
        ax.axvline(np.median(struct_weights), color='r', linestyle='--',
                   label=f'Median={np.median(struct_weights):.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Scatter plot of shared edges
    ax = axes[1, 0]
    both_present = (W_functional > 0) & (W_structural > 0)
    if np.sum(both_present) > 0:
        func_shared = W_functional[both_present]
        struct_shared = W_structural[both_present]
        
        ax.scatter(struct_shared, func_shared, alpha=0.5, s=20, color='#2ca02c')
        ax.set_xlabel("Structural Weight")
        ax.set_ylabel("Functional Weight")
        ax.set_title(f"Shared Edges (n={len(func_shared)})")
        
        # Add regression line if sufficient data
        if len(func_shared) > 1 and np.var(struct_shared) > 1e-10:
            z = np.polyfit(struct_shared, func_shared, 1)
            p = np.poly1d(z)
            x_range = np.linspace(struct_shared.min(), struct_shared.max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
            
            # Correlation
            if np.var(func_shared) > 1e-10:
                r, _ = pearsonr(struct_shared, func_shared)
                ax.text(0.05, 0.95, f'Pearson r = {r:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No shared edges", ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
    
    # Weight ratio histogram
    ax = axes[1, 1]
    if np.sum(both_present) > 0:
        func_shared = W_functional[both_present]
        struct_shared = W_structural[both_present]
        
        # Avoid division by zero
        valid_ratios = struct_shared > 0
        if np.sum(valid_ratios) > 0:
            ratios = func_shared[valid_ratios] / struct_shared[valid_ratios]
            ax.hist(ratios, bins=30, edgecolor='black', alpha=0.7, color='#9467bd')
            ax.set_xlabel("Functional / Structural Weight Ratio")
            ax.set_ylabel("Count")
            ax.set_title("Weight Ratio Distribution")
            ax.axvline(np.median(ratios), color='r', linestyle='--',
                      label=f'Median={np.median(ratios):.3f}')
            ax.legend()
            ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No shared edges", ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_weight_comparison_summary(
    all_results: List[Dict],
    output_path: Path
):
    """Create summary plot comparing all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weighted Correlation Summary: All Configurations", 
                 fontsize=14, fontweight='bold')
    
    # Filter results with valid correlations
    valid_results = [r for r in all_results if r.get('correlation_possible', False)]
    
    if len(valid_results) == 0:
        print("  ⚠️  No valid correlations to plot")
        plt.close()
        return
    
    # Extract data
    configs = [f"{r['stimulus']}_{r['model_type']}_{r['train_split']}" 
               for r in valid_results]
    pearson_rs = [r['pearson_r'] for r in valid_results]
    spearman_rs = [r['spearman_r'] for r in valid_results]
    n_shared = [r['n_shared_edges'] for r in valid_results]
    
    # Pearson correlation by configuration
    ax = axes[0, 0]
    colors = ['#1f77b4' if 'nacl' in c else '#ff7f0e' for c in configs]
    ax.barh(range(len(configs)), pearson_rs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c[:30] for c in configs], fontsize=8)
    ax.set_xlabel("Pearson r")
    ax.set_title("Pearson Correlation by Configuration")
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Spearman correlation by configuration
    ax = axes[0, 1]
    ax.barh(range(len(configs)), spearman_rs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c[:30] for c in configs], fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Spearman Correlation by Configuration")
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Number of shared edges
    ax = axes[1, 0]
    ax.barh(range(len(configs)), n_shared, color=colors, alpha=0.7)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c[:30] for c in configs], fontsize=8)
    ax.set_xlabel("Number of Shared Edges")
    ax.set_title("Shared Edge Count")
    ax.grid(axis='x', alpha=0.3)
    
    # Correlation scatter
    ax = axes[1, 1]
    ax.scatter(pearson_rs, spearman_rs, s=100, alpha=0.6, c=colors)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Pearson vs Spearman Correlation")
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, label='y=x')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run weighted connectome analysis."""
    
    print("=" * 80)
    print("WEIGHTED CONNECTOME ANALYSIS (W_param Extraction)")
    print("=" * 80)
    print()
    
    # Load structural weights
    print("Step 1: Loading structural connectome weights...")
    print("-" * 80)
    W_struct, struct_neurons = load_structural_weights()
    
    if W_struct is None:
        print("⚠️  Cannot proceed without structural connectome")
        return
    
    print(f"  Loaded {len(struct_neurons)} neurons")
    print()
    
    # Load functional neuron names
    print("Step 2: Loading functional neuron names...")
    print("-" * 80)
    data_file = PROJECT_ROOT / "data" / "Head_Activity_OH15500.mat"
    mat_data = loadmat(data_file)
    functional_neurons = [str(mat_data['neurons'][i, 0][0]) 
                          for i in range(mat_data['neurons'].shape[0])]
    print(f"  Loaded {len(functional_neurons)} functional neurons")
    
    # Find common neurons
    struct_set = set(struct_neurons)
    func_set = set(functional_neurons)
    common_neurons = sorted(list(struct_set & func_set))
    print(f"  Common neurons: {len(common_neurons)}")
    print()
    
    # Create alignment indices
    func_to_idx = {n: i for i, n in enumerate(functional_neurons)}
    struct_to_idx = {n: i for i, n in enumerate(struct_neurons)}
    
    func_indices = [func_to_idx[n] for n in common_neurons]
    struct_indices = [struct_to_idx[n] for n in common_neurons]
    
    # Align structural connectome
    W_struct_aligned = W_struct[np.ix_(struct_indices, struct_indices)]
    print(f"✓ Aligned structural connectome to {len(common_neurons)} neurons")
    print()
    
    # Load and analyze all configurations
    print("Step 3: Analyzing W_param weights for all configurations...")
    print("-" * 80)
    
    results_csv = RESULTS_DIR / "all_results.csv"
    results_df = pd.read_csv(results_csv)
    
    all_weight_results = []
    
    for idx, row in results_df.iterrows():
        stimulus = row['stimulus']
        model_type = row['model_type']
        train_split = row['train_split']
        
        config_id = f"{model_type}_{train_split}"
        print(f"\n[{stimulus} - {config_id}]")
        
        # Load result file
        result_file = RESULTS_DIR / "models" / stimulus / config_id / "result.npz"
        if not result_file.exists():
            print(f"  ⚠️  Result file not found")
            continue
        
        data = np.load(result_file)
        
        # Extract W_param
        W_param = extract_wparam_weights(result_file)
        
        if W_param is None:
            print(f"  ⚠️  No W_param available - using sign_adj for binary comparison")
            W_functional_full = data['sign_adj'].copy()
        else:
            # Compute absolute weights
            W_functional_full = np.abs(W_param)
            print(f"  ✓ Extracted W_param: shape {W_param.shape}")
            print(f"      Weight range: [{W_functional_full[W_functional_full > 0].min():.4f}, "
                  f"{W_functional_full[W_functional_full > 0].max():.4f}]")
        
        # Align to common neurons
        W_functional = W_functional_full[np.ix_(func_indices, func_indices)]
        
        # Analyze weights
        weight_stats = analyze_weight_correlation(
            W_functional, W_struct_aligned, common_neurons
        )
        
        # Add metadata
        weight_stats['stimulus'] = stimulus
        weight_stats['model_type'] = model_type
        weight_stats['train_split'] = train_split
        all_weight_results.append(weight_stats)
        
        # Print results
        print(f"  Shared edges: {weight_stats['n_shared_edges']}")
        print(f"  Func weight: {weight_stats['func_weights_mean']:.4f} ± {weight_stats['func_weights_std']:.4f}")
        print(f"  Struct weight: {weight_stats['struct_weights_mean']:.4f} ± {weight_stats['struct_weights_std']:.4f}")
        
        if weight_stats['correlation_possible']:
            print(f"  Pearson r: {weight_stats['pearson_r']:.4f} (p={weight_stats['pearson_p']:.4e})")
            print(f"  Spearman ρ: {weight_stats['spearman_r']:.4f} (p={weight_stats['spearman_p']:.4e})")
        else:
            print(f"  ⚠️  Insufficient variance for correlation")
        
        # Generate plots
        plot_dir = FIGURES_DIR / stimulus / config_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_weight_distributions(
            W_functional, W_struct_aligned,
            f"{stimulus.capitalize()} - {config_id}",
            plot_dir / "weight_distributions.png"
        )
    
    # Save results
    print("\n" + "=" * 80)
    print("Step 4: Saving results...")
    print("-" * 80)
    
    # Save correlation results
    results_df_weights = pd.DataFrame([
        {
            'stimulus': r['stimulus'],
            'model_type': r['model_type'],
            'train_split': r['train_split'],
            'n_shared_edges': r['n_shared_edges'],
            'correlation_possible': r['correlation_possible'],
            'pearson_r': r.get('pearson_r', np.nan),
            'pearson_p': r.get('pearson_p', np.nan),
            'spearman_r': r.get('spearman_r', np.nan),
            'spearman_p': r.get('spearman_p', np.nan),
            'func_weight_mean': r['func_weights_mean'],
            'func_weight_std': r['func_weights_std'],
            'struct_weight_mean': r['struct_weights_mean'],
            'struct_weight_std': r['struct_weights_std']
        }
        for r in all_weight_results
    ])
    
    results_df_weights.to_csv(OUTPUT_DIR / "weight_correlations.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'weight_correlations.csv'}")
    
    # Save detailed edge information
    # Convert numpy booleans to Python booleans for JSON serialization
    for r in all_weight_results:
        if 'correlation_possible' in r:
            r['correlation_possible'] = bool(r['correlation_possible'])
    
    with open(OUTPUT_DIR / "weighted_edge_details.json", 'w') as f:
        json.dump(all_weight_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
    print(f"✓ Saved: {OUTPUT_DIR / 'weighted_edge_details.json'}")
    
    # Generate summary plots
    plot_weight_comparison_summary(
        all_weight_results,
        FIGURES_DIR / "weight_correlation_summary.png"
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  Correlation results: weight_correlations.csv")
    print(f"  Detailed edges: weighted_edge_details.json")
    print(f"  Figures: {FIGURES_DIR}")
    print()

if __name__ == "__main__":
    main()
