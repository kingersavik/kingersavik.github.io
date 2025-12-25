#!/usr/bin/env python3
"""
SCRIPT 04: Connectome Comparison
================================

Compares functional connectivity (SBTG predictions) against Cook et al. 2019 
structural connectome.

Analyses:
1. Graph statistics comparison
2. Edge overlap analysis (precision, recall, F1)
3. Weighted edge correlation (functional weights vs structural weights)
4. Visualization of matches, misses, and novel edges
5. Path-2 (indirect) connection analysis
6. Random baseline comparison
7. Cross-stimulus consistency
8. Permutation test for statistical significance

USAGE:
    python pipeline/04_connectome_comparison.py

INPUTS:
    - results/comprehensive_sbtg/models/{stim}/{config}/result.npz
    - results/intermediate/connectome/A_struct.npy

OUTPUTS:
    - results/comprehensive_sbtg/connectome_comparison/edge_comparisons.csv
    - results/comprehensive_sbtg/connectome_comparison/path2_comparisons.csv
    - results/comprehensive_sbtg/connectome_comparison/permutation_tests.csv
    - results/comprehensive_sbtg/connectome_comparison/figures/*.png

Expected runtime: ~3-5 minutes
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
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_sbtg"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
OUTPUT_DIR = RESULTS_DIR / "connectome_comparison"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONNECTOME LOADING
# ============================================================================

def load_structural_connectome() -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load Cook et al. 2019 structural connectome.
    
    Returns:
        A_struct: (n, n) binary adjacency matrix
        neuron_names: List of neuron names
        metadata: Dict with connectome statistics
    """
    if not CONNECTOME_DIR.exists():
        print(f"Warning: Connectome directory not found: {CONNECTOME_DIR}")
        print("Attempting to load from alternative locations...")
        
    # Try to load structural adjacency matrix
    struct_file = CONNECTOME_DIR / "A_struct.npy"
    nodes_file = CONNECTOME_DIR / "nodes.json"
    
    if struct_file.exists() and nodes_file.exists():
        A_struct = np.load(struct_file)
        with open(nodes_file, 'r') as f:
            neuron_names = json.load(f)
        
        # Compute statistics
        n_neurons = A_struct.shape[0]
        n_edges = int(np.sum(A_struct > 0))
        density = n_edges / (n_neurons * (n_neurons - 1))
        
        # Degree distribution
        in_degree = np.sum(A_struct > 0, axis=0)
        out_degree = np.sum(A_struct > 0, axis=1)
        
        metadata = {
            "n_neurons": n_neurons,
            "n_edges": n_edges,
            "density": density,
            "mean_in_degree": float(np.mean(in_degree)),
            "mean_out_degree": float(np.mean(out_degree)),
            "max_in_degree": int(np.max(in_degree)),
            "max_out_degree": int(np.max(out_degree))
        }
        
        print("✓ Loaded structural connectome")
        print(f"  Neurons: {n_neurons}")
        print(f"  Edges: {n_edges}")
        print(f"  Density: {density:.4f}")
        
        return A_struct, neuron_names, metadata
    else:
        print("⚠️  Structural connectome files not found")
        print("   Creating placeholder for demonstration...")
        # Return None to indicate missing data
        return None, None, None

def align_connectome_to_neurons(
    A_struct: np.ndarray,
    struct_neurons: List[str],
    functional_neurons: List[str]
) -> np.ndarray:
    """
    Align structural connectome to match functional neuron ordering.
    
    Args:
        A_struct: Structural adjacency matrix
        struct_neurons: Neuron names in structural connectome
        functional_neurons: Neuron names in functional data
    
    Returns:
        A_aligned: Aligned adjacency matrix (may be smaller if neurons missing)
    """
    # Find common neurons
    struct_set = set(struct_neurons)
    common_neurons = [n for n in functional_neurons if n in struct_set]
    
    if len(common_neurons) == 0:
        print("⚠️  No overlapping neurons found between structural and functional data")
        return None
    
    # Create index mappings
    struct_to_idx = {n: i for i, n in enumerate(struct_neurons)}
    common_indices = [struct_to_idx[n] for n in common_neurons]
    
    # Extract aligned submatrix
    A_aligned = A_struct[np.ix_(common_indices, common_indices)]
    
    print(f"✓ Aligned connectome: {len(common_neurons)} common neurons")
    
    return A_aligned, common_neurons

# ============================================================================
# GRAPH STATISTICS
# ============================================================================

def compute_graph_statistics(A: np.ndarray, name: str) -> Dict:
    """
    Compute comprehensive graph statistics.
    
    Args:
        A: Adjacency matrix (can be weighted or binary)
        name: Graph identifier
    
    Returns:
        Dict of statistics
    """
    # Binarize for topology analysis
    A_binary = (A != 0).astype(int)
    n = A.shape[0]
    
    # Basic statistics
    n_edges = int(np.sum(A_binary > 0))
    density = n_edges / (n * (n - 1)) if n > 1 else 0
    
    # Degree distributions
    in_degree = np.sum(A_binary, axis=0)
    out_degree = np.sum(A_binary, axis=1)
    total_degree = in_degree + out_degree
    
    # Reciprocity (bidirectional connections)
    reciprocal = np.sum(A_binary * A_binary.T) / 2
    reciprocity = reciprocal / n_edges if n_edges > 0 else 0
    
    # Clustering coefficient (simple approximation)
    triangles = 0
    for i in range(n):
        neighbors_out = np.where(A_binary[i, :] > 0)[0]
        for j in neighbors_out:
            neighbors_j = np.where(A_binary[j, :] > 0)[0]
            triangles += len(set(neighbors_out) & set(neighbors_j))
    
    clustering = triangles / (3 * n_edges) if n_edges > 0 else 0
    
    stats = {
        "name": name,
        "n_neurons": n,
        "n_edges": n_edges,
        "density": density,
        "mean_in_degree": float(np.mean(in_degree)),
        "std_in_degree": float(np.std(in_degree)),
        "mean_out_degree": float(np.mean(out_degree)),
        "std_out_degree": float(np.std(out_degree)),
        "max_in_degree": int(np.max(in_degree)),
        "max_out_degree": int(np.max(out_degree)),
        "reciprocity": reciprocity,
        "clustering": clustering,
        "in_degree_dist": in_degree.tolist(),
        "out_degree_dist": out_degree.tolist()
    }
    
    return stats

# ============================================================================
# EDGE COMPARISON
# ============================================================================

def compare_edge_sets(
    A_functional: np.ndarray,
    A_structural: np.ndarray,
    threshold: float = 0.0
) -> Dict:
    """
    Compare functional and structural edge sets.
    
    Args:
        A_functional: Functional adjacency matrix (weighted)
        A_structural: Structural adjacency matrix (binary)
        threshold: Threshold for binarizing functional edges
    
    Returns:
        Dict with precision, recall, F1, and confusion matrix
    """
    # Binarize both matrices
    A_func_binary = (np.abs(A_functional) > threshold).astype(int)
    A_struct_binary = (A_structural > 0).astype(int)
    
    # Flatten (exclude diagonal)
    n = A_functional.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_true = A_struct_binary[mask].flatten()
    y_pred = A_func_binary[mask].flatten()
    
    # Confusion matrix
    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    # Metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "n_functional_edges": int(np.sum(A_func_binary)),
        "n_structural_edges": int(np.sum(A_struct_binary))
    }

def compute_weight_correlation(
    A_functional: np.ndarray,
    A_structural: np.ndarray
) -> Dict:
    """
    Compute correlation between functional and structural edge weights.
    
    Only considers edges that exist in both networks.
    
    Args:
        A_functional: Functional adjacency with weights
        A_structural: Structural adjacency with weights (synapse counts)
    
    Returns:
        Dict with correlation coefficients and statistics
    """
    # Find edges present in both networks
    both_present = (A_functional != 0) & (A_structural != 0)
    
    if np.sum(both_present) == 0:
        return {
            "n_shared_edges": 0,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan
        }
    
    # Extract weights for shared edges (use absolute value of functional weights)
    func_weights = np.abs(A_functional[both_present])
    struct_weights = A_structural[both_present]
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(func_weights, struct_weights)
    spearman_r, spearman_p = spearmanr(func_weights, struct_weights)
    
    return {
        "n_shared_edges": int(np.sum(both_present)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "func_weights_mean": float(np.mean(func_weights)),
        "func_weights_std": float(np.std(func_weights)),
        "struct_weights_mean": float(np.mean(struct_weights)),
        "struct_weights_std": float(np.std(struct_weights))
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_degree_distributions(
    stats_list: List[Dict],
    output_path: Path
):
    """Compare degree distributions across graphs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Degree Distribution Comparison", fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # In-degree histogram
    ax = axes[0, 0]
    for i, stats in enumerate(stats_list):
        ax.hist(stats['in_degree_dist'], bins=20, alpha=0.5,
                label=stats['name'], color=colors[i % len(colors)])
    ax.set_xlabel("In-Degree")
    ax.set_ylabel("Count")
    ax.set_title("In-Degree Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Out-degree histogram
    ax = axes[0, 1]
    for i, stats in enumerate(stats_list):
        ax.hist(stats['out_degree_dist'], bins=20, alpha=0.5,
                label=stats['name'], color=colors[i % len(colors)])
    ax.set_xlabel("Out-Degree")
    ax.set_ylabel("Count")
    ax.set_title("Out-Degree Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Statistics comparison table
    ax = axes[1, 0]
    ax.axis('off')
    table_data = []
    for stats in stats_list:
        table_data.append([
            stats['name'][:20],
            f"{stats['n_edges']}",
            f"{stats['density']:.3f}",
            f"{stats['mean_in_degree']:.1f}",
            f"{stats['reciprocity']:.3f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Graph", "Edges", "Density", "Avg In-Deg", "Reciprocity"],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title("Graph Statistics Summary")
    
    # Edge count comparison
    ax = axes[1, 1]
    names = [s['name'][:15] for s in stats_list]
    edges = [s['n_edges'] for s in stats_list]
    ax.bar(range(len(names)), edges, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Number of Edges")
    ax.set_title("Total Edge Count")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved degree distribution comparison: {output_path}")

def plot_weight_correlation(
    A_functional: np.ndarray,
    A_structural: np.ndarray,
    correlation_stats: Dict,
    output_path: Path,
    title: str = "Functional vs Structural Edge Weights"
):
    """Scatter plot of functional vs structural edge weights."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Find shared edges
    both_present = (A_functional != 0) & (A_structural != 0)
    
    if np.sum(both_present) == 0:
        print("  No shared edges to plot")
        return
    
    func_weights = np.abs(A_functional[both_present])
    struct_weights = A_structural[both_present]
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(struct_weights, func_weights, alpha=0.5, s=20)
    ax.set_xlabel("Structural Weight (Synapse Count)")
    ax.set_ylabel("Functional Weight (|SBTG Coefficient|)")
    ax.set_title(f"Weight Correlation (n={correlation_stats['n_shared_edges']} edges)")
    
    # Add regression line
    if len(struct_weights) > 1:
        z = np.polyfit(struct_weights, func_weights, 1)
        p = np.poly1d(z)
        ax.plot(struct_weights, p(struct_weights), "r--", alpha=0.8, linewidth=2)
    
    # Add correlation info
    text = f"Pearson r = {correlation_stats['pearson_r']:.3f}\n"
    text += f"Spearman ρ = {correlation_stats['spearman_r']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(alpha=0.3)
    
    # Histogram of weight ratios
    ax = axes[1]
    if len(func_weights) > 0 and np.all(struct_weights > 0):
        ratios = func_weights / struct_weights
        ax.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Functional / Structural Weight Ratio")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Weight Ratios")
        ax.axvline(np.median(ratios), color='r', linestyle='--', linewidth=2,
                   label=f'Median = {np.median(ratios):.2f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved weight correlation plot: {output_path}")

def plot_confusion_matrix(
    comparison: Dict,
    output_path: Path,
    title: str = "Edge Detection Confusion Matrix"
):
    """Visualize edge detection performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Confusion matrix heatmap
    ax = axes[0]
    cm = np.array([
        [comparison['true_positives'], comparison['false_negatives']],
        [comparison['false_positives'], comparison['true_negatives']]
    ])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted +', 'Predicted -'],
                yticklabels=['Actual +', 'Actual -'],
                cbar_kws={'label': 'Count'})
    ax.set_title("Confusion Matrix")
    
    # Metrics bar chart
    ax = axes[1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [comparison['precision'], comparison['recall'], comparison['f1_score']]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1])
    ax.set_title("Performance Metrics")
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {output_path}")

# ============================================================================
# EXTENDED ANALYSES (v2)
# ============================================================================

def compute_path2_analysis(A_struct: np.ndarray) -> Dict:
    """
    Analyze path-2 (indirect) connections in structural connectome.
    
    Many "false positives" in functional connectivity may actually be
    indirect connections through an intermediary neuron.
    
    Args:
        A_struct: Structural adjacency matrix
    
    Returns:
        Dict with path-2 statistics and the path-2 reachability matrix
    """
    A_binary = (A_struct > 0).astype(float)
    n = A_struct.shape[0]
    
    # Compute 2-hop reachability: A^2
    A_squared = A_binary @ A_binary
    np.fill_diagonal(A_squared, 0)
    
    # Path-2 only: reachable in 2 hops but NOT in 1 hop
    A_path2_only = (A_squared > 0) & (A_binary == 0)
    
    # Combined reachability (1 or 2 hops)
    A_reachable_2 = ((A_binary > 0) | (A_squared > 0)).astype(int)
    
    n_direct = int(A_binary.sum())
    n_path2_only = int(A_path2_only.sum())
    n_total_reachable = int(A_reachable_2.sum())
    n_possible = n * (n - 1)
    
    return {
        "n_direct_edges": n_direct,
        "n_path2_only": n_path2_only,
        "n_total_reachable_2hop": n_total_reachable,
        "direct_density": n_direct / n_possible,
        "path2_density": n_path2_only / n_possible,
        "reachable_2hop_density": n_total_reachable / n_possible,
        "A_path2_only": A_path2_only,
        "A_reachable_2": A_reachable_2
    }


def compare_with_path2_ground_truth(
    A_functional: np.ndarray,
    A_struct: np.ndarray,
    A_reachable_2: np.ndarray
) -> Dict:
    """
    Compare functional edges against both direct and path-2 ground truth.
    
    This provides a more lenient evaluation where indirect connections
    are also considered correct.
    """
    A_func_binary = (A_functional != 0).astype(int)
    A_struct_binary = (A_struct > 0).astype(int)
    
    n = A_functional.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    func_flat = A_func_binary[mask].flatten()
    struct_flat = A_struct_binary[mask].flatten()
    reachable_flat = A_reachable_2[mask].flatten()
    
    # Standard metrics (direct only)
    tp_direct = int(((func_flat == 1) & (struct_flat == 1)).sum())
    fp_direct = int(((func_flat == 1) & (struct_flat == 0)).sum())
    
    # Path-2 metrics (direct OR indirect)
    tp_path2 = int(((func_flat == 1) & (reachable_flat == 1)).sum())
    fp_path2 = int(((func_flat == 1) & (reachable_flat == 0)).sum())
    
    # How many "false positives" are actually path-2 connections?
    fp_explained_by_path2 = fp_direct - fp_path2
    
    n_func = int(func_flat.sum())
    n_struct = int(struct_flat.sum())
    n_reachable = int(reachable_flat.sum())
    
    # Metrics for path-2 ground truth
    precision_path2 = tp_path2 / n_func if n_func > 0 else 0
    recall_path2 = tp_path2 / n_reachable if n_reachable > 0 else 0
    f1_path2 = 2 * precision_path2 * recall_path2 / (precision_path2 + recall_path2) if (precision_path2 + recall_path2) > 0 else 0
    
    return {
        "tp_direct": tp_direct,
        "fp_direct": fp_direct,
        "tp_path2": tp_path2,
        "fp_path2": fp_path2,
        "fp_explained_by_path2": fp_explained_by_path2,
        "pct_fp_explained": fp_explained_by_path2 / fp_direct * 100 if fp_direct > 0 else 0,
        "precision_path2": precision_path2,
        "recall_path2": recall_path2,
        "f1_path2": f1_path2,
        "n_reachable_2hop": n_reachable
    }


def compute_random_baseline(n_predicted: int, n_structural: int, n_possible: int) -> Dict:
    """
    Compute expected metrics for a random baseline predictor.
    
    If we predict k edges uniformly at random, what would we expect?
    """
    density = n_structural / n_possible
    
    expected_tp = n_predicted * density
    precision = density
    recall = expected_tp / n_structural if n_structural > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "random_expected_tp": expected_tp,
        "random_precision": precision,
        "random_recall": recall,
        "random_f1": f1
    }


def compute_correlation_baseline(timeseries: np.ndarray, A_struct: np.ndarray, threshold_percentile: float = 90) -> Dict:
    """
    Compute simple correlation-based baseline for comparison.
    
    This answers: "How well does simple Pearson correlation predict edges?"
    
    Args:
        timeseries: (T, N) timeseries data
        A_struct: (N, N) structural adjacency
        threshold_percentile: Percentile threshold for binarizing correlations
    
    Returns:
        Dict with baseline metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(timeseries.T)
    np.fill_diagonal(corr_matrix, 0)
    
    # Take absolute value (we care about connection strength, not sign)
    abs_corr = np.abs(corr_matrix)
    
    # Threshold to get binary predictions
    threshold = np.percentile(abs_corr[~np.eye(len(abs_corr), dtype=bool)], threshold_percentile)
    pred_binary = (abs_corr >= threshold).astype(int)
    
    # Ground truth
    struct_binary = (A_struct > 0).astype(int)
    
    # Flatten (exclude diagonal)
    n = len(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    pred_flat = pred_binary[mask]
    struct_flat = struct_binary[mask]
    
    tp = int(((pred_flat == 1) & (struct_flat == 1)).sum())
    fp = int(((pred_flat == 1) & (struct_flat == 0)).sum())
    fn = int(((pred_flat == 0) & (struct_flat == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "corr_threshold": threshold,
        "corr_n_predicted": int(pred_flat.sum()),
        "corr_tp": tp,
        "corr_precision": precision,
        "corr_recall": recall,
        "corr_f1": f1
    }


def compute_cross_stimulus_consistency(all_comparisons: List[Dict]) -> Dict:
    """
    Analyze which edges are consistently detected across stimuli.
    
    Edges that appear in multiple stimulus conditions are more likely to be real.
    """
    df = pd.DataFrame(all_comparisons)
    
    # Group by stimulus
    stimuli = df['stimulus'].unique()
    
    if len(stimuli) < 2:
        return {"n_stimuli": len(stimuli), "consistency_available": False}
    
    # For each model_type/train_split, compare edges across stimuli
    consistency_results = []
    
    for (model_type, train_split), group in df.groupby(['model_type', 'train_split']):
        if len(group) < 2:
            continue
        
        # Average F1 across stimuli for this config
        avg_f1 = group['f1_score'].mean()
        std_f1 = group['f1_score'].std()
        
        consistency_results.append({
            'model_type': model_type,
            'train_split': train_split,
            'avg_f1_across_stimuli': avg_f1,
            'std_f1_across_stimuli': std_f1,
            'n_stimuli': len(group)
        })
    
    return {
        "n_stimuli": len(stimuli),
        "consistency_available": True,
        "stimuli": list(stimuli),
        "config_consistency": consistency_results
    }


def run_permutation_test(
    A_functional: np.ndarray,
    A_struct: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Permutation test for edge detection significance.
    
    Null hypothesis: Functional edges are randomly distributed.
    We shuffle the functional adjacency and compute F1, then compare to observed.
    """
    np.random.seed(seed)
    
    A_func_binary = (A_functional != 0).astype(int)
    A_struct_binary = (A_struct > 0).astype(int)
    
    n = A_functional.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    func_flat = A_func_binary[mask].flatten()
    struct_flat = A_struct_binary[mask].flatten()
    
    # Observed F1
    from sklearn.metrics import f1_score
    observed_f1 = f1_score(struct_flat, func_flat, zero_division=0)
    
    # Permutation distribution
    null_f1s = []
    n_edges = int(func_flat.sum())
    
    for _ in range(n_permutations):
        # Create random prediction with same number of edges
        perm_pred = np.zeros_like(func_flat)
        random_indices = np.random.choice(len(func_flat), size=n_edges, replace=False)
        perm_pred[random_indices] = 1
        
        null_f1 = f1_score(struct_flat, perm_pred, zero_division=0)
        null_f1s.append(null_f1)
    
    null_f1s = np.array(null_f1s)
    
    # P-value: fraction of permutations with F1 >= observed
    p_value = (null_f1s >= observed_f1).mean()
    
    return {
        "observed_f1": observed_f1,
        "null_mean_f1": float(null_f1s.mean()),
        "null_std_f1": float(null_f1s.std()),
        "null_max_f1": float(null_f1s.max()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run connectome comparison analysis."""
    
    print("=" * 80)
    print("CONNECTOME COMPARISON ANALYSIS")
    print("=" * 80)
    print()
    
    # Load structural connectome
    print("Step 1: Loading structural connectome...")
    print("-" * 80)
    A_struct, struct_neurons, struct_stats = load_structural_connectome()
    
    if A_struct is None:
        print("⚠️  Cannot proceed without structural connectome")
        print("    Please ensure connectome data is available in:")
        print(f"    {CONNECTOME_DIR}")
        return
    
    print()
    
    # Load comprehensive SBTG results
    print("Step 2: Loading SBTG results...")
    print("-" * 80)
    
    results_csv = RESULTS_DIR / "all_results.csv"
    if not results_csv.exists():
        print(f"⚠️  Results file not found: {results_csv}")
        print("    Please run 05_comprehensive_sbtg_analysis.py first")
        return
    
    results_df = pd.read_csv(results_csv)
    print(f"✓ Loaded {len(results_df)} model configurations")
    print()
    
    # Analyze each configuration
    print("Step 3: Comparing functional vs structural connectivity...")
    print("-" * 80)
    
    all_comparisons = []
    all_graph_stats = [compute_graph_statistics(A_struct, "Structural Connectome")]
    
    # Load functional neuron names from first result
    first_result = results_df.iloc[0]
    first_file = RESULTS_DIR / "models" / first_result['stimulus'] / f"{first_result['model_type']}_{first_result['train_split']}" / "result.npz"
    functional_neurons_file = first_file.parent / "neuron_names.json"
    
    if functional_neurons_file.exists():
        with open(functional_neurons_file, 'r') as f:
            functional_neurons = json.load(f)
    else:
        # Try to load from data directory metadata
        print("  Loading neuron names from NeuroPAL data...")
        from scipy.io import loadmat
        data_files = list((PROJECT_ROOT / "data").glob("*.mat"))
        if data_files:
            mat_data = loadmat(data_files[0])
            functional_neurons = [str(n[0]) for n in mat_data['neuron_names'][0]]
        else:
            print("  ⚠️  Could not load functional neuron names")
            return
    
    print(f"  Functional neurons: {len(functional_neurons)}")
    print(f"  Structural neurons: {len(struct_neurons)}")
    
    # Find common neurons
    struct_set = set(struct_neurons)
    func_set = set(functional_neurons)
    common_neurons = sorted(list(struct_set & func_set))
    
    print(f"  Common neurons: {len(common_neurons)}")
    print(f"  Example common neurons: {', '.join(common_neurons[:10])}")
    print()
    
    if len(common_neurons) == 0:
        print("  ⚠️  No common neurons found - cannot compare")
        return
    
    # Create alignment indices
    func_to_idx = {n: i for i, n in enumerate(functional_neurons)}
    struct_to_idx = {n: i for i, n in enumerate(struct_neurons)}
    
    func_indices = [func_to_idx[n] for n in common_neurons]
    struct_indices = [struct_to_idx[n] for n in common_neurons]
    
    # Align structural connectome to common neurons
    A_struct_aligned = A_struct[np.ix_(struct_indices, struct_indices)]
    
    print(f"✓ Aligned structural connectome to {len(common_neurons)} neurons")
    print(f"  Aligned edges: {int(np.sum(A_struct_aligned > 0))}")
    print()
    
    # ==========================================================================
    # NEW: Path-2 Analysis
    # ==========================================================================
    print("Step 3a: Computing path-2 (indirect) connections...")
    print("-" * 80)
    
    path2_info = compute_path2_analysis(A_struct_aligned)
    print(f"  Direct (path-1) edges: {path2_info['n_direct_edges']}")
    print(f"  Path-2 only (indirect): {path2_info['n_path2_only']}")
    print(f"  Total reachable in ≤2 hops: {path2_info['n_total_reachable_2hop']}")
    print(f"  Path-2 density: {path2_info['path2_density']:.3f}")
    print()
    
    A_reachable_2 = path2_info['A_reachable_2']
    
    # ==========================================================================
    # Analyze each configuration
    # ==========================================================================
    print("Step 3b: Comparing functional vs structural connectivity...")
    print("-" * 80)
    
    all_path2_comparisons = []
    all_permutation_results = []
    
    for idx, row in results_df.iterrows():
        stimulus = row['stimulus']
        model_type = row['model_type']
        train_split = row['train_split']
        
        config_id = f"{model_type}_{train_split}"
        print(f"\n[{stimulus} - {config_id}]")
        
        # Load functional adjacency
        result_file = RESULTS_DIR / "models" / stimulus / config_id / "result.npz"
        if not result_file.exists():
            print(f"  ⚠️  Result file not found: {result_file}")
            continue
        
        data = np.load(result_file)
        sign_adj = data['sign_adj']
        volatility_adj = data.get('volatility_adj', np.zeros_like(sign_adj))
        
        # Combined functional adjacency (use sign_adj as primary, with absolute values)
        A_functional_full = sign_adj.copy()
        
        # Align functional adjacency to common neurons
        A_functional = A_functional_full[np.ix_(func_indices, func_indices)]
        
        # Compute graph statistics
        func_stats = compute_graph_statistics(A_functional, f"{stimulus}_{config_id}")
        all_graph_stats.append(func_stats)
        
        # Compare edge sets (standard)
        comparison = compare_edge_sets(A_functional, A_struct_aligned, threshold=0.0)
        comparison['stimulus'] = stimulus
        comparison['model_type'] = model_type
        comparison['train_split'] = train_split
        
        # NEW: Random baseline comparison
        n_possible = len(common_neurons) * (len(common_neurons) - 1)
        baseline = compute_random_baseline(
            comparison['n_functional_edges'],
            comparison['n_structural_edges'],
            n_possible
        )
        comparison['random_f1'] = baseline['random_f1']
        comparison['f1_vs_random_ratio'] = comparison['f1_score'] / baseline['random_f1'] if baseline['random_f1'] > 0 else float('inf')
        
        all_comparisons.append(comparison)
        
        print(f"  Precision: {comparison['precision']:.3f}")
        print(f"  Recall: {comparison['recall']:.3f}")
        print(f"  F1-Score: {comparison['f1_score']:.3f} (vs random: {baseline['random_f1']:.3f}, ratio: {comparison['f1_vs_random_ratio']:.2f}x)")
        
        # NEW: Path-2 comparison
        path2_comp = compare_with_path2_ground_truth(A_functional, A_struct_aligned, A_reachable_2)
        path2_comp['stimulus'] = stimulus
        path2_comp['model_type'] = model_type
        path2_comp['train_split'] = train_split
        all_path2_comparisons.append(path2_comp)
        
        print(f"  Path-2 F1: {path2_comp['f1_path2']:.3f} ({path2_comp['pct_fp_explained']:.1f}% of FP explained by indirect paths)")
        
        # NEW: Permutation test (only for best config per stimulus to save time)
        if idx < 4:  # Run for first few configs as examples
            perm_result = run_permutation_test(A_functional, A_struct_aligned, n_permutations=500)
            perm_result['stimulus'] = stimulus
            perm_result['model_type'] = model_type
            perm_result['train_split'] = train_split
            all_permutation_results.append(perm_result)
            
            sig_marker = "**" if perm_result['significant_at_01'] else ("*" if perm_result['significant_at_05'] else "")
            print(f"  Permutation test: p={perm_result['p_value']:.4f} {sig_marker}")
        
        # Compute weight correlation (if structural weights available)
        weight_corr = compute_weight_correlation(A_functional, A_struct_aligned)
        print(f"  Shared edges: {weight_corr['n_shared_edges']}")
        if weight_corr['n_shared_edges'] > 0:
            print(f"  Pearson r: {weight_corr['pearson_r']:.3f}")
            print(f"  Spearman ρ: {weight_corr['spearman_r']:.3f}")
        
        # Save individual plots
        plot_dir = FIGURES_DIR / stimulus / config_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(
            comparison,
            plot_dir / "confusion_matrix.png",
            title=f"{stimulus.capitalize()} - {config_id}"
        )
        
        if weight_corr['n_shared_edges'] > 0:
            plot_weight_correlation(
                A_functional, A_struct_aligned, weight_corr,
                plot_dir / "weight_correlation.png",
                title=f"{stimulus.capitalize()} - {config_id}: Weight Correlation"
            )
    
    # Save comparison results
    comparison_df = pd.DataFrame(all_comparisons)
    comparison_df.to_csv(OUTPUT_DIR / "edge_comparisons.csv", index=False)
    print(f"\n✓ Saved edge comparisons: {OUTPUT_DIR / 'edge_comparisons.csv'}")
    
    # NEW: Save path-2 comparisons
    path2_df = pd.DataFrame(all_path2_comparisons)
    path2_df.to_csv(OUTPUT_DIR / "path2_comparisons.csv", index=False)
    print(f"✓ Saved path-2 comparisons: {OUTPUT_DIR / 'path2_comparisons.csv'}")
    
    # NEW: Save permutation test results
    if all_permutation_results:
        perm_df = pd.DataFrame(all_permutation_results)
        perm_df.to_csv(OUTPUT_DIR / "permutation_tests.csv", index=False)
        print(f"✓ Saved permutation tests: {OUTPUT_DIR / 'permutation_tests.csv'}")
    
    # ==========================================================================
    # NEW: Cross-stimulus consistency analysis
    # ==========================================================================
    print("\nStep 4: Cross-stimulus consistency analysis...")
    print("-" * 80)
    
    consistency = compute_cross_stimulus_consistency(all_comparisons)
    if consistency['consistency_available']:
        print(f"  Stimuli analyzed: {', '.join(consistency['stimuli'])}")
        print("\n  Configuration consistency across stimuli:")
        for cfg in consistency['config_consistency']:
            print(f"    {cfg['model_type']}/{cfg['train_split']}: "
                  f"F1 = {cfg['avg_f1_across_stimuli']:.3f} ± {cfg['std_f1_across_stimuli']:.3f}")
    
    # Save consistency results
    with open(OUTPUT_DIR / "cross_stimulus_consistency.json", 'w') as f:
        json.dump(consistency, f, indent=2, default=str)
    print(f"✓ Saved cross-stimulus consistency: {OUTPUT_DIR / 'cross_stimulus_consistency.json'}")
    
    # Generate summary visualizations
    print("\nStep 5: Generating summary visualizations...")
    print("-" * 80)
    
    plot_degree_distributions(all_graph_stats, FIGURES_DIR / "degree_distributions.png")
    
    # ==========================================================================
    # NEW: Generate extended analysis report
    # ==========================================================================
    print("\nStep 6: Generating extended analysis report...")
    print("-" * 80)
    
    report_lines = [
        "# Extended Connectome Comparison Report",
        "",
        "## Path-2 (Indirect Connection) Analysis",
        "",
        f"- Direct (path-1) edges: {path2_info['n_direct_edges']}",
        f"- Path-2 only (indirect): {path2_info['n_path2_only']}",
        f"- Total reachable in ≤2 hops: {path2_info['n_total_reachable_2hop']}",
        "",
        "### Implication for False Positives",
        "",
    ]
    
    avg_fp_explained = path2_df['pct_fp_explained'].mean()
    report_lines.append(f"On average, **{avg_fp_explained:.1f}%** of false positives are actually path-2 connections.")
    report_lines.append("")
    report_lines.append("This suggests functional connectivity captures indirect influence pathways.")
    report_lines.append("")
    
    report_lines.extend([
        "## Random Baseline Comparison",
        "",
        f"| Config | F1 Score | Random F1 | Improvement |",
        f"|--------|----------|-----------|-------------|",
    ])
    
    for _, row in comparison_df.iterrows():
        report_lines.append(
            f"| {row['stimulus']}/{row['model_type'][:8]}/{row['train_split'][:6]} | "
            f"{row['f1_score']:.3f} | {row['random_f1']:.3f} | {row['f1_vs_random_ratio']:.2f}x |"
        )
    
    report_lines.extend([
        "",
        "## Permutation Test Results",
        "",
    ])
    
    if all_permutation_results:
        for res in all_permutation_results:
            sig = "✓" if res['significant_at_05'] else "✗"
            report_lines.append(f"- {res['stimulus']}/{res['model_type']}/{res['train_split']}: "
                              f"p={res['p_value']:.4f} {sig}")
    
    report_path = REPORTS_DIR / "extended_analysis.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"✓ Saved extended report: {report_path}")
    
    # Summary report
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print()
    print("NEW OUTPUTS:")
    print(f"  - path2_comparisons.csv (indirect connection analysis)")
    print(f"  - permutation_tests.csv (statistical significance)")
    print(f"  - cross_stimulus_consistency.json (edge reproducibility)")
    print(f"  - extended_analysis.md (comprehensive report)")
    print()

if __name__ == "__main__":
    main()
