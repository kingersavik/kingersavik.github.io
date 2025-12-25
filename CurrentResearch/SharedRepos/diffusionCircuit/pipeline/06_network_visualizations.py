#!/usr/bin/env python3
"""
SCRIPT 06: Network Visualizations
=================================

Creates comprehensive network visualizations for SBTG results.

Generates:
- Network diagrams with weighted edges (top 200 strongest)
- Degree distribution plots (in/out/total)
- Hub neuron analysis (top 10 by degree)
- Edge weight distributions (excitatory/inhibitory)
- Node and graph statistics

USAGE:
    python pipeline/06_network_visualizations.py

INPUTS:
    - results/comprehensive_sbtg/models/{stim}/{config}/result.npz

OUTPUTS:
    - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/network_graph.png
    - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/degree_distributions.png
    - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/edge_properties.png
    - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/node_statistics.csv

Expected runtime: ~5 minutes
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_sbtg"
OUTPUT_DIR = RESULTS_DIR / "visualizations"
FIGURES_DIR = OUTPUT_DIR / "figures"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"

for dir_path in [OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# NETWORK CONSTRUCTION
# ============================================================================

def build_networkx_graph(
    sign_adj: np.ndarray,
    mu_hat: np.ndarray,
    neuron_names: List[str],
    threshold: float = 0.0
) -> nx.DiGraph:
    """
    Build NetworkX directed graph from SBTG adjacency.
    
    Args:
        sign_adj: Binary sign adjacency matrix
        mu_hat: Coupling weight matrix
        neuron_names: List of neuron names
        threshold: Minimum |mu_hat| value to include edge
    
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add all neurons as nodes
    G.add_nodes_from(neuron_names)
    
    # Add edges
    n = len(neuron_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            if sign_adj[i, j] != 0 and abs(mu_hat[i, j]) >= threshold:
                G.add_edge(
                    neuron_names[j],  # source (j affects i)
                    neuron_names[i],  # target
                    weight=float(mu_hat[i, j]),
                    abs_weight=float(abs(mu_hat[i, j])),
                    sign=int(sign_adj[i, j])
                )
    
    return G

# ============================================================================
# GRAPH STATISTICS
# ============================================================================

def compute_node_statistics(G: nx.DiGraph) -> pd.DataFrame:
    """Compute per-node graph statistics."""
    
    nodes = list(G.nodes())
    stats = []
    
    for node in nodes:
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        
        # Weighted strengths
        in_edges = [(u, v, d) for u, v, d in G.in_edges(node, data=True)]
        out_edges = [(u, v, d) for u, v, d in G.out_edges(node, data=True)]
        
        in_strength = sum(abs(d['weight']) for _, _, d in in_edges)
        out_strength = sum(abs(d['weight']) for _, _, d in out_edges)
        
        # Excitatory/Inhibitory
        in_exc = sum(1 for _, _, d in in_edges if d['sign'] > 0)
        in_inh = sum(1 for _, _, d in in_edges if d['sign'] < 0)
        out_exc = sum(1 for _, _, d in out_edges if d['sign'] > 0)
        out_inh = sum(1 for _, _, d in out_edges if d['sign'] < 0)
        
        stats.append({
            'neuron': node,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': in_degree + out_degree,
            'in_strength': in_strength,
            'out_strength': out_strength,
            'total_strength': in_strength + out_strength,
            'in_excitatory': in_exc,
            'in_inhibitory': in_inh,
            'out_excitatory': out_exc,
            'out_inhibitory': out_inh
        })
    
    return pd.DataFrame(stats)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_network_graph(
    G: nx.DiGraph,
    node_stats: pd.DataFrame,
    output_path: Path,
    title: str = "Functional Connectivity Network",
    top_k_edges: int = 200,
    seed: int = 42
):
    """
    Plot network with spring layout, colored by sign, sized by strength.
    
    Args:
        G: NetworkX graph
        node_stats: Per-node statistics
        output_path: Where to save figure
        title: Plot title
        top_k_edges: Number of strongest edges to show
        seed: Random seed for layout
    """
    if G.number_of_edges() == 0:
        print(f"  Skipping {output_path.name}: no edges")
        return
    
    # Filter to top edges
    edges_sorted = sorted(
        G.edges(data=True),
        key=lambda x: x[2]['abs_weight'],
        reverse=True
    )
    
    if top_k_edges > 0 and len(edges_sorted) > top_k_edges:
        edges_sorted = edges_sorted[:top_k_edges]
        subG = nx.DiGraph()
        subG.add_nodes_from(G.nodes())
        subG.add_edges_from(edges_sorted)
    else:
        subG = G
    
    # Layout
    pos = nx.spring_layout(subG, seed=seed, k=0.5, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Node sizes by total strength
    strengths = node_stats.set_index('neuron')['total_strength']
    strengths = strengths.reindex(subG.nodes()).fillna(0.0)
    max_strength = strengths.max() if strengths.max() > 0 else 1.0
    node_sizes = 100 + 400 * (strengths / max_strength)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        node_size=node_sizes.to_numpy(),
        node_color='#34495e',
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        subG, pos, ax=ax,
        font_size=6,
        font_weight='bold',
        font_color='white'
    )
    
    # Draw edges colored by sign
    weights = np.array([d['weight'] for _, _, d in edges_sorted])
    abs_weights = np.array([d['abs_weight'] for _, _, d in edges_sorted])
    
    # Normalize for colors and widths
    max_abs = abs_weights.max() if abs_weights.max() > 0 else 1.0
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    cmap = plt.colormaps['RdBu_r']  # Red=inhibitory, Blue=excitatory
    
    edge_colors = cmap(norm(weights))
    edge_widths = 0.5 + 2.5 * (abs_weights / max_abs)
    
    # Draw edges as line collection
    segments = []
    colors = []
    widths = []
    
    for (src, tgt, _), color, width in zip(edges_sorted, edge_colors, edge_widths):
        if src in pos and tgt in pos:
            segments.append([pos[src], pos[tgt]])
            colors.append(color)
            widths.append(width)
    
    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=widths,
        alpha=0.6,
        zorder=1
    )
    ax.add_collection(lc)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Coupling Weight (mu_hat)', rotation=270, labelpad=20, fontsize=12)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#3498db', label='Excitatory (positive)'),
        mpatches.Patch(color='#e74c3c', label='Inhibitory (negative)'),
        plt.Line2D([0], [0], color='gray', linewidth=3, label=f'Top {min(top_k_edges, len(edges_sorted))} edges'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_degree_distributions(
    node_stats: pd.DataFrame,
    output_path: Path,
    title: str = "Degree Distributions"
):
    """Plot in-degree, out-degree, and total degree distributions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # In-degree
    ax = axes[0, 0]
    ax.hist(node_stats['in_degree'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    ax.set_xlabel("In-Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"In-Degree (mean={node_stats['in_degree'].mean():.1f})")
    ax.grid(alpha=0.3)
    
    # Out-degree
    ax = axes[0, 1]
    ax.hist(node_stats['out_degree'], bins=20, edgecolor='black', alpha=0.7, color='#e74c3c')
    ax.set_xlabel("Out-Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"Out-Degree (mean={node_stats['out_degree'].mean():.1f})")
    ax.grid(alpha=0.3)
    
    # Total degree
    ax = axes[1, 0]
    ax.hist(node_stats['total_degree'], bins=20, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax.set_xlabel("Total Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"Total Degree (mean={node_stats['total_degree'].mean():.1f})")
    ax.grid(alpha=0.3)
    
    # Top hubs
    ax = axes[1, 1]
    top_hubs = node_stats.nlargest(10, 'total_degree')[['neuron', 'total_degree']]
    ax.barh(range(10), top_hubs['total_degree'].values, color='#9b59b6')
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_hubs['neuron'].values, fontsize=9)
    ax.set_xlabel("Total Degree")
    ax.set_title("Top 10 Hub Neurons")
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_edge_properties(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Edge Properties"
):
    """Plot edge weight distributions and exc/inh balance."""
    
    edges = list(G.edges(data=True))
    if len(edges) == 0:
        print(f"  Skipping {output_path.name}: no edges")
        return
    
    weights = [d['weight'] for _, _, d in edges]
    abs_weights = [d['abs_weight'] for _, _, d in edges]
    signs = [d['sign'] for _, _, d in edges]
    
    exc_count = sum(1 for s in signs if s > 0)
    inh_count = sum(1 for s in signs if s < 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Weight distribution (signed)
    ax = axes[0, 0]
    ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='#34495e')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel("Coupling Weight")
    ax.set_ylabel("Count")
    ax.set_title("Weight Distribution (Signed)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Absolute weight distribution
    ax = axes[0, 1]
    ax.hist(abs_weights, bins=50, edgecolor='black', alpha=0.7, color='#16a085')
    ax.set_xlabel("Absolute Coupling Weight")
    ax.set_ylabel("Count")
    ax.set_title(f"Weight Magnitude (median={np.median(abs_weights):.4f})")
    ax.grid(alpha=0.3)
    
    # Excitatory vs Inhibitory
    ax = axes[1, 0]
    counts = [exc_count, inh_count]
    labels = [f'Excitatory\n(n={exc_count})', f'Inhibitory\n(n={inh_count})']
    colors = ['#3498db', '#e74c3c']
    ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel("Edge Count")
    ax.set_title("Excitatory vs Inhibitory Edges")
    ax.grid(axis='y', alpha=0.3)
    
    # Cumulative weight distribution
    ax = axes[1, 1]
    sorted_abs = np.sort(abs_weights)[::-1]
    cumsum = np.cumsum(sorted_abs)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    ax.plot(range(len(sorted_abs)), cumsum_pct, linewidth=2, color='#e67e22')
    ax.axhline(80, color='red', linestyle='--', label='80%')
    ax.set_xlabel("Number of Strongest Edges")
    ax.set_ylabel("Cumulative Weight (%)")
    ax.set_title("Cumulative Weight Distribution")
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
    """Generate visualizations for all SBTG results."""
    
    print("=" * 80)
    print("NETWORK VISUALIZATION GENERATION")
    print("=" * 80)
    print()
    
    # Load results
    results_csv = RESULTS_DIR / "all_results.csv"
    if not results_csv.exists():
        print(f"⚠️  Results file not found: {results_csv}")
        print("   Please run 05_comprehensive_sbtg_analysis.py first")
        return
    
    results_df = pd.read_csv(results_csv)
    print(f"✓ Found {len(results_df)} configurations")
    print()
    
    # Process each configuration
    for idx, row in results_df.iterrows():
        stimulus = row['stimulus']
        model_type = row['model_type']
        train_split = row['train_split']
        
        config_id = f"{model_type}_{train_split}"
        print(f"\n[{stimulus} - {config_id}]")
        
        # Load result files
        result_dir = RESULTS_DIR / "models" / stimulus / config_id
        result_file = result_dir / "result.npz"
        neuron_file = result_dir / "neuron_names.json"
        
        if not result_file.exists() or not neuron_file.exists():
            print(f"  ⚠️  Missing files in {result_dir}")
            continue
        
        # Load data
        data = np.load(result_file)
        with open(neuron_file, 'r') as f:
            neuron_names = json.load(f)
        
        sign_adj = data['sign_adj']
        mu_hat = data.get('mu_hat', sign_adj)  # Use sign_adj if mu_hat missing
        
        # Build graph
        G = build_networkx_graph(sign_adj, mu_hat, neuron_names, threshold=0.0)
        
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        if G.number_of_edges() == 0:
            print(f"  ⚠️  No edges to visualize")
            continue
        
        # Compute statistics
        node_stats = compute_node_statistics(G)
        
        # Create output directory
        viz_dir = FIGURES_DIR / stimulus / config_id
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plot_network_graph(
            G, node_stats,
            viz_dir / "network_graph.png",
            title=f"{stimulus.capitalize()} - {model_type} ({train_split})",
            top_k_edges=200
        )
        
        plot_degree_distributions(
            node_stats,
            viz_dir / "degree_distributions.png",
            title=f"{stimulus.capitalize()} - {model_type} ({train_split}): Degrees"
        )
        
        plot_edge_properties(
            G,
            viz_dir / "edge_properties.png",
            title=f"{stimulus.capitalize()} - {model_type} ({train_split}): Edges"
        )
        
        # Save statistics
        node_stats.to_csv(viz_dir / "node_statistics.csv", index=False)
        
        # Graph-level statistics
        graph_stats = {
            "stimulus": stimulus,
            "model_type": model_type,
            "train_split": train_split,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_in_degree": node_stats['in_degree'].mean(),
            "avg_out_degree": node_stats['out_degree'].mean(),
            "avg_total_degree": node_stats['total_degree'].mean(),
            "n_excitatory": int(sum(1 for _, _, d in G.edges(data=True) if d['sign'] > 0)),
            "n_inhibitory": int(sum(1 for _, _, d in G.edges(data=True) if d['sign'] < 0))
        }
        
        with open(viz_dir / "graph_statistics.json", 'w') as f:
            json.dump(graph_stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"Outputs saved to: {FIGURES_DIR}")
    print()

if __name__ == "__main__":
    main()
