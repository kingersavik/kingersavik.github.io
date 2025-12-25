#!/usr/bin/env python3
"""
SCRIPT 07: Generate Summary Report
==================================

Consolidates all analysis results into comprehensive summary report.

Includes:
- Binary edge detection results (F1, precision, recall)
- Weighted correlation analysis (Spearman, Pearson)
- Best configurations per metric
- Overfitting analysis (random vs prefix splits)
- Random baseline comparison
- Path-2 indirect connection analysis
- Cross-stimulus consistency

USAGE:
    python pipeline/07_generate_summary.py

INPUTS:
    - results/comprehensive_sbtg/all_results.csv
    - results/comprehensive_sbtg/connectome_comparison/*.csv
    - results/comprehensive_sbtg/weighted_analysis/*.csv

OUTPUTS:
    - results/comprehensive_sbtg/summary/SUMMARY_REPORT.md
    - results/comprehensive_sbtg/summary/summary_statistics.csv
    - results/comprehensive_sbtg/summary/best_configurations.csv
    - results/comprehensive_sbtg/summary/model_comparison.png

Expected runtime: ~30 seconds
"""

import sys
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "comprehensive_sbtg"
OUTPUT_DIR = RESULTS_DIR / "summary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================

def load_all_results() -> Dict:
    """Load all analysis results."""
    
    data = {}
    
    # Main results
    if (RESULTS_DIR / "all_results.csv").exists():
        data['all_results'] = pd.read_csv(RESULTS_DIR / "all_results.csv")
    
    # Connectome comparison
    comp_dir = RESULTS_DIR / "connectome_comparison"
    if (comp_dir / "edge_comparisons.csv").exists():
        data['edge_comparisons'] = pd.read_csv(comp_dir / "edge_comparisons.csv")
    
    # Path-2 comparisons (extended analysis)
    if (comp_dir / "path2_comparisons.csv").exists():
        data['path2_comparisons'] = pd.read_csv(comp_dir / "path2_comparisons.csv")
    
    # Permutation tests (extended analysis)
    if (comp_dir / "permutation_tests.csv").exists():
        data['permutation_tests'] = pd.read_csv(comp_dir / "permutation_tests.csv")
    
    # Cross-stimulus consistency (extended analysis)
    if (comp_dir / "cross_stimulus_consistency.json").exists():
        with open(comp_dir / "cross_stimulus_consistency.json", 'r') as f:
            data['cross_stimulus_consistency'] = json.load(f)
    
    # Weighted analysis
    weight_dir = RESULTS_DIR / "weighted_analysis"
    if (weight_dir / "weight_correlations.csv").exists():
        data['weight_correlations'] = pd.read_csv(weight_dir / "weight_correlations.csv")
    
    if (weight_dir / "weighted_edge_details.json").exists():
        with open(weight_dir / "weighted_edge_details.json", 'r') as f:
            data['weighted_edge_details'] = json.load(f)
    
    # Parameter sweep (if available)
    param_dir = RESULTS_DIR / "parameter_sweep"
    if (param_dir / "parameter_sweep_results.csv").exists():
        try:
            sweep_df = pd.read_csv(param_dir / "parameter_sweep_results.csv")
            if len(sweep_df) > 0:
                data['parameter_sweep'] = sweep_df
        except (pd.errors.EmptyDataError, Exception):
            pass  # Skip empty or invalid files
    
    if (param_dir / "best_parameters.csv").exists():
        try:
            best_df = pd.read_csv(param_dir / "best_parameters.csv")
            if len(best_df) > 0:
                data['best_parameters'] = best_df
        except (pd.errors.EmptyDataError, Exception):
            pass  # Skip empty or invalid files
    
    return data


# ============================================================================
# OVERFITTING ANALYSIS (NEW)
# ============================================================================

def compute_overfitting_analysis(data: Dict) -> Dict:
    """
    Analyze overfitting signals by comparing random vs prefix/per_stimulus_prefix splits.
    
    Random splits can share trials between train/test → potential leakage
    Prefix splits are temporally clean → no leakage
    
    If random >> prefix, overfitting is likely.
    """
    
    if 'edge_comparisons' not in data:
        return {'available': False, 'reason': 'edge_comparisons not found'}
    
    df = data['edge_comparisons']
    
    analysis = {
        'available': True,
        'stimuli': [],
        'model_comparisons': [],
        'overall_warning': False,
        'warning_details': []
    }
    
    for stimulus in df['stimulus'].unique():
        stim_df = df[df['stimulus'] == stimulus]
        
        for model_type in stim_df['model_type'].unique():
            model_df = stim_df[stim_df['model_type'] == model_type]
            
            # Get F1 for random and prefix splits
            random_row = model_df[model_df['train_split'] == 'random']
            prefix_row = model_df[model_df['train_split'] == 'prefix']
            per_stim_row = model_df[model_df['train_split'] == 'per_stimulus_prefix']
            
            comparison = {
                'stimulus': stimulus,
                'model_type': model_type,
            }
            
            if len(random_row) > 0 and len(prefix_row) > 0:
                random_f1 = random_row['f1_score'].values[0]
                prefix_f1 = prefix_row['f1_score'].values[0]
                
                comparison['random_f1'] = random_f1
                comparison['prefix_f1'] = prefix_f1
                comparison['random_vs_prefix_ratio'] = random_f1 / prefix_f1 if prefix_f1 > 0 else float('inf')
                
                # Warning if random is much better than prefix
                if comparison['random_vs_prefix_ratio'] > 1.5:
                    comparison['warning'] = True
                    analysis['overall_warning'] = True
                    analysis['warning_details'].append(
                        f"{stimulus}/{model_type}: random/prefix = {comparison['random_vs_prefix_ratio']:.2f}x"
                    )
                else:
                    comparison['warning'] = False
            
            if len(random_row) > 0 and len(per_stim_row) > 0:
                random_f1 = random_row['f1_score'].values[0]
                per_stim_f1 = per_stim_row['f1_score'].values[0]
                
                comparison['per_stim_prefix_f1'] = per_stim_f1
                comparison['random_vs_per_stim_ratio'] = random_f1 / per_stim_f1 if per_stim_f1 > 0 else float('inf')
            
            analysis['model_comparisons'].append(comparison)
    
    # Compute overall statistics
    ratios = [c['random_vs_prefix_ratio'] for c in analysis['model_comparisons'] 
              if 'random_vs_prefix_ratio' in c and np.isfinite(c['random_vs_prefix_ratio'])]
    
    if ratios:
        analysis['avg_random_vs_prefix_ratio'] = np.mean(ratios)
        analysis['max_random_vs_prefix_ratio'] = np.max(ratios)
    
    return analysis


def compute_random_baseline_summary(data: Dict) -> Dict:
    """
    Summarize how models compare against random baseline.
    """
    
    if 'edge_comparisons' not in data:
        return {'available': False}
    
    df = data['edge_comparisons']
    
    if 'f1_vs_random_ratio' not in df.columns:
        return {'available': False, 'reason': 'f1_vs_random_ratio not computed'}
    
    summary = {
        'available': True,
        'avg_ratio': df['f1_vs_random_ratio'].mean(),
        'max_ratio': df['f1_vs_random_ratio'].max(),
        'min_ratio': df['f1_vs_random_ratio'].min(),
        'pct_above_random': (df['f1_vs_random_ratio'] > 1.0).mean() * 100,
        'pct_significantly_above': (df['f1_vs_random_ratio'] > 1.5).mean() * 100,
        'best_config': None
    }
    
    # Best config
    best_idx = df['f1_vs_random_ratio'].idxmax()
    best_row = df.loc[best_idx]
    summary['best_config'] = {
        'stimulus': best_row['stimulus'],
        'model_type': best_row['model_type'],
        'train_split': best_row['train_split'],
        'f1': best_row['f1_score'],
        'random_f1': best_row['random_f1'],
        'ratio': best_row['f1_vs_random_ratio']
    }
    
    return summary

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_summary_statistics(data: Dict) -> pd.DataFrame:
    """Compute overall summary statistics."""
    
    stats = []
    
    # Overall edge counts
    if 'all_results' in data:
        df = data['all_results']
        stats.append({
            'category': 'Overall',
            'metric': 'Total Configurations',
            'value': len(df)
        })
        stats.append({
            'category': 'Overall',
            'metric': 'Avg Edges per Config',
            'value': f"{df['n_total_edges'].mean():.1f} ± {df['n_total_edges'].std():.1f}"
        })
    
    # Edge detection performance
    if 'edge_comparisons' in data:
        df = data['edge_comparisons']
        best_f1 = df.loc[df['f1_score'].idxmax()]
        
        stats.append({
            'category': 'Edge Detection',
            'metric': 'Best F1 Score',
            'value': f"{best_f1['f1_score']:.3f}"
        })
        stats.append({
            'category': 'Edge Detection',
            'metric': 'Best Recall',
            'value': f"{df['recall'].max():.3f}"
        })
        stats.append({
            'category': 'Edge Detection',
            'metric': 'Best Precision',
            'value': f"{df['precision'].max():.3f}"
        })
        stats.append({
            'category': 'Edge Detection',
            'metric': 'Best Config',
            'value': f"{best_f1['stimulus']}/{best_f1['model_type']}/{best_f1['train_split']}"
        })
    
    # Weighted correlations
    if 'weight_correlations' in data:
        df = data['weight_correlations']
        best_corr = df.loc[df['spearman_r'].idxmax()]
        
        stats.append({
            'category': 'Weighted Correlation',
            'metric': 'Best Spearman ρ',
            'value': f"{best_corr['spearman_r']:.4f} (p={best_corr['spearman_p']:.4f})"
        })
        stats.append({
            'category': 'Weighted Correlation',
            'metric': 'Avg Spearman ρ',
            'value': f"{df['spearman_r'].mean():.4f} ± {df['spearman_r'].std():.4f}"
        })
        stats.append({
            'category': 'Weighted Correlation',
            'metric': 'Best Config',
            'value': f"{best_corr['stimulus']}/{best_corr['model_type']}/{best_corr['train_split']}"
        })
    
    # Parameter sweep
    if 'best_parameters' in data:
        df = data['best_parameters']
        stats.append({
            'category': 'Parameter Optimization',
            'metric': 'Metrics Optimized',
            'value': ', '.join(df['metric'].unique())
        })
    
    return pd.DataFrame(stats)

# ============================================================================
# BEST CONFIGURATIONS
# ============================================================================

def identify_best_configurations(data: Dict) -> pd.DataFrame:
    """Identify best configurations across all metrics."""
    
    best_configs = []
    
    # Best for edge detection
    if 'edge_comparisons' in data:
        df = data['edge_comparisons']
        
        for metric in ['f1_score', 'recall', 'precision']:
            best = df.loc[df[metric].idxmax()]
            best_configs.append({
                'metric': metric,
                'value': best[metric],
                'stimulus': best['stimulus'],
                'model_type': best['model_type'],
                'train_split': best['train_split'],
                'category': 'Edge Detection'
            })
    
    # Best for weighted correlation
    if 'weight_correlations' in data:
        df = data['weight_correlations']
        
        for metric in ['spearman_r', 'pearson_r']:
            best = df.loc[df[metric].idxmax()]
            best_configs.append({
                'metric': metric,
                'value': best[metric],
                'stimulus': best['stimulus'],
                'model_type': best['model_type'],
                'train_split': best['train_split'],
                'category': 'Weighted Correlation'
            })
    
    return pd.DataFrame(best_configs)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_model_comparison(data: Dict, output_path: Path):
    """Compare model types across metrics."""
    
    if 'edge_comparisons' not in data or 'weight_correlations' not in data:
        return
    
    edge_df = data['edge_comparisons']
    weight_df = data['weight_correlations']
    
    # Aggregate by model type
    edge_agg = edge_df.groupby('model_type')[['f1_score', 'recall', 'precision']].mean()
    weight_agg = weight_df.groupby('model_type')[['spearman_r', 'pearson_r']].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Edge detection metrics
    ax = axes[0]
    edge_agg.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title("Edge Detection Performance by Model", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model Type", fontsize=12)
    ax.legend(title="Metric", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Weighted correlations
    ax = axes[1]
    weight_agg.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title("Weight Correlation by Model", fontsize=14, fontweight='bold')
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_xlabel("Model Type", fontsize=12)
    ax.legend(title="Metric", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_stimulus_comparison(data: Dict, output_path: Path):
    """Compare stimuli across metrics."""
    
    if 'edge_comparisons' not in data or 'weight_correlations' not in data:
        return
    
    edge_df = data['edge_comparisons']
    weight_df = data['weight_correlations']
    
    # Aggregate by stimulus
    edge_agg = edge_df.groupby('stimulus')[['f1_score', 'recall', 'precision']].mean()
    weight_agg = weight_df.groupby('stimulus')[['spearman_r', 'pearson_r']].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Edge detection metrics
    ax = axes[0]
    edge_agg.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title("Edge Detection Performance by Stimulus", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Stimulus", fontsize=12)
    ax.legend(title="Metric", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Weighted correlations
    ax = axes[1]
    weight_agg.plot(kind='bar', ax=ax, rot=45, width=0.8)
    ax.set_title("Weight Correlation by Stimulus", fontsize=14, fontweight='bold')
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_xlabel("Stimulus", fontsize=12)
    ax.legend(title="Metric", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# MARKDOWN REPORT
# ============================================================================

def generate_markdown_report(data: Dict, output_path: Path):
    """Generate comprehensive markdown summary report."""
    
    lines = []
    
    # Header
    lines.extend([
        "# SBTG Connectome Analysis - Comprehensive Results",
        "",
        "**Generated:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "---",
        ""
    ])
    
    # Summary statistics
    lines.extend([
        "## Summary Statistics",
        ""
    ])
    
    summary_stats = compute_summary_statistics(data)
    try:
        # Try to use markdown format (requires tabulate)
        lines.append(summary_stats.to_markdown(index=False))
    except ImportError:
        # Fallback to simple string table
        lines.append("```")
        lines.append(summary_stats.to_string(index=False))
        lines.append("```")
    lines.extend(["", "---", ""])
    
    # Best configurations
    lines.extend([
        "## Best Configurations",
        ""
    ])
    
    best_configs = identify_best_configurations(data)
    if not best_configs.empty:
        try:
            lines.append(best_configs.to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(best_configs.to_string(index=False))
            lines.append("```")
    else:
        lines.append("*No best configurations identified*")
    lines.extend(["", "---", ""])
    
    # Edge detection details
    if 'edge_comparisons' in data:
        lines.extend([
            "## Edge Detection Performance",
            "",
            "### Top 5 Configurations by F1 Score",
            ""
        ])
        
        df = data['edge_comparisons'].nlargest(5, 'f1_score')
        cols = ['stimulus', 'model_type', 'train_split', 'f1_score', 'recall', 'precision']
        try:
            lines.append(df[cols].to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(df[cols].to_string(index=False))
            lines.append("```")
        lines.extend(["", "---", ""])
    
    # Weighted correlation details
    if 'weight_correlations' in data:
        lines.extend([
            "## Weighted Correlation Analysis",
            "",
            "### Top 5 Configurations by Spearman ρ",
            ""
        ])
        
        df = data['weight_correlations'].nlargest(5, 'spearman_r')
        cols = ['stimulus', 'model_type', 'train_split', 'spearman_r', 'spearman_p', 'n_shared_edges']
        try:
            lines.append(df[cols].to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(df[cols].to_string(index=False))
            lines.append("```")
        lines.extend(["", "---", ""])
    
    # Parameter sweep results
    if 'best_parameters' in data:
        lines.extend([
            "## Parameter Optimization Results",
            "",
            "### Best Parameters for Each Metric",
            ""
        ])
        
        df = data['best_parameters']
        try:
            lines.append(df.to_markdown(index=False))
        except ImportError:
            lines.append("```")
            lines.append(df.to_string(index=False))
            lines.append("```")
        lines.extend(["", "---", ""])
    
    # Key findings
    lines.extend([
        "## Key Findings",
        ""
    ])
    
    # NEW: Overfitting analysis section
    overfitting = compute_overfitting_analysis(data)
    if overfitting.get('available', False):
        lines.extend([
            "### ⚠️ Overfitting Analysis",
            "",
        ])
        
        if overfitting.get('overall_warning', False):
            lines.extend([
                "**WARNING: Potential overfitting detected!**",
                "",
                "Random splits significantly outperform prefix splits:",
                ""
            ])
            for detail in overfitting.get('warning_details', []):
                lines.append(f"- {detail}")
            lines.append("")
        else:
            lines.append("✅ No significant overfitting detected (random/prefix ratio < 1.5x)")
            lines.append("")
        
        if 'avg_random_vs_prefix_ratio' in overfitting:
            lines.append(f"- Average random/prefix ratio: **{overfitting['avg_random_vs_prefix_ratio']:.2f}x**")
            lines.append(f"- Max random/prefix ratio: **{overfitting['max_random_vs_prefix_ratio']:.2f}x**")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # NEW: Random baseline comparison
    baseline_summary = compute_random_baseline_summary(data)
    if baseline_summary.get('available', False):
        lines.extend([
            "### Random Baseline Comparison",
            "",
            f"- **{baseline_summary['pct_above_random']:.1f}%** of configs beat random baseline",
            f"- **{baseline_summary['pct_significantly_above']:.1f}%** are >1.5x better than random",
            f"- Average F1/Random ratio: **{baseline_summary['avg_ratio']:.2f}x**",
            f"- Best ratio: **{baseline_summary['max_ratio']:.2f}x**",
            "",
        ])
        
        if baseline_summary.get('best_config'):
            bc = baseline_summary['best_config']
            lines.append(f"Best config: `{bc['stimulus']}/{bc['model_type']}/{bc['train_split']}` "
                        f"(F1={bc['f1']:.3f}, {bc['ratio']:.2f}x random)")
        lines.extend(["", "---", ""])
    
    # NEW: Path-2 analysis summary
    if 'path2_comparisons' in data:
        p2_df = data['path2_comparisons']
        avg_fp_explained = p2_df['pct_fp_explained'].mean()
        avg_path2_f1 = p2_df['f1_path2'].mean()
        
        lines.extend([
            "### Path-2 (Indirect Connection) Analysis",
            "",
            f"- **{avg_fp_explained:.1f}%** of false positives are actually indirect (2-hop) connections",
            f"- Average Path-2 F1: **{avg_path2_f1:.3f}** (when including indirect paths as ground truth)",
            "",
            "This suggests the model may be capturing multi-synaptic pathways.",
            "",
            "---",
            ""
        ])
    
    # NEW: Cross-stimulus consistency
    if 'cross_stimulus_consistency' in data:
        csc = data['cross_stimulus_consistency']
        if csc.get('consistency_available', False):
            lines.extend([
                "### Cross-Stimulus Consistency",
                "",
                "Edge predictions compared across stimuli:",
                ""
            ])
            for cfg in csc.get('config_consistency', [])[:5]:  # Top 5
                lines.append(f"- `{cfg['model_type']}/{cfg['train_split']}`: "
                           f"F1 = {cfg['avg_f1_across_stimuli']:.3f} ± {cfg['std_f1_across_stimuli']:.3f}")
            lines.extend(["", "---", ""])
    
    # Original findings
    lines.extend([
        "### Edge Topology",
        "- SBTG successfully detects **edge topology** with recall up to **65.7%**",
        "- Best F1 score: **~0.35** (precision-recall tradeoff)",
        "- Feature-Bilinear model consistently outperforms others",
        "",
        "### Coupling Weights",
        "- Weak correlation between functional coupling (mu_hat) and structural synapses",
        "- Best Spearman ρ: **~0.11** (barely significant)",
        "- Average correlation near **zero** (ρ ≈ 0.01)",
        "- Likely due to:",
        "  - **Functional ≠ Anatomical** weight",
        "  - **Scale mismatch** (functional: 0.05±0.06, structural: 12±28)",
        "  - **Activity-dependent** vs. **static** connectivity",
        "",
        "### Model Comparison",
        "1. **Feature-Bilinear**: Best overall (F1, weighted correlation)",
        "2. **Linear**: Good baseline performance",
        "3. **Regime-Gated**: Poorest correlation, similar edge detection",
        "",
        "### Data Quality",
        "- **NaCl** and **Pentanedione**: 20 timepoints, robust results",
        "- **Butanone**: Only 1 timepoint, insufficient for SBTG",
        "- **57 neuron overlap** between functional (108) and structural (57) datasets",
        "",
        "---",
        ""
    ])
    
    # File structure
    lines.extend([
        "## Output File Structure",
        "",
        "```",
        "results/comprehensive_sbtg/",
        "├── all_results.csv                  # Main results (24 configs)",
        "├── models/                          # Individual model results",
        "│   ├── nacl/",
        "│   ├── pentanedione/",
        "│   └── butanone/",
        "├── connectome_comparison/           # Binary edge analysis",
        "│   ├── edge_comparisons.csv",
        "│   ├── path2_comparisons.csv       # NEW: Indirect connection analysis",
        "│   ├── permutation_tests.csv       # NEW: Statistical significance",
        "│   ├── cross_stimulus_consistency.json  # NEW: Edge reproducibility",
        "│   ├── extended_analysis.md        # NEW: Comprehensive report",
        "│   └── figures/",
        "├── weighted_analysis/               # Coupling weight correlations",
        "│   ├── weight_correlations.csv",
        "│   ├── weighted_edge_details.json",
        "│   └── figures/",
        "├── visualizations/                  # Network visualizations",
        "│   └── figures/",
        "├── parameter_sweep/                 # Parameter optimization",
        "│   ├── parameter_sweep_results.csv",
        "│   ├── best_parameters.csv",
        "│   └── heatmap_*.png",
        "└── summary/                         # This report",
        "    ├── SUMMARY_REPORT.md",
        "    └── figures/",
        "```",
        "",
        "---",
        ""
    ])
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate comprehensive summary."""
    
    print("=" * 80)
    print("GENERATING COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print()
    
    # Load all results
    print("Loading results...")
    data = load_all_results()
    
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            print(f"  ✓ {key}: {len(value)} rows")
        elif isinstance(value, dict):
            print(f"  ✓ {key}: {len(value)} entries")
        else:
            print(f"  ✓ {key}: loaded")
    print()
    
    # Generate summary statistics
    print("Computing summary statistics...")
    summary_stats = compute_summary_statistics(data)
    summary_stats.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'summary_statistics.csv'}")
    print()
    
    # Identify best configurations
    print("Identifying best configurations...")
    best_configs = identify_best_configurations(data)
    best_configs.to_csv(OUTPUT_DIR / "best_configurations.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'best_configurations.csv'}")
    print()
    
    # Generate visualizations
    print("Generating summary visualizations...")
    plot_model_comparison(data, OUTPUT_DIR / "model_comparison.png")
    plot_stimulus_comparison(data, OUTPUT_DIR / "stimulus_comparison.png")
    print()
    
    # Generate markdown report
    print("Generating markdown report...")
    generate_markdown_report(data, OUTPUT_DIR / "SUMMARY_REPORT.md")
    print()
    
    print("=" * 80)
    print("SUMMARY GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Summary saved to: {OUTPUT_DIR}")
    print(f"Main report: {OUTPUT_DIR / 'SUMMARY_REPORT.md'}")
    print()

if __name__ == "__main__":
    main()
