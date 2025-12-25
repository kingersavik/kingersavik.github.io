#!/usr/bin/env python3
"""
SCRIPT 03: Comprehensive SBTG Analysis
======================================

This script performs exhaustive SBTG model training and evaluation across:
- 3 model architectures: linear, feature_bilinear, regime_gated
- 4 train/test strategies: prefix, per_stimulus_prefix, random, odd_even
- 2 stimuli: nacl, pentanedione (butanone skipped - only 1 timepoint)

Uses optimized hyperparameters from script 02 (hyperparameter_search) if available.

Outputs comprehensive results with:
- Dataset statistics and summaries
- Model comparisons across all dimensions
- Saved results for downstream analysis

USAGE:
    python pipeline/03_comprehensive_analysis.py

INPUTS:
    - data/Head_Activity_OH16230.mat (NeuroPAL recordings)
    - data/Head_Activity_OH15500.mat (NaCl recordings)
    - results/hyperparameter_search/best_hyperparameters.json (optional, from script 02)

OUTPUTS:
    - results/comprehensive_sbtg/all_results.csv (24 rows)
    - results/comprehensive_sbtg/models/{stim}/{config}/result.npz
    - results/comprehensive_sbtg/models/{stim}/{config}/neuron_names.json
    - results/comprehensive_sbtg/figures/*.png

Expected runtime: ~15-25 minutes (CPU)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import spearmanr
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sbtg import SBTGStructuredVolatilityEstimator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comprehensive_sbtg"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create output directories
for dir_path in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations to test
MODEL_CONFIGS = {
    "linear": {
        "model_type": "linear",
        "description": "Linear coupling model (baseline)"
    },
    "feature_bilinear": {
        "model_type": "feature_bilinear",
        "feature_dim": 16,
        "feature_hidden_dim": 64,
        "feature_num_layers": 2,
        "description": "Feature-based bilinear model (learned features)"
    },
    "regime_gated": {
        "model_type": "regime_gated",
        "num_regimes": 2,
        "gate_hidden_dim": 64,
        "gate_num_layers": 2,
        "description": "Regime-gated model (context switching)"
    }
}

# Train/test split strategies
SPLIT_STRATEGIES = {
    "prefix": "First train_frac of pooled windows",
    "per_stimulus_prefix": "First train_frac within each stimulus segment",
    "random": "Random subset of windows",
    "odd_even": "Alternating windows by time parity"
}

# Stimuli (butanone excluded - only 1 timepoint, insufficient for SBTG)
STIMULI = ["nacl", "pentanedione"]

# Temporal phase divisions
PHASE_DIVISIONS = [4, 6]

# Path to optimized hyperparameters from script 02
HYPERPARAM_SEARCH_DIR = PROJECT_ROOT / "results" / "hyperparameter_search"
BEST_HYPERPARAMS_FILE = HYPERPARAM_SEARCH_DIR / "best_hyperparameters.json"


def load_optimized_hyperparams() -> Dict:
    """
    Load optimized hyperparameters from script 02 if available.
    Falls back to defaults if not found.
    """
    default_params = {
        "window_length": 2,  # Required for structured SBTG
        "dsm_hidden_dim": 128,
        "dsm_num_layers": 3,
        "dsm_noise_std": 0.1,
        "dsm_epochs": 200,
        "dsm_batch_size": 128,
        "dsm_lr": 1e-3,
        "train_frac": 0.7,
        "hac_max_lag": 5,
        "fdr_alpha": 0.05,
        "fdr_method": "fdr_by",
        "structured_hidden_dim": 64,
        "structured_num_layers": 2,
        "structured_l1_lambda": 0.001,
        "structured_init_scale": 0.1
    }
    
    if BEST_HYPERPARAMS_FILE.exists():
        print(f"  Loading optimized hyperparameters from: {BEST_HYPERPARAMS_FILE}")
        with open(BEST_HYPERPARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        
        # Merge optimized params into defaults (use first stimulus's params as base)
        # Individual stimulus params can be accessed via best_params[stimulus]
        if best_params:
            first_stim = list(best_params.keys())[0]
            opt = best_params[first_stim]
            
            # Update only the optimized params, keep defaults for others
            if "dsm_lr" in opt:
                default_params["dsm_lr"] = opt["dsm_lr"]
            if "dsm_epochs" in opt:
                default_params["dsm_epochs"] = opt["dsm_epochs"]
            if "dsm_noise_std" in opt:
                default_params["dsm_noise_std"] = opt["dsm_noise_std"]
            if "structured_hidden_dim" in opt:
                default_params["structured_hidden_dim"] = opt["structured_hidden_dim"]
            if "structured_l1_lambda" in opt:
                default_params["structured_l1_lambda"] = opt["structured_l1_lambda"]
            if "fdr_alpha" in opt:
                default_params["fdr_alpha"] = opt["fdr_alpha"]
            
            print(f"    ✓ Using optimized params: lr={default_params['dsm_lr']}, "
                  f"epochs={default_params['dsm_epochs']}, "
                  f"hidden={default_params['structured_hidden_dim']}")
    else:
        print(f"  No optimized hyperparameters found. Using defaults.")
        print(f"  (Run script 02_hyperparameter_search.py first for optimization)")
    
    return default_params


# SBTG Hyperparameters - will be loaded at runtime
HYPERPARAMS = {
    "window_length": 2,  # Required for structured SBTG
    "dsm_hidden_dim": 128,
    "dsm_num_layers": 3,
    "dsm_noise_std": 0.1,
    "dsm_epochs": 200,  # Increased for thorough training
    "dsm_batch_size": 128,
    "dsm_lr": 1e-3,
    "train_frac": 0.7,
    "hac_max_lag": 5,
    "fdr_alpha": 0.05,  # More stringent than default 0.1
    "fdr_method": "fdr_by",  # Benjamini-Yekutieli (conservative)
    "structured_hidden_dim": 64,
    "structured_num_layers": 2,
    "structured_l1_lambda": 0.001,
    "structured_init_scale": 0.1
}

# Neuron set - use neurons actually present in the NeuroPAL recording
# These will be determined dynamically from the data
NEURON_SET = None  # Will be populated from data

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DatasetStatistics:
    """Statistics about the neural recording dataset."""
    stimulus_name: str
    n_neurons: int
    n_timepoints: int
    n_worms: int
    timepoint_mean: float
    timepoint_std: float
    min_value: float
    max_value: float
    example_neurons: List[str]
    recording_fps: float

@dataclass
class ModelResult:
    """Results from a single SBTG model training."""
    stimulus: str
    model_type: str
    train_split: str
    n_sign_edges: int
    n_volatility_edges: int
    hyperparameters: Dict
    timestamp: str

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize neuron names to uppercase."""
    return name.strip().upper()

def load_neuropal_data(data_dir: Path) -> Dict:
    """
    Load preprocessed NeuroPAL recording data.
    
    Returns:
        Dict with keys: neuron_names, norm_traces, fps, stim_names, 
                       stim_times, stims_per_worm, worm_ids
    """
    neuropal_path = data_dir / "Head_Activity_OH16230.mat"
    if not neuropal_path.exists():
        raise FileNotFoundError(f"NeuroPAL file not found: {neuropal_path}")
    
    mat = loadmat(neuropal_path, simplify_cells=True)
    
    return {
        "neuron_names": [normalize_name(str(n)) for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "stim_names": [str(s) for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "stims_per_worm": [np.asarray(row, dtype=int) for row in mat["stims"]],
        "worm_ids": [str(f) for f in mat["files"]]
    }

def collect_worm_trace(neuron_traces: List, worm_idx: int, num_worms: int) -> Optional[np.ndarray]:
    """
    Collect and average left/right traces for a single worm.
    
    Args:
        neuron_traces: List of traces for a neuron across all worms
        worm_idx: Index of target worm
        num_worms: Total number of worms
    
    Returns:
        Averaged trace or None if no data available
    """
    segments = []
    
    # Check both left and right (bilateral neurons stored as worm_idx and worm_idx + num_worms)
    for offset in [worm_idx, worm_idx + num_worms]:
        if offset >= len(neuron_traces):
            continue
        arr = np.asarray(neuron_traces[offset], dtype=float)
        if arr.size == 0:
            continue
        segments.append(arr)
    
    if not segments:
        return None
    
    # Align to shortest segment
    min_len = min(seg.shape[-1] for seg in segments)
    stacked = np.stack([seg[-min_len:] for seg in segments], axis=0)
    return stacked.mean(axis=0)

def build_stimulus_timeseries(
    stimulus_name: str,
    neuropal_data: Dict,
    neuron_subset: Optional[List[str]] = None
) -> Tuple[np.ndarray, DatasetStatistics]:
    """
    Build time series matrix for a specific stimulus.
    
    Args:
        stimulus_name: Name of stimulus (e.g., "butanone")
        neuropal_data: Data dictionary from load_neuropal_data()
        neuron_subset: List of neurons to include (if None, use all available)
    
    Returns:
        X: (T, n) time series matrix
        stats: DatasetStatistics object with summary information
    """
    # Find stimulus index
    try:
        stim_idx = neuropal_data["stim_names"].index(stimulus_name)
    except ValueError:
        raise ValueError(f"Stimulus '{stimulus_name}' not found in {neuropal_data['stim_names']}")
    
    # Use all neurons if subset not specified
    if neuron_subset is None:
        neuron_subset = neuropal_data["neuron_names"]
    
    # Map neuron names to indices
    name_to_idx = {name: i for i, name in enumerate(neuropal_data["neuron_names"])}
    node_indices = []
    for name in neuron_subset:
        if name not in name_to_idx:
            continue  # Skip neurons not in recording
        node_indices.append(name_to_idx[name])
    
    if len(node_indices) == 0:
        raise ValueError(f"No matching neurons found for {stimulus_name}")
    
    num_worms = len(neuropal_data["worm_ids"])
    
    # Collect traces for each neuron across all worms
    all_segments = []
    
    for node_idx in node_indices:
        neuron_traces = neuropal_data["norm_traces"][node_idx]
        worm_segments = []
        
        for worm_idx in range(num_worms):
            # Check if this worm has the target stimulus
            worm_stims = neuropal_data["stims_per_worm"][worm_idx]
            if stim_idx not in worm_stims:
                continue
            
            # Get trace for this worm
            trace = collect_worm_trace(neuron_traces, worm_idx, num_worms)
            if trace is None:
                continue
            
            # Extract stimulus segment
            stim_local_idx = np.where(worm_stims == stim_idx)[0][0]
            start_time = int(neuropal_data["stim_times"][stim_local_idx, 0])
            end_time = int(neuropal_data["stim_times"][stim_local_idx, 1])
            
            if end_time > len(trace):
                end_time = len(trace)
            
            segment = trace[start_time:end_time]
            worm_segments.append(segment)
        
        # Concatenate all worm segments for this neuron
        if worm_segments:
            node_timeseries = np.concatenate(worm_segments)
        else:
            # Use zeros if no data available
            node_timeseries = np.zeros(1)
        
        all_segments.append(node_timeseries)
    
    # Find minimum length and align
    if len(all_segments) == 0:
        raise ValueError(f"No valid neuron data for {stimulus_name}")
        
    min_length = min(seg.shape[0] for seg in all_segments)
    X = np.column_stack([seg[:min_length] for seg in all_segments])
    
    # Compute statistics
    actual_neurons = [neuropal_data["neuron_names"][idx] for idx in node_indices]
    stats = DatasetStatistics(
        stimulus_name=stimulus_name,
        n_neurons=X.shape[1],
        n_timepoints=X.shape[0],
        n_worms=num_worms,
        timepoint_mean=float(np.mean(X)),
        timepoint_std=float(np.std(X)),
        min_value=float(np.min(X)),
        max_value=float(np.max(X)),
        example_neurons=actual_neurons[:5],
        recording_fps=neuropal_data["fps"]
    )
    
    return X, stats, actual_neurons

# ============================================================================
# SBTG TRAINING
# ============================================================================

def train_sbtg_model(
    X: np.ndarray,
    model_config: Dict,
    train_split: str,
    hyperparams: Dict,
    verbose: bool = False
) -> Tuple[object, Dict]:
    """
    Train a single SBTG model configuration.
    
    Args:
        X: (T, n) time series matrix
        model_config: Model architecture configuration
        train_split: Train/test split strategy
        hyperparams: SBTG hyperparameters
        verbose: Print training progress
    
    Returns:
        result: SBTGVolatilityResult object
        metadata: Training metadata dict
    """
    # Merge model config with hyperparameters
    full_config = {**hyperparams, **model_config, "train_split": train_split}
    
    # Remove description key
    full_config.pop("description", None)
    
    # Initialize estimator
    estimator = SBTGStructuredVolatilityEstimator(
        **full_config,
        verbose=verbose
    )
    
    # Fit model
    result = estimator.fit(X)
    
    # Collect metadata
    metadata = {
        "model_type": model_config["model_type"],
        "train_split": train_split,
        "n_timepoints": X.shape[0],
        "n_neurons": X.shape[1],
        "n_sign_edges": int(np.sum(result.sign_adj != 0)),
        "n_volatility_edges": int(np.sum(result.volatility_adj != 0)),
        "n_total_edges": int(np.sum((result.sign_adj != 0) | (result.volatility_adj != 0))),
        "hyperparameters": hyperparams.copy(),
        "timestamp": datetime.now().isoformat()
    }
    
    return result, metadata

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_dataset_summary(stats_dict: Dict[str, DatasetStatistics], output_path: Path):
    """Create summary visualization of dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NeuroPAL Dataset Summary", fontsize=16, fontweight='bold')
    
    stimuli = list(stats_dict.keys())
    n_timepoints = [stats_dict[s].n_timepoints for s in stimuli]
    n_worms = [stats_dict[s].n_worms for s in stimuli]
    means = [stats_dict[s].timepoint_mean for s in stimuli]
    stds = [stats_dict[s].timepoint_std for s in stimuli]
    
    # Timepoints per stimulus
    axes[0, 0].bar(stimuli, n_timepoints, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_title("Timepoints per Stimulus")
    axes[0, 0].set_ylabel("Number of Timepoints")
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Worms per stimulus
    axes[0, 1].bar(stimuli, n_worms, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_title("Worms per Stimulus")
    axes[0, 1].set_ylabel("Number of Worms")
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Mean activity
    axes[1, 0].bar(stimuli, means, yerr=stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    axes[1, 0].set_title("Mean Activity ± Std Dev")
    axes[1, 0].set_ylabel("Normalized Fluorescence")
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Dataset info table
    axes[1, 1].axis('off')
    table_data = []
    for stim in stimuli:
        s = stats_dict[stim]
        table_data.append([
            stim.capitalize(),
            f"{s.n_neurons}",
            f"{s.n_timepoints}",
            f"{s.recording_fps:.1f} Hz"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Stimulus", "Neurons", "Timepoints", "FPS"],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title("Dataset Overview")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved dataset summary: {output_path}")

def plot_edge_comparison(results_df: pd.DataFrame, output_path: Path):
    """Compare edge counts across model types and splits."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Edge Detection Across Configurations", fontsize=16, fontweight='bold')
    
    # Group by model type
    model_comparison = results_df.groupby(['stimulus', 'model_type'])[['n_sign_edges', 'n_volatility_edges']].mean()
    
    # Plot 1: Sign edges by model type
    ax = axes[0, 0]
    model_comparison['n_sign_edges'].unstack().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title("Sign-Based Edges by Model Type")
    ax.set_ylabel("Number of Edges")
    ax.set_xlabel("Stimulus")
    ax.legend(title="Model Type", loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Volatility edges by model type
    ax = axes[0, 1]
    model_comparison['n_volatility_edges'].unstack().plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title("Volatility Edges by Model Type")
    ax.set_ylabel("Number of Edges")
    ax.set_xlabel("Stimulus")
    ax.legend(title="Model Type", loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Edges by train split
    split_comparison = results_df.groupby(['stimulus', 'train_split'])['n_sign_edges'].mean()
    ax = axes[1, 0]
    split_comparison.unstack().plot(kind='bar', ax=ax)
    ax.set_title("Sign Edges by Train/Test Split")
    ax.set_ylabel("Number of Edges")
    ax.set_xlabel("Stimulus")
    ax.legend(title="Split Strategy", loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Total edges heatmap
    ax = axes[1, 1]
    pivot_data = results_df.pivot_table(
        values='n_total_edges',
        index='model_type',
        columns='train_split',
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Total Edges'})
    ax.set_title("Average Total Edges (All Stimuli)")
    ax.set_xlabel("Train Split")
    ax.set_ylabel("Model Type")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved edge comparison: {output_path}")

# ============================================================================
# REPORTING
# ============================================================================

def generate_markdown_report(
    stats_dict: Dict[str, DatasetStatistics],
    results_df: pd.DataFrame,
    hyperparams: Dict,
    output_path: Path
):
    """Generate comprehensive markdown report."""
    
    report = []
    report.append("# Comprehensive SBTG Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # Dataset Summary
    report.append("## Dataset Summary\n")
    report.append("### NeuroPAL Recording Statistics\n")
    report.append("| Stimulus | Neurons | Timepoints | Worms | Mean ± Std | Range | FPS |")
    report.append("|----------|---------|------------|-------|------------|-------|-----|")
    
    for stim_name, stats in stats_dict.items():
        report.append(
            f"| {stim_name.capitalize()} | {stats.n_neurons} | {stats.n_timepoints} | "
            f"{stats.n_worms} | {stats.timepoint_mean:.3f} ± {stats.timepoint_std:.3f} | "
            f"[{stats.min_value:.3f}, {stats.max_value:.3f}] | {stats.recording_fps:.1f} |"
        )
    
    report.append(f"\n**Example Neurons:** {', '.join(stats_dict[list(stats_dict.keys())[0]].example_neurons)}\n")
    report.append(f"**Total Neurons Available:** {stats_dict[list(stats_dict.keys())[0]].n_neurons} neurons in NeuroPAL recording\n")
    
    # Hyperparameters
    report.append("\n## Hyperparameter Configuration\n")
    report.append("### SBTG Model Parameters\n")
    report.append("| Parameter | Value | Description |")
    report.append("|-----------|-------|-------------|")
    
    param_descriptions = {
        "window_length": "Window length for score model",
        "dsm_hidden_dim": "Hidden dimension for denoising score model",
        "dsm_num_layers": "Number of layers in score network",
        "dsm_noise_std": "Noise standard deviation for corruption",
        "dsm_epochs": "Training epochs for score model",
        "dsm_batch_size": "Batch size for training",
        "dsm_lr": "Learning rate",
        "train_frac": "Fraction of data used for training",
        "hac_max_lag": "Maximum lag for HAC variance estimation",
        "fdr_alpha": "FDR control level",
        "fdr_method": "FDR method (Benjamini-Yekutieli)",
        "structured_l1_lambda": "L1 regularization on coupling matrix"
    }
    
    for param, value in hyperparams.items():
        desc = param_descriptions.get(param, "")
        report.append(f"| `{param}` | {value} | {desc} |")
    
    # Model Configurations
    report.append("\n### Model Architectures Tested\n")
    for model_name, config in MODEL_CONFIGS.items():
        report.append(f"\n**{model_name.upper()}:** {config['description']}")
        extra_params = {k: v for k, v in config.items() if k not in ['model_type', 'description']}
        if extra_params:
            report.append(f"  - Additional parameters: {extra_params}")
    
    # Train/Test Splits
    report.append("\n### Train/Test Split Strategies\n")
    for split_name, desc in SPLIT_STRATEGIES.items():
        report.append(f"- **{split_name}:** {desc}")
    
    # Results Summary
    report.append("\n## Results Summary\n")
    report.append("### Edge Detection Performance\n")
    
    # Best configurations per stimulus
    report.append("#### Best Configurations by Stimulus\n")
    for stim in STIMULI:
        stim_df = results_df[results_df['stimulus'] == stim]
        if len(stim_df) > 0 and 'n_total_edges' in stim_df.columns:
            best_row = stim_df.loc[stim_df['n_total_edges'].idxmax()]
            report.append(f"\n**{stim.capitalize()}:**")
            report.append(f"- Model: {best_row['model_type']}")
            report.append(f"- Split: {best_row['train_split']}")
            report.append(f"- Sign edges: {best_row['n_sign_edges']}")
            report.append(f"- Volatility edges: {best_row['n_volatility_edges']}")
            report.append(f"- Total edges: {best_row['n_total_edges']}")
        else:
            report.append(f"\n**{stim.capitalize()}:** No successful configurations")
    
    # Overall statistics
    report.append("\n### Overall Statistics\n")
    report.append(f"- **Total configurations tested:** {len(results_df)}")
    if len(results_df) > 0 and 'n_sign_edges' in results_df.columns:
        report.append(f"- **Average sign edges:** {results_df['n_sign_edges'].mean():.1f} ± {results_df['n_sign_edges'].std():.1f}")
        report.append(f"- **Average volatility edges:** {results_df['n_volatility_edges'].mean():.1f} ± {results_df['n_volatility_edges'].std():.1f}")
    else:
        report.append("- **No edge statistics available**")
    
    # Recommendations
    report.append("\n## Recommendations and Next Steps\n")
    report.append("### Optimal Hyperparameters by Dataset\n")
    
    for stim in STIMULI:
        stim_df = results_df[results_df['stimulus'] == stim]
        if len(stim_df) > 0 and 'n_total_edges' in stim_df.columns:
            stim_df_valid = stim_df.dropna(subset=['n_total_edges'])
            if len(stim_df_valid) > 0:
                best_model = stim_df_valid.groupby('model_type')['n_total_edges'].mean().idxmax()
                best_split = stim_df_valid.groupby('train_split')['n_total_edges'].mean().idxmax()
                
                report.append(f"\n**{stim.capitalize()}:**")
                report.append(f"- Recommended model: **{best_model}**")
                report.append(f"- Recommended split: **{best_split}**")
            else:
                report.append(f"\n**{stim.capitalize()}:** No successful configurations")
        else:
            report.append(f"\n**{stim.capitalize()}:** No data available")
    
    report.append("\n### Parameters for Further Exploration\n")
    report.append("1. **FDR Alpha:** Current value 0.05 is conservative. Try [0.01, 0.05, 0.1] for sensitivity analysis")
    report.append("2. **DSM Epochs:** Current 200. Could increase to 300-500 for more complex models")
    report.append("3. **L1 Lambda:** Current 0.001. Explore [0, 0.0001, 0.001, 0.01] for sparsity tuning")
    report.append("4. **Train Fraction:** Current 0.7. Try [0.6, 0.7, 0.8] for stability assessment")
    report.append("5. **Feature Dimensions:** For feature_bilinear, explore feature_dim [8, 16, 32]")
    report.append("6. **Number of Regimes:** For regime_gated, test [2, 3, 4] regimes")
    
    # Limitations
    report.append("\n### Known Limitations\n")
    report.append("1. **Data Availability:** Analysis limited to neurons present in NeuroPAL recordings")
    report.append("2. **Temporal Resolution:** Fixed window_length=2 required by structured SBTG")
    report.append("3. **Statistical Power:** Limited by number of worms and stimulus presentations")
    report.append("4. **Causality:** Detected edges represent statistical associations, not proven causal links")
    report.append("5. **Hyperparameter Tuning:** Current parameters based on preliminary exploration, not exhaustive grid search")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"  Saved comprehensive report: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run comprehensive SBTG analysis pipeline."""
    global HYPERPARAMS
    
    print("=" * 80)
    print("SCRIPT 03: COMPREHENSIVE SBTG ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # ========================================================================
    # STEP 0: Load optimized hyperparameters from script 02
    # ========================================================================
    print("STEP 0: Loading hyperparameters...")
    print("-" * 80)
    HYPERPARAMS = load_optimized_hyperparams()
    print()
    
    # ========================================================================
    # STEP 1: Load and summarize data
    # ========================================================================
    print("STEP 1: Loading and analyzing dataset...")
    print("-" * 80)
    
    neuropal_data = load_neuropal_data(DATA_DIR)
    print(f"✓ Loaded NeuroPAL data")
    print(f"  Total neurons in recording: {len(neuropal_data['neuron_names'])}")
    print(f"  Total worms: {len(neuropal_data['worm_ids'])}")
    print(f"  Stimuli: {', '.join(neuropal_data['stim_names'])}")
    print(f"  Recording FPS: {neuropal_data['fps']:.1f} Hz")
    print()
    
    # Build timeseries for each stimulus
    timeseries_data = {}
    stats_dict = {}
    neuron_names_dict = {}
    
    for stimulus in STIMULI:
        print(f"Building timeseries for {stimulus}...")
        X, stats, neurons = build_stimulus_timeseries(stimulus, neuropal_data, neuron_subset=None)  # Use all available neurons
        timeseries_data[stimulus] = X
        stats_dict[stimulus] = stats
        neuron_names_dict[stimulus] = neurons
        
        print(f"  ✓ {stimulus.capitalize()}:")
        print(f"      Shape: ({stats.n_timepoints} timepoints, {stats.n_neurons} neurons)")
        print(f"      Mean ± Std: {stats.timepoint_mean:.3f} ± {stats.timepoint_std:.3f}")
        print(f"      Range: [{stats.min_value:.3f}, {stats.max_value:.3f}]")
        print(f"      Example neurons: {', '.join(stats.example_neurons)}")
        print()
    
    # Create dataset summary visualization
    plot_dataset_summary(stats_dict, FIGURES_DIR / "dataset_summary.png")
    
    # ========================================================================
    # STEP 2: Train models across all configurations
    # ========================================================================
    print("\nSTEP 2: Training SBTG models across all configurations...")
    print("-" * 80)
    print(f"Model types: {list(MODEL_CONFIGS.keys())}")
    print(f"Train splits: {list(SPLIT_STRATEGIES.keys())}")
    print(f"Total configurations per stimulus: {len(MODEL_CONFIGS) * len(SPLIT_STRATEGIES)}")
    print()
    
    all_results = []
    results_data = {}
    
    for stimulus in STIMULI:
        print(f"\n{'='*80}")
        print(f"TRAINING: {stimulus.upper()}")
        print('='*80)
        
        X = timeseries_data[stimulus]
        stimulus_results = {}
        neuron_list = neuron_names_dict[stimulus]  # Get neuron names for this stimulus
        
        for model_name, model_config in MODEL_CONFIGS.items():
            for split_name in SPLIT_STRATEGIES.keys():
                config_id = f"{model_name}_{split_name}"
                print(f"\n  [{config_id}]")
                
                try:
                    result, metadata = train_sbtg_model(
                        X, model_config, split_name, HYPERPARAMS, verbose=False
                    )
                    
                    metadata['stimulus'] = stimulus
                    all_results.append(metadata)
                    stimulus_results[config_id] = result
                    
                    print(f"    ✓ Success!")
                    print(f"      Sign edges: {metadata['n_sign_edges']}")
                    print(f"      Volatility edges: {metadata['n_volatility_edges']}")
                    print(f"      Total edges: {metadata['n_total_edges']}")
                    
                    # Save individual result
                    result_dir = OUTPUT_DIR / "models" / stimulus / config_id
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    np.savez(
                        result_dir / "result.npz",
                        sign_adj=result.sign_adj,
                        volatility_adj=result.volatility_adj,
                        p_mean=result.p_mean,
                        p_volatility=result.p_volatility,
                        mu_hat=result.mu_hat  # Add coupling weights
                    )
                    
                    with open(result_dir / "metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Save neuron names for alignment with connectome
                    with open(result_dir / "neuron_names.json", 'w') as f:
                        json.dump(neuron_list, f, indent=2)
                    
                except Exception as e:
                    print(f"    ✗ ERROR: {str(e)[:100]}")
                    all_results.append({
                        'stimulus': stimulus,
                        'model_type': model_name,
                        'train_split': split_name,
                        'error': str(e)
                    })
        
        results_data[stimulus] = stimulus_results
    
    # ========================================================================
    # STEP 3: Generate visualizations and reports
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Generating visualizations and reports...")
    print('='*80)
    
    # Create results dataframe
    results_df = pd.DataFrame([r for r in all_results if 'error' not in r])
    
    # Save results table
    results_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"✓ Saved results table: all_results.csv")
    
    # Generate visualizations
    plot_edge_comparison(results_df, FIGURES_DIR / "edge_comparison.png")
    
    # Generate comprehensive report
    generate_markdown_report(
        stats_dict, results_df, HYPERPARAMS,
        REPORTS_DIR / "comprehensive_analysis.md"
    )
    
    # ========================================================================
    # STEP 4: Print summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print('='*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Models: {OUTPUT_DIR / 'models'}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print()
    
    # Print summary table
    print("Configuration Performance Summary:")
    print("-" * 100)
    print(f"{'Model Type':<20} {'Train Split':<25} {'Butanone':<15} {'NaCl':<15} {'Pentanedione':<15}")
    print("-" * 100)
    
    for model_name in MODEL_CONFIGS.keys():
        for split_name in SPLIT_STRATEGIES.keys():
            row = f"{model_name:<20} {split_name:<25}"
            
            for stim in STIMULI:
                stim_df = results_df[
                    (results_df['stimulus'] == stim) &
                    (results_df['model_type'] == model_name) &
                    (results_df['train_split'] == split_name)
                ]
                if len(stim_df) > 0:
                    edge_str = f"{int(stim_df.iloc[0]['n_total_edges'])}"
                    row += f"{edge_str:<15}"
                else:
                    row += f"{'ERROR':<15}"
            
            print(row)
    
    print("-" * 100)
    print(f"\nSuccess rate: {len(results_df)}/{len(all_results)} ({100*len(results_df)/len(all_results):.1f}%)")
    print()

if __name__ == "__main__":
    main()
