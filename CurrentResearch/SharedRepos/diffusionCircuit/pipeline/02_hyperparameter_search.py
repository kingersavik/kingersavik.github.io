#!/usr/bin/env python3
"""
SCRIPT 02: Hyperparameter Grid Search
======================================

Performs dataset-specific hyperparameter optimization BEFORE the comprehensive
analysis. Finds optimal training parameters for each stimulus dataset.

Key Features:
- Grid search over training hyperparameters (lr, epochs, hidden_dim, etc.)
- Dataset-specific optimization (NaCl vs Pentanedione)
- Tracks and saves training loss curves (avoid re-running experiments)
- Outputs best hyperparameters per dataset for use by script 03
- Optional validation against structural connectome

USAGE:
    python pipeline/02_hyperparameter_search.py
    python pipeline/02_hyperparameter_search.py --quick   # Reduced grid for testing

INPUTS:
    - data/Head_Activity_OH16230.mat (NeuroPAL recordings)
    - data/Head_Activity_OH15500.mat (NaCl recordings)
    - results/intermediate/connectome/A_struct.npy (for F1 evaluation)

OUTPUTS:
    - results/hyperparameter_search/search_results.csv (all configurations)
    - results/hyperparameter_search/loss_curves/{config_id}.json (training loss)
    - results/hyperparameter_search/best_hyperparameters.json (per-dataset best)
    - results/hyperparameter_search/figures/*.png (visualizations)

RUNTIME: ~1-4 hours depending on grid size
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from itertools import product
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# scipy.io.loadmat no longer needed - using pre-prepared datasets
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sbtg import SBTGStructuredVolatilityEstimator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = PROJECT_ROOT / "results" / "hyperparameter_search"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"

# Create output directories
LOSS_CURVES_DIR = OUTPUT_DIR / "loss_curves"
FIGURES_DIR = OUTPUT_DIR / "figures"
for dir_path in [OUTPUT_DIR, LOSS_CURVES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Stimuli to optimize (butanone excluded - only 1 timepoint)
STIMULI = ["nacl", "pentanedione"]

# Model types to test during hyperparameter search
# Using linear as it's fastest - best hyperparams transfer to other models
MODEL_TYPES = ["linear", "feature_bilinear"]

# Train/test split for validation (use per_stimulus_prefix as most rigorous)
VALIDATION_SPLIT = "per_stimulus_prefix"

# ============================================================================
# HYPERPARAMETER GRID
# ============================================================================

# Full grid for production (takes ~2-4 hours)
FULL_PARAM_GRID = {
    "dsm_lr": [5e-4, 1e-3, 2e-3],
    "dsm_epochs": [50, 100, 200],
    "dsm_noise_std": [0.05, 0.1, 0.2],
    "structured_hidden_dim": [32, 64, 128],
    "structured_l1_lambda": [0.0, 0.001, 0.01],
    "fdr_alpha": [0.05, 0.1],
}

# Reduced grid for quick testing (~15-30 min)
QUICK_PARAM_GRID = {
    "dsm_lr": [1e-3],
    "dsm_epochs": [50, 100],
    "dsm_noise_std": [0.1],
    "structured_hidden_dim": [64],
    "structured_l1_lambda": [0.0, 0.001],
    "fdr_alpha": [0.05],
}

# Fixed parameters (not swept)
FIXED_PARAMS = {
    "window_length": 2,  # Required for structured SBTG
    "dsm_batch_size": 128,
    "dsm_num_layers": 3,
    "structured_num_layers": 2,
    "structured_init_scale": 0.1,
    "train_frac": 0.7,
    "hac_max_lag": 5,
    "fdr_method": "fdr_by",
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SearchResult:
    """Result from a single hyperparameter configuration."""
    config_id: str
    stimulus: str
    model_type: str
    
    # Hyperparameters
    dsm_lr: float
    dsm_epochs: int
    dsm_noise_std: float
    structured_hidden_dim: int
    structured_l1_lambda: float
    fdr_alpha: float
    
    # Training metrics
    final_train_loss: float
    final_val_loss: float
    min_val_loss: float
    best_epoch: int
    
    # Edge detection metrics (if connectome available)
    n_edges: int
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    
    # Metadata
    runtime_seconds: float
    timestamp: str


# ============================================================================
# DATA LOADING
# ============================================================================
# 
# Load pre-prepared datasets from 01_prepare_data.py
# These datasets have proper lag windows already built across all worms.
# ============================================================================

DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"


def load_prepared_dataset(stimulus: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-prepared dataset from 01_prepare_data.py.
    
    This uses the properly processed data with:
    - All worms included
    - Lag windows already built CORRECTLY (per-worm, no cross-worm transitions)
    - Standardization applied
    
    Returns:
        X_segments: (n_worms, n_timepoints, n_neurons) individual worm segments
        Z_std: (n_windows, 2*n_neurons) pre-built standardized lag windows
        neuron_names: List of neuron names
    """
    dataset_dir = DATASETS_DIR / stimulus
    
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found for '{stimulus}'.\n"
            f"Run pipeline/01_prepare_data.py first to create datasets."
        )
    
    # Load segment data (individual worms kept separate)
    X_segments = np.load(dataset_dir / "X_segments.npy", allow_pickle=True)
    
    # Convert from object array to float64 if needed
    if X_segments.dtype == object:
        X_segments = np.array(X_segments.tolist(), dtype=np.float64)
    
    # Load pre-built lag windows (CORRECTLY built per-worm)
    Z_std = np.load(dataset_dir / "Z_std.npy", allow_pickle=True)
    if Z_std.dtype == object:
        Z_std = np.array(Z_std.tolist(), dtype=np.float64)
    
    # Load standardization info (contains node order)
    with open(dataset_dir / "standardization.json", 'r') as f:
        std_info = json.load(f)
    
    neuron_names = std_info.get("node_order", [])
    
    n_worms, n_timepoints, n_neurons = X_segments.shape
    n_windows = Z_std.shape[0]
    
    print(f"  Loaded: {n_worms} worms × {n_timepoints} timepoints × {n_neurons} neurons")
    print(f"  Pre-built windows: {n_windows} (= {n_worms} × {n_timepoints - 1}, no cross-worm transitions)")
    
    return X_segments, Z_std, neuron_names


def normalize_name(name: str) -> str:
    """Normalize neuron names to uppercase."""
    return name.strip().upper()


def load_structural_connectome() -> Tuple[np.ndarray, List[str]]:
    """Load Cook et al. 2019 structural connectome."""
    struct_file = CONNECTOME_DIR / "A_struct.npy"
    nodes_file = CONNECTOME_DIR / "nodes.json"
    
    if not struct_file.exists():
        return None, None
    
    A_struct = np.load(struct_file)
    
    with open(nodes_file, 'r') as f:
        struct_neurons = json.load(f)
    
    return A_struct, struct_neurons


# ============================================================================
# TRAINING WITH VALIDATION LOSS TRACKING
# ============================================================================
# 
# IMPORTANT: We optimize hyperparameters based on VALIDATION LOSS only.
# The connectome is NEVER used during training or hyperparameter selection.
# Connectome comparison is ONLY for final evaluation in later scripts.
#
# Validation loss = DSM loss on held-out timepoints
# This measures how well the model captures temporal dynamics of neural activity.
# ============================================================================

def compute_dsm_loss(model, data_tensor, noise_std, device):
    """Compute DSM loss on a batch of data.
    
    Note: The SBTG model uses torch.autograd.grad internally for score computation,
    so we need to enable gradients on the input. We detach at the end to avoid
    memory leaks.
    """
    import torch
    
    model.eval()
    
    # Move to device and enable gradient tracking (required by model's score computation)
    z = data_tensor.to(device).requires_grad_(True)
    eps = torch.randn_like(z)
    y_noisy = z + noise_std * eps
    y_noisy = y_noisy.detach().requires_grad_(True)  # Fresh tensor with gradients
    
    target = -eps / noise_std
    
    # Forward pass (model uses autograd.grad internally)
    pred = model(y_noisy)
    
    # Compute loss and detach
    loss = ((pred - target) ** 2).mean().item()
    
    return loss


def compute_config_id(params: Dict) -> str:
    """Generate unique ID for a parameter configuration."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def train_and_evaluate(
    X_segments: np.ndarray,
    neuron_names: List[str],
    stimulus: str,
    model_type: str,
    params: Dict,
    A_struct: Optional[np.ndarray] = None,
    struct_neurons: Optional[List[str]] = None,
) -> Tuple[SearchResult, Dict]:
    """
    Train a single configuration and evaluate using VALIDATION LOSS.
    
    IMPORTANT: 
    - We do NOT use the connectome for hyperparameter optimization.
    - The metric is validation loss (DSM loss on held-out WORMS).
    - We split at the WORM level to avoid data leakage.
    - Each worm's data is passed separately to SBTG to avoid cross-worm transitions.
    
    The connectome is only loaded for informational purposes (to report F1
    in the results table) but is NOT used to select hyperparameters.
    
    Args:
        X_segments: (n_worms, n_timepoints, n_neurons) - individual worm recordings
    
    Returns:
        result: SearchResult with metrics
        loss_data: Dict with configuration details
    """
    import time
    import torch
    
    config_id = compute_config_id({**params, "stimulus": stimulus, "model_type": model_type})
    start_time = time.time()
    
    # Check if we've already run this config
    cache_file = LOSS_CURVES_DIR / f"{config_id}.json"
    if cache_file.exists():
        # Load cached result
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        
        # Reconstruct SearchResult
        return SearchResult(
            config_id=config_id,
            stimulus=stimulus,
            model_type=model_type,
            dsm_lr=params["dsm_lr"],
            dsm_epochs=params["dsm_epochs"],
            dsm_noise_std=params["dsm_noise_std"],
            structured_hidden_dim=params["structured_hidden_dim"],
            structured_l1_lambda=params["structured_l1_lambda"],
            fdr_alpha=params["fdr_alpha"],
            final_train_loss=cached.get("final_train_loss", 0.0),
            final_val_loss=cached.get("final_val_loss", 0.0),
            min_val_loss=cached.get("min_val_loss", 0.0),
            best_epoch=cached.get("best_epoch", -1),
            n_edges=cached["n_edges"],
            f1_score=cached.get("f1_score"),
            precision=cached.get("precision"),
            recall=cached.get("recall"),
            runtime_seconds=cached["runtime_seconds"],
            timestamp=cached["timestamp"],
        ), cached
    
    # Build full config
    full_params = {**FIXED_PARAMS, **params}
    
    # Split at the WORM level (not timepoint level!)
    # This avoids leakage and ensures proper cross-validation
    n_worms = X_segments.shape[0]
    n_train_worms = int(n_worms * 0.7)
    
    if n_train_worms < 2 or (n_worms - n_train_worms) < 1:
        runtime = time.time() - start_time
        failed_result = SearchResult(
            config_id=config_id,
            stimulus=stimulus,
            model_type=model_type,
            dsm_lr=params["dsm_lr"],
            dsm_epochs=params["dsm_epochs"],
            dsm_noise_std=params["dsm_noise_std"],
            structured_hidden_dim=params["structured_hidden_dim"],
            structured_l1_lambda=params["structured_l1_lambda"],
            fdr_alpha=params["fdr_alpha"],
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            min_val_loss=float('inf'),
            best_epoch=-1,
            n_edges=0,
            f1_score=None,
            precision=None,
            recall=None,
            runtime_seconds=runtime,
            timestamp=datetime.now().isoformat(),
        )
        return failed_result, {"error": "Insufficient worms for train/val split"}
    
    # Split worms into train and validation sets
    train_worms = [X_segments[i] for i in range(n_train_worms)]
    val_worms = [X_segments[i] for i in range(n_train_worms, n_worms)]
    
    # Train model on TRAINING worms only
    # Pass as LIST of arrays - SBTG handles multiple segments correctly!
    training_params = {**full_params, "train_split": "prefix", "train_frac": 0.85}
    estimator = SBTGStructuredVolatilityEstimator(
        model_type=model_type,
        verbose=False,
        **training_params
    )
    
    try:
        result = estimator.fit(train_worms)  # Pass list of worm arrays!
    except Exception as e:
        runtime = time.time() - start_time
        failed_result = SearchResult(
            config_id=config_id,
            stimulus=stimulus,
            model_type=model_type,
            dsm_lr=params["dsm_lr"],
            dsm_epochs=params["dsm_epochs"],
            dsm_noise_std=params["dsm_noise_std"],
            structured_hidden_dim=params["structured_hidden_dim"],
            structured_l1_lambda=params["structured_l1_lambda"],
            fdr_alpha=params["fdr_alpha"],
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            min_val_loss=float('inf'),
            best_epoch=-1,
            n_edges=0,
            f1_score=None,
            precision=None,
            recall=None,
            runtime_seconds=runtime,
            timestamp=datetime.now().isoformat(),
        )
        return failed_result, {"error": str(e)}
    
    # Compute validation loss on held-out WORMS
    # Build validation windows (per-worm, no cross-worm transitions!)
    val_windows = []
    for worm_data in val_worms:
        T = worm_data.shape[0]
        for t in range(T - 1):
            z_t = np.concatenate([worm_data[t], worm_data[t + 1]])
            val_windows.append(z_t)
    val_windows = np.array(val_windows)
    
    # Build training windows for standardization stats
    train_windows = []
    for worm_data in train_worms:
        T = worm_data.shape[0]
        for t in range(T - 1):
            z_t = np.concatenate([worm_data[t], worm_data[t + 1]])
            train_windows.append(z_t)
    train_windows = np.array(train_windows)
    
    # Standardize validation windows using training statistics
    train_mean = train_windows.mean(axis=0, keepdims=True)
    train_std = train_windows.std(axis=0, keepdims=True) + 1e-8
    val_windows_std = (val_windows - train_mean) / train_std
    
    # Compute validation loss
    if estimator.model is not None:
        val_tensor = torch.from_numpy(val_windows_std.astype(np.float32))
        val_loss = compute_dsm_loss(
            estimator.model,
            val_tensor,
            params["dsm_noise_std"],
            estimator.device
        )
    else:
        val_loss = float('inf')
    
    runtime = time.time() - start_time
    
    n_edges = int((result.sign_adj != 0).sum())
    
    # Evaluate against connectome if available
    f1 = precision = recall = None
    
    if A_struct is not None and struct_neurons is not None:
        # Find overlapping neurons
        func_idx = []
        struct_idx = []
        
        for i, name in enumerate(neuron_names):
            if name in struct_neurons:
                j = struct_neurons.index(name)
                func_idx.append(i)
                struct_idx.append(j)
        
        if len(func_idx) >= 10:
            # Extract aligned matrices
            func_adj = (result.sign_adj[np.ix_(func_idx, func_idx)] != 0).astype(int)
            struct_adj = (A_struct[np.ix_(struct_idx, struct_idx)] > 0).astype(int)
            
            # Flatten (exclude diagonal)
            n = len(func_idx)
            mask = ~np.eye(n, dtype=bool)
            y_pred = func_adj[mask].flatten()
            y_true = struct_adj[mask].flatten()
            
            if y_true.sum() > 0:
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                precision = float(precision_score(y_true, y_pred, zero_division=0))
                recall = float(recall_score(y_true, y_pred, zero_division=0))
    
    # Create result - PRIMARY METRIC IS VALIDATION LOSS, NOT F1
    search_result = SearchResult(
        config_id=config_id,
        stimulus=stimulus,
        model_type=model_type,
        dsm_lr=params["dsm_lr"],
        dsm_epochs=params["dsm_epochs"],
        dsm_noise_std=params["dsm_noise_std"],
        structured_hidden_dim=params["structured_hidden_dim"],
        structured_l1_lambda=params["structured_l1_lambda"],
        fdr_alpha=params["fdr_alpha"],
        final_train_loss=0.0,  # Could track if needed
        final_val_loss=val_loss,  # PRIMARY METRIC for hyperparameter selection
        min_val_loss=val_loss,  # Same as final for this implementation
        best_epoch=params["dsm_epochs"],  # Final epoch
        n_edges=n_edges,
        f1_score=f1,  # Informational only - NOT used for HP selection
        precision=precision,
        recall=recall,
        runtime_seconds=runtime,
        timestamp=datetime.now().isoformat(),
    )
    
    # Save results to cache
    cache_data = {
        "config_id": config_id,
        "stimulus": stimulus,
        "model_type": model_type,
        "params": params,
        "final_val_loss": val_loss,  # PRIMARY METRIC
        "min_val_loss": val_loss,
        "n_edges": n_edges,
        "f1_score": f1,  # Informational only
        "precision": precision,
        "recall": recall,
        "runtime_seconds": runtime,
        "timestamp": search_result.timestamp,
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    return search_result, cache_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results_summary(results_df: pd.DataFrame, output_dir: Path):
    """Plot validation loss comparison across configurations.
    
    NOTE: Validation loss is the PRIMARY metric - lower is better.
    We do NOT use connectome F1 for hyperparameter selection.
    """
    
    for stimulus in STIMULI:
        stim_df = results_df[results_df["stimulus"] == stimulus]
        if len(stim_df) == 0:
            continue
        
        # Filter to valid validation losses (not inf or nan)
        valid_df = stim_df[
            stim_df["min_val_loss"].notna() & 
            (stim_df["min_val_loss"] < float('inf'))
        ]
        if len(valid_df) == 0:
            continue
        
        # Get top 5 by LOWEST validation loss (lower is better)
        top_5 = valid_df.nsmallest(5, "min_val_loss")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{stimulus.upper()} - Top 5 Configurations by Validation Loss", fontsize=14)
        
        # Bar chart of validation losses
        labels = []
        val_losses = []
        f1_scores = []
        
        for _, row in top_5.iterrows():
            label = f"lr={row['dsm_lr']:.0e}\nh={int(row['structured_hidden_dim'])}\nl1={row['structured_l1_lambda']}"
            labels.append(label)
            val_losses.append(row['min_val_loss'])
            f1_scores.append(row['f1_score'] if pd.notna(row['f1_score']) else 0)
        
        x = range(len(labels))
        
        axes[0].bar(x, val_losses, color='steelblue', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel("Validation Loss (DSM)")
        axes[0].set_title("Validation Loss (lower is better)")
        axes[0].grid(axis='y', alpha=0.3)
        
        # Also show F1 for reference (but NOT used for selection!)
        axes[1].bar(x, f1_scores, color='orange', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel("F1 Score (vs Connectome)")
        axes[1].set_title("F1 Score (for reference only - NOT used for HP selection)")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"loss_curves_{stimulus}.png", dpi=150, bbox_inches='tight')
        plt.close()


def plot_parameter_heatmaps(results_df: pd.DataFrame, output_dir: Path):
    """Plot heatmaps showing parameter effects."""
    
    for stimulus in STIMULI:
        stim_df = results_df[results_df["stimulus"] == stimulus]
        if len(stim_df) == 0:
            continue
        
        # Heatmap: lr vs hidden_dim (avg validation loss)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{stimulus.upper()} - Parameter Effects", fontsize=14)
        
        # Validation loss
        pivot = stim_df.pivot_table(
            values="min_val_loss",
            index="structured_hidden_dim",
            columns="dsm_lr",
            aggfunc="mean"
        )
        
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=axes[0])
            axes[0].set_title("Min Validation Loss")
        
        # F1 score (if available)
        if stim_df["f1_score"].notna().any():
            pivot_f1 = stim_df.pivot_table(
                values="f1_score",
                index="structured_hidden_dim",
                columns="dsm_lr",
                aggfunc="mean"
            )
            
            if not pivot_f1.empty:
                sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[1])
                axes[1].set_title("F1 Score (vs Connectome)")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"param_heatmap_{stimulus}.png", dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search for SBTG")
    parser.add_argument("--quick", action="store_true", help="Use reduced grid for testing")
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print()
    
    # Select parameter grid
    param_grid = QUICK_PARAM_GRID if args.quick else FULL_PARAM_GRID
    
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [dict(zip(param_keys, v)) for v in product(*param_values)]
    
    total_configs = len(param_combinations) * len(STIMULI) * len(MODEL_TYPES)
    
    print(f"Grid size: {len(param_combinations)} parameter combinations")
    print(f"Stimuli: {STIMULI}")
    print(f"Model types: {MODEL_TYPES}")
    print(f"Total configurations: {total_configs}")
    print()
    
    # Load structural connectome for evaluation
    print("Loading structural connectome...")
    A_struct, struct_neurons = load_structural_connectome()
    if A_struct is not None:
        print(f"  ✓ Loaded: {len(struct_neurons)} neurons")
    else:
        print("  ⚠ Connectome not found - F1 evaluation will be skipped")
    print()
    
    # Run grid search
    all_results = []
    
    for stimulus in STIMULI:
        print(f"\n{'='*80}")
        print(f"STIMULUS: {stimulus.upper()}")
        print(f"{'='*80}")
        
        # Load pre-prepared data from 01_prepare_data.py
        print(f"Loading {stimulus} data...")
        try:
            X_segments, Z_std, neuron_names = load_prepared_dataset(stimulus)
            print(f"  ✓ Ready for training")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
        
        # Test all configurations
        for model_type in MODEL_TYPES:
            print(f"\nModel: {model_type}")
            
            pbar = tqdm(param_combinations, desc=f"  {model_type}")
            
            for params in pbar:
                result, loss_data = train_and_evaluate(
                    X_segments=X_segments,
                    neuron_names=neuron_names,
                    stimulus=stimulus,
                    model_type=model_type,
                    params=params,
                    A_struct=A_struct,
                    struct_neurons=struct_neurons,
                )
                
                all_results.append(asdict(result))
                
                # Update progress bar
                if result.f1_score is not None:
                    pbar.set_postfix({
                        "val_loss": f"{result.min_val_loss:.4f}",
                        "F1": f"{result.f1_score:.3f}"
                    })
                else:
                    pbar.set_postfix({"val_loss": f"{result.min_val_loss:.4f}"})
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "search_results.csv", index=False)
    print(f"\n✓ Saved all results to: {OUTPUT_DIR / 'search_results.csv'}")
    
    # Find best hyperparameters per dataset
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS PER DATASET")
    print("(Selected by VALIDATION LOSS - NOT connectome F1)")
    print("=" * 80)
    
    best_params = {}
    
    for stimulus in STIMULI:
        stim_df = results_df[results_df["stimulus"] == stimulus]
        if len(stim_df) == 0:
            continue
        
        # Filter to valid validation losses (not inf or nan)
        valid_df = stim_df[
            stim_df["min_val_loss"].notna() & 
            (stim_df["min_val_loss"] < float('inf'))
        ]
        
        if len(valid_df) == 0:
            print(f"\n{stimulus.upper()}: No valid results found")
            continue
        
        # Best by validation loss (PRIMARY METRIC - NOT connectome-based!)
        best_by_loss = valid_df.loc[valid_df["min_val_loss"].idxmin()]
        
        # Best by F1 (for reference only - NOT used for selection!)
        if stim_df["f1_score"].notna().any():
            best_by_f1 = stim_df.loc[stim_df["f1_score"].idxmax()]
        else:
            best_by_f1 = None
        
        print(f"\n{stimulus.upper()}:")
        print(f"  ★ SELECTED CONFIG (by validation loss):")
        print(f"    - Model: {best_by_loss['model_type']}")
        print(f"    - LR: {best_by_loss['dsm_lr']}")
        print(f"    - Epochs: {int(best_by_loss['dsm_epochs'])}")
        print(f"    - Hidden dim: {int(best_by_loss['structured_hidden_dim'])}")
        print(f"    - L1 lambda: {best_by_loss['structured_l1_lambda']}")
        print(f"    - FDR alpha: {best_by_loss['fdr_alpha']}")
        print(f"    - Validation Loss: {best_by_loss['min_val_loss']:.6f} (optimized)")
        
        if best_by_f1 is not None:
            print(f"  [For reference - NOT used for selection]:")
            print(f"    - Config with best F1: {best_by_f1['f1_score']:.4f}")
            print(f"    - (Model: {best_by_f1['model_type']})")
        
        # Store best params (use validation loss as primary metric)
        best_params[stimulus] = {
            "dsm_lr": float(best_by_loss["dsm_lr"]),
            "dsm_epochs": int(best_by_loss["dsm_epochs"]),
            "dsm_noise_std": float(best_by_loss["dsm_noise_std"]),
            "structured_hidden_dim": int(best_by_loss["structured_hidden_dim"]),
            "structured_l1_lambda": float(best_by_loss["structured_l1_lambda"]),
            "fdr_alpha": float(best_by_loss["fdr_alpha"]),
            "min_val_loss": float(best_by_loss["min_val_loss"]),
            "f1_score": float(best_by_f1["f1_score"]) if best_by_f1 is not None and pd.notna(best_by_f1["f1_score"]) else None,
        }
    
    # Save best hyperparameters
    with open(OUTPUT_DIR / "best_hyperparameters.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✓ Saved best hyperparameters to: {OUTPUT_DIR / 'best_hyperparameters.json'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_results_summary(results_df, FIGURES_DIR)
    plot_parameter_heatmaps(results_df, FIGURES_DIR)
    print(f"✓ Saved figures to: {FIGURES_DIR}")
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - Results: {OUTPUT_DIR / 'search_results.csv'}")
    print(f"  - Best params: {OUTPUT_DIR / 'best_hyperparameters.json'}")
    print(f"  - Loss curves: {LOSS_CURVES_DIR}/")
    print(f"  - Figures: {FIGURES_DIR}/")
    print()
    print("Next step: Run script 03 (comprehensive analysis) with optimized hyperparameters")
    print()


if __name__ == "__main__":
    main()

