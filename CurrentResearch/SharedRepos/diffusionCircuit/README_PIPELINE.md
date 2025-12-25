# SBTG Pipeline - Complete Technical Documentation

**Last Updated:** December 25, 2025  
**Status:** Production - All scripts tested and verified (v2 with extended analyses)

This document provides exhaustive technical documentation for the complete SBTG (Score-Based Temporal Graphical models) analysis pipeline for C. elegans functional connectome inference.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Dependencies](#data-dependencies)
4. [SBTG Implementation Details](#sbtg-implementation-details)
5. [Pipeline Scripts (01-07)](#pipeline-scripts)
6. [Detailed Script Documentation](#detailed-script-documentation)
7. [Data Flow](#data-flow)
8. [Results Organization](#results-organization)
9. [Assumptions and Limitations](#assumptions-and-limitations)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose
This pipeline infers **functional connectivity** from C. elegans calcium imaging data using Score-Based Temporal Graphical (SBTG) models and validates results against the **structural connectome** from Cook et al. (2019).

### Core Functionality
- **Input:** NeuroPAL calcium imaging timeseries (∆F/F0 traces)
- **Processing:** SBTG model training with denoising score matching
- **Output:** Directed functional networks with edge signs and weights
- **Validation:** Comparison against anatomical connectome

### Key Capabilities
1. **Multiple model architectures:** Linear, Feature-Bilinear, Regime-Gated
2. **Flexible train/test splits:** Prefix, Per-stimulus prefix, Random, Odd-even
3. **Statistical rigor:** FDR-corrected edge detection, HAC variance estimation
4. **Comprehensive evaluation:** Precision/recall, weighted correlations, network analysis
5. **Production-quality visualizations:** Network diagrams, degree distributions, motif analysis

---

## System Architecture

```
CURRENT/
├── pipeline/                    # Analysis scripts (01-10)
├── sbtg/                       # SBTG implementation
│   ├── sbtg_main.py           # Main API (SBTGStructuredVolatilityEstimator)
│   ├── __init__.py            # Package init
│   └── old_impl.py            # Legacy implementation (not used)
└── data/                       # Input datasets
    ├── Head_Activity_OH15500.mat          # NaCl stimulus
    ├── Head_Activity_OH16230.mat          # Pentanedione, Butanone
    ├── SI 6 Cell class lists.xlsx         # Cook neuron metadata
    └── SI 7 Cell class connectome....xlsx # Cook connectome matrices
```

### Technology Stack
- **Python:** 3.10+
- **Deep Learning:** PyTorch 2.x (for score network training)
- **Scientific Computing:** NumPy, SciPy, pandas
- **Visualization:** matplotlib, seaborn, NetworkX
- **Statistics:** scikit-learn, statsmodels

---

## Data Dependencies

### 1. NeuroPAL Calcium Imaging Data

**Files:**
- `Head_Activity_OH15500.mat` (NaCl)
- `Head_Activity_OH16230.mat` (Pentanedione, Butanone)

**Format:** MATLAB `.mat` files (v7.3 compatible)

**Structure:**
```python
{
    'neurons': (109, 1) cell array       # Neuron names (strings)
    'norm_traces': (109, 1) cell array   # ∆F/F0 traces per neuron
    'stim_names': (3, 1) cell array      # ["butanone", "nacl", "pentanedione"]
    'stim_times': (3, 2) array           # [start_frame, end_frame] per stimulus
    'stims': (21, 3) array               # Per-worm stimulus indices
    'files': (21, 1) cell array          # Worm IDs
    'fps': scalar                         # Frame rate (4 Hz)
    'is_L': (109, 1) boolean             # Left/right laterality
}
```

**Neuron Traces Format:**
- Each entry in `norm_traces[i]` is a cell array of shape `(2×n_worms,)`
- First `n_worms` entries: measurements from first recording session
- Next `n_worms` entries: measurements from bilateral neurons (if applicable)
- Each measurement: 1D array of fluorescence values over time

**Key Properties:**
- **Neurons:** 109 total (57 overlap with structural connectome)
- **Timepoints:** Varies by stimulus (~20 for NaCl/Pentanedione, 1 for Butanone)
- **Worms:** 21 individuals
- **Sampling rate:** 4 Hz

**Preprocessing Applied:**
- Background subtraction
- ∆F/F0 normalization (dFF = (F - F0) / F0)
- NeuroPAL-based neuron identification

### 2. Structural Connectome (Cook et al. 2019)

**Files:**
- `SI 6 Cell class lists.xlsx` - Neuron metadata (class, type, lineage)
- `SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx` - Adjacency matrices

**Sheets in SI 7:**
- `Hermaphrodite_Chemical` - Chemical synapse counts
- `Hermaphrodite_Electrical` - Gap junction counts

**Matrix Format:**
- **Dimensions:** 95×95 cell classes (hermaphrodite head)
- **Values:** Synapse/gap junction counts (integers ≥ 0)
- **Type:** Directed (rows = postsynaptic, columns = presynaptic)

**Processing:**
- Extract overlapping neurons between NeuroPAL (109) and Cook (95)
- Result: 57 neurons with both functional and structural data
- Combined weight: `A_struct = A_chem + 0.5 * A_gap`

### 3. Intermediate Data Generated

After running `01_prepare_data.py`:

**Connectome Alignment:**
```
results/intermediate/connectome/
├── A_chem.npy              # Chemical synapse matrix (57×57)
├── A_gap.npy               # Gap junction matrix (57×57)
├── A_struct.npy            # Combined structural matrix (57×57)
└── nodes.json              # List of 57 overlapping neuron names
```

**Lag-Window Datasets:**
```
results/intermediate/datasets/{stimulus}/
├── Z_std.npy               # Standardized lag-window data (N×2n)
├── standardization.json    # Mean/std for each neuron
└── metadata.json           # Dataset info (n_samples, n_neurons, etc.)
```

**Lag-Window Format:**
- **Input:** Time series X of shape (T, n)
- **Output:** Paired observations Z = [X_t, X_{t+1}] of shape (T-1, 2n)
- **Standardization:** Each neuron zero-meaned and unit-variance

---

## SBTG Implementation Details

### Core File: `sbtg/sbtg_main.py`

**Main Class:** `SBTGStructuredVolatilityEstimator`

**Model Types:**
1. **Linear:** Simple cross-lag linear model
   - Energy: $U(x_t, x_{t+1}) = g_0(x_t) + g_1(x_{t+1}) + x_{t+1}^T W x_t$
   - Fastest training, good baseline

2. **Feature-Bilinear:** Adds feature transformations
   - Includes nonlinear feature mappings before bilinear term
   - Best performance in testing (F1=0.353)

3. **Regime-Gated:** Adds regime-dependent gating
   - Dynamically adjusts connectivity based on state
   - Most complex, longer training time

### Key Hyperparameters

**Denoising Score Matching (DSM):**
- `dsm_epochs`: Training epochs (default: 100)
- `dsm_batch_size`: Mini-batch size (default: 128)
- `dsm_lr`: Learning rate (default: 1e-3)
- `dsm_noise_std`: Noise level for denoising (default: 0.1)

**Network Architecture:**
- `structured_hidden_dim`: Hidden layer size (default: 64)
- `structured_num_layers`: Number of layers (default: 2)
- `structured_init_scale`: W initialization scale (default: 0.1)
- `structured_l1_lambda`: L1 regularization on W (default: 0.0)

**Statistical Testing:**
- `fdr_alpha`: False discovery rate threshold (default: 0.1)
- `fdr_method`: FDR correction method ("bh" or "fdr_by", default: "bh")
- `hac_max_lag`: HAC variance estimation lags (default: 5)

**Data Processing:**
- `window_length`: Lookback window for volatility (default: 2)
- `train_frac`: Train/test split fraction (default: 0.7)

### Train/Test Split Strategies

**1. Prefix (`prefix`):**
- Train: First 70% of timepoints
- Test: Last 30% of timepoints
- **Use case:** Time-ordered split, tests future prediction
- **Assumption:** Stationarity across time

**2. Per-stimulus Prefix (`per_stimulus_prefix`):**
- Train: First 70% per stimulus per worm
- Test: Last 30% per stimulus per worm
- **Use case:** Ensures all stimuli in train/test
- **Assumption:** Within-stimulus stationarity

**3. Random (`random`):**
- Train: Random 70% of timepoints
- Test: Remaining 30%
- **Use case:** IID assumption, no temporal structure
- **Assumption:** Samples independent

**4. Odd-Even (`odd_even`):**
- Train: Even-indexed timepoints
- Test: Odd-indexed timepoints
- **Use case:** Alternating pattern, good for autocorrelated data
- **Assumption:** Minimal temporal autocorrelation

### Output: `SBTGVolatilityResult`

**Attributes:**
- `sign_adj`: (n, n) binary adjacency with signs {-1, 0, +1}
  - +1: Excitatory connection (i → j increases j's activity)
  - -1: Inhibitory connection (i → j decreases j's activity)
  - 0: No significant connection
  
- `volatility_adj`: (n, n) binary volatility adjacency {0, 1}
  - 1: Connection affects volatility (not just mean)
  - 0: No volatility effect

- `p_mean`: (n, n) p-values for mean transfer test
  - Tests: H0: W[i,j] = 0 (no mean effect)

- `p_volatility`: (n, n) p-values for volatility transfer test
  - Tests: H0: No change in volatility

- `mu_hat`: (n, n) coupling weight matrix
  - **Feature-Bilinear:** Extracted from W parameter
  - **Linear:** Direct W matrix values
  - **Regime-Gated:** Averaged across regimes
  - **Magnitude:** Represents functional coupling strength

**Statistical Framework:**
1. **Null Hypothesis (Mean):** Neuron i has no directed effect on neuron j's mean activity
2. **Test Statistic:** Denoising score-based transfer test
3. **Variance Estimation:** HAC (Newey-West) with up to 5 lags
4. **Multiple Testing Correction:** Benjamini-Hochberg or Benjamini-Yekutieli FDR
5. **Significance Threshold:** α = 0.1 (default, configurable to 0.05)

---

## Pipeline Scripts

### Complete Pipeline Sequence

```
01_prepare_data.py              [Data preparation & connectome alignment]
    ↓
02_hyperparameter_search.py     [Grid search for optimal hyperparameters]
    ↓
03_comprehensive_analysis.py    [Multi-config SBTG training]
    ↓
04_connectome_comparison.py     [Binary edge evaluation + extended analyses]
    ↓
05_weighted_analysis.py         [Weighted correlation analysis]
    ↓
06_network_visualizations.py    [Network diagrams & degree distributions]
    ↓
07_generate_summary.py          [Comprehensive summary report]
```

**Archived scripts** (in `pipeline/archive/`, gitignored):
- Old scripts were superseded by the new streamlined 7-script pipeline

### Script Status

| Script | Description | Status | Runtime | Dependencies |
|--------|-------------|--------|---------|--------------|
| 01_prepare_data.py | Connectome alignment & lag-window datasets | ✅ Production | ~2 min | data/*.mat, *.xlsx |
| 02_hyperparameter_search.py | Grid search (validation loss metric) | ✅ Production | ~5 min (quick), ~1 hr (full) | 01 |
| 03_comprehensive_analysis.py | Multi-config training (3×4×2 configs) | ✅ Production | ~15 min | 01, 02 (optional) |
| 04_connectome_comparison.py | Binary edge evaluation, path-2, permutation tests | ✅ Production | ~3 min | 03 |
| 05_weighted_analysis.py | Weight correlation (W_param vs synapse counts) | ✅ Production | ~2 min | 03 |
| 06_network_visualizations.py | Network graphs, degree distributions | ✅ Production | ~5 min | 03 |
| 07_generate_summary.py | Final SUMMARY_REPORT.md generation | ✅ Production | ~30 sec | 03, 04, 05, 06 |

---

## Detailed Script Documentation

### Script 01: `01_prepare_data.py`

**Purpose:** Data preparation and connectome alignment

**Inputs:**
- `data/Head_Activity_OH16230.mat` - NeuroPAL recordings
- `data/SI 6 Cell class lists.xlsx` - Neuron metadata
- `data/SI 7 Cell class connectome....xlsx` - Structural adjacency

**Outputs:**
- `results/intermediate/connectome/A_chem.npy` - Chemical synapses (57×57)
- `results/intermediate/connectome/A_gap.npy` - Gap junctions (57×57)
- `results/intermediate/connectome/A_struct.npy` - Combined (57×57)
- `results/intermediate/connectome/nodes.json` - Neuron names (57)
- `results/intermediate/datasets/{stim}/Z_std.npy` - Standardized lag-window data
- `results/tables/connectome/name_alignment.csv` - Alignment table
- `results/figures/connectome/` - Heatmaps and histograms

**Key Functions:**
```python
def load_grouped_matrix(path, sheet_name):
    """Load Cook adjacency matrix from Excel"""
    # Handles merged cells, extracts neuron names
    # Returns pandas DataFrame with neuron names as index/columns

def normalize_name(name):
    """Normalize neuron names to uppercase"""
    # AVAL → AVAL, avar → AVAL
    
def align_connectome_to_neurons(A_struct, struct_neurons, func_neurons):
    """Find overlapping neurons and align matrices"""
    # Input: Full structural (95×95), functional names (109)
    # Output: Aligned structural (57×57), functional subset (57)
    
def build_lag_window_dataset(traces, window_length=2):
    """Create [X_t, X_{t+1}] pairs for SBTG"""
    # Input: (T, n) time series
    # Output: (T-1, 2n) lag-window format
```

**Processing Steps:**
1. Load NeuroPAL neuron names from MAT file
2. Load Cook connectome matrices from Excel
3. Normalize all neuron names (uppercase)
4. Find overlapping neurons (intersection)
5. Extract aligned submatrices (57×57)
6. Combine: A_struct = A_chem + 0.5 * A_gap
7. Build lag-window datasets per stimulus
8. Standardize (zero mean, unit variance per neuron)
9. Save intermediate files
10. Generate QC visualizations

**Assumptions:**
- Neuron names can be normalized to match
- Chemical synapses and gap junctions combine linearly
- 0.5 weight for gap junctions (symmetric, bidirectional)
- All neurons measured across sufficient worms (≥18)

**Configuration:**
```python
MIN_WORMS = 18              # Minimum worms for neuron inclusion
STIMULI = None              # Process all stimuli (or specify list)
WINDOW_LENGTH = 2           # Lag window size
```

---

### Script 02: `02_train_models.py`

**Purpose:** Train SBTG models for each stimulus (single configuration)

**Inputs:**
- `results/intermediate/datasets/{stim}/standardization.json` - Node ordering
- `data/Head_Activity_OH16230.mat` - Raw NeuroPAL data (loaded per stimulus)

**Outputs:**
- `results/models/sbtg/best/{stim}/result.npz` - Trained model results
  - Arrays: sign_adj, volatility_adj, p_mean, p_volatility, mu_hat
- `results/models/sbtg/best/{stim}/config.json` - Hyperparameters
- `results/figures/sbtg_training/` - Training diagnostics (if verbose)

**Key Functions:**
```python
def load_neuropal_data():
    """Load NeuroPAL MAT file and extract relevant fields"""
    # Returns dict with neuron_names, norm_traces, stim_info

def build_stimulus_timeseries(stimulus_name, neuropal_data):
    """Aggregate traces across worms for one stimulus"""
    # Handles bilateral neurons (average L/R)
    # Returns: (T, n) matrix, DatasetStatistics

def train_sbtg_model(X, model_type='linear', train_split='prefix'):
    """Train SBTG model"""
    estimator = SBTGStructuredVolatilityEstimator(
        model_type=model_type,
        train_split=train_split,
        window_length=5,
        fdr_alpha=0.05,
        dsm_epochs=100
    )
    result = estimator.fit(X)
    return result
```

**Processing Steps:**
1. Load NeuroPAL data
2. For each stimulus:
   a. Build timeseries matrix (T, n)
   b. Create train/test split
   c. Initialize SBTG estimator
   d. Train score network (DSM)
   e. Compute transfer tests
   f. FDR correction
   g. Extract sign_adj, volatility_adj, mu_hat
   h. Save results

**Model Selection:**
```python
# Default configuration (editable)
MODEL_TYPE = 'feature_bilinear'  # Best performer
TRAIN_SPLIT = 'per_stimulus_prefix'  # Most reliable
WINDOW_LENGTH = 5
FDR_ALPHA = 0.05  # More conservative than default 0.1
```

**Assumptions:**
- Traces from different worms can be aggregated
- Bilateral neurons (L/R pairs) can be averaged
- Data is approximately stationary within stimulus
- Sufficient timepoints for DSM convergence (≥10)

---

### Script 03: `03_evaluate_results.py`

**Purpose:** Evaluate functional networks against structural connectome

**Inputs:**
- `results/models/sbtg/best/{stim}/result.npz` - Trained results
- `results/intermediate/connectome/A_struct.npy` - Ground truth

**Outputs:**
- `results/tables/evaluation/edge_metrics.csv` - Precision, recall, F1
- `results/figures/evaluation/roc_curves.png` - ROC analysis
- `results/figures/evaluation/pr_curves.png` - Precision-recall curves
- `results/reports/evaluation_summary.md` - Markdown summary

**Key Metrics:**
```python
# Binary edge detection
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# ROC analysis
roc_auc = area_under_ROC_curve

# Weighted correlation (if mu_hat available)
pearson_r = correlation(|mu_hat|, structural_weights)
spearman_r = rank_correlation(|mu_hat|, structural_weights)
```

**Evaluation Framework:**
1. **Binary Edges:**
   - Functional: sign_adj != 0
   - Structural: A_struct > 0
   - Metrics: Precision, Recall, F1, ROC-AUC

2. **Weighted Edges:**
   - Functional: |mu_hat| (coupling strength)
   - Structural: A_struct (synapse count)
   - Metrics: Pearson r, Spearman ρ

3. **Top-k Analysis:**
   - Precision among k strongest functional edges
   - k ∈ [10, 50, 100, 200, 500]

**Assumptions:**
- Functional edges should overlap with structural edges
- Edge weight magnitude correlates with synapse count
- Neurons aligned correctly between datasets

---

### Script 04: `04_analyze_circuits.py`

**Purpose:** Circuit motif analysis and network visualization

**Inputs:**
- `results/models/sbtg/best/{stim}/result.npz` - Trained results
- `results/intermediate/connectome/A_struct.npy` - Structural reference

**Outputs:**
- `results/tables/circuit_dynamics/motif_counts.csv` - Motif enumeration
- `results/figures/circuits/{stim}_network.png` - Network visualization
- `results/figures/circuits/{stim}_motifs.png` - Motif diagrams
- `results/reports/circuit_analysis.md` - Analysis summary

**Key Analyses:**
1. **Motif Detection:**
   - Feed-forward loops (A→B→C, A→C)
   - Feedback loops (A→B→A)
   - Triangles (A↔B↔C↔A)
   - Convergent/Divergent patterns

2. **Network Properties:**
   - Degree distributions (in/out/total)
   - Hub identification (high-degree nodes)
   - Clustering coefficient
   - Path length statistics

3. **Edge Type Analysis:**
   - Excitatory vs Inhibitory balance
   - Volatility edge prevalence
   - Sign homophily in motifs

**Visualization Features:**
```python
# NetworkX graph visualization
- Node colors: By neuron type (sensory, inter, motor)
- Node sizes: By degree centrality
- Edge colors: Red (inhibitory), Blue (excitatory), Green (volatility)
- Edge widths: By |mu_hat| magnitude
- Layout: Spring (force-directed) or hierarchical
```

**Assumptions:**
- Motifs reflect functional circuits
- Edge signs indicate circuit function (exc/inh balance)
- Hub neurons play central roles in information processing

---

### Script 05: `05_comprehensive_sbtg_analysis.py`

**Purpose:** Exhaustive multi-configuration training (PRODUCTION PIPELINE)

**Configuration Matrix:**
- **Models:** 3 (linear, feature_bilinear, regime_gated)
- **Splits:** 4 (prefix, per_stimulus_prefix, random, odd_even)
- **Stimuli:** 3 (nacl, pentanedione, butanone)
- **Total:** 3 × 4 × 3 = 36 configurations

**Inputs:**
- `data/Head_Activity_OH15500.mat` - NaCl
- `data/Head_Activity_OH16230.mat` - Pentanedione, Butanone

**Outputs:**
- `results/comprehensive_sbtg/all_results.csv` - Summary table (36 rows)
- `results/comprehensive_sbtg/models/{stim}/{config}/result.npz` - Individual results
- `results/comprehensive_sbtg/models/{stim}/{config}/neuron_names.json` - Neuron list
- `results/comprehensive_sbtg/figures/dataset_summary.png` - Data overview

**Key Features:**
1. **Parallel Training:** All configs trained sequentially
2. **Robust Error Handling:** Skips failed configs (e.g., butanone)
3. **Comprehensive Logging:** Timestamps, edge counts, metadata
4. **mu_hat Extraction:** Saves coupling weights for weighted analysis

**Hyperparameters (Fixed):**
```python
DEFAULT_HYPERPARAMS = {
    "window_length": 5,
    "dsm_epochs": 100,
    "dsm_batch_size": 128,
    "dsm_lr": 1e-3,
    "train_frac": 0.7,
    "hac_max_lag": 5,
    "fdr_alpha": 0.05,  # More stringent
    "fdr_method": "fdr_by",  # Benjamini-Yekutieli (conservative)
    "structured_hidden_dim": 64,
    "structured_num_layers": 2,
    "structured_l1_lambda": 0.001,
    "structured_init_scale": 0.1
}
```

**Output Schema (`all_results.csv`):**
```
model_type          | linear, feature_bilinear, regime_gated
train_split         | prefix, per_stimulus_prefix, random, odd_even
n_timepoints        | Number of training timepoints
n_neurons           | 109 (all) or subset
n_sign_edges        | Count of sign_adj != 0
n_volatility_edges  | Count of volatility_adj != 0
n_total_edges       | sign + volatility (union)
hyperparameters     | JSON string of config
timestamp           | ISO 8601 timestamp
stimulus            | nacl, pentanedione, butanone
```

**Data Handling:**
```python
def load_neuropal_data():
    """Load MAT file with simplify_cells=True"""
    # Handles nested cell arrays
    # Extracts: neuron_names, norm_traces, stim_info

def collect_worm_trace(neuron_traces, worm_idx, num_worms):
    """Average bilateral neurons (L/R)"""
    # Checks both worm_idx and worm_idx + num_worms
    # Aligns to shortest segment
    # Returns averaged trace

def build_stimulus_timeseries(stimulus_name, neuropal_data, neuron_subset=None):
    """Aggregate all worms for one stimulus"""
    # Finds stimulus index in stim_names
    # Loops over worms, collects traces
    # Concatenates into (T, n) matrix
```

**Quality Control:**
- Minimum 10 timepoints required (butanone fails with 1)
- Checks for NaN/Inf in traces
- Validates neuron overlap with connectome
- Logs all warnings and errors

**Assumptions:**
- All model types use same hyperparameters (fair comparison)
- Sufficient data for all splits (prefix may fail if T < 15)
- Bilateral neurons can be averaged without loss

---

### Script 06: `06_connectome_comparison.py`

**Purpose:** Binary edge evaluation with extended analyses (precision/recall/F1, path-2, baselines, permutation tests)

**Inputs:**
- `results/comprehensive_sbtg/all_results.csv` - Config metadata
- `results/comprehensive_sbtg/models/{stim}/{config}/result.npz` - Model outputs
- `results/intermediate/connectome/A_struct.npy` - Ground truth (57×57)

**Outputs:**
- `results/comprehensive_sbtg/connectome_comparison/edge_comparisons.csv` - Core metrics + random baseline
- `results/comprehensive_sbtg/connectome_comparison/path2_comparisons.csv` - **NEW:** Indirect connection analysis
- `results/comprehensive_sbtg/connectome_comparison/permutation_tests.csv` - **NEW:** Statistical significance
- `results/comprehensive_sbtg/connectome_comparison/cross_stimulus_consistency.json` - **NEW:** Edge reproducibility
- `results/comprehensive_sbtg/connectome_comparison/reports/extended_analysis.md` - **NEW:** Comprehensive report
- `results/comprehensive_sbtg/connectome_comparison/figures/` - Confusion matrices, weight correlations

**Core Metrics:**
```python
# Binary classification
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Edge counts
n_functional_edges = sign_adj != 0
n_structural_edges = A_struct > 0
```

**Extended Analyses (v2):**

1. **Path-2 Analysis (`compute_path2_analysis`)**
   - Computes 2-hop reachability: A² matrix
   - Identifies indirect structural paths
   - Key finding: 1,877 path-2 only connections vs 808 direct edges

2. **Latent Confounder Detection**
   - 43 neurons in connectome not measured
   - 1,117 path-2 connections go through unmeasured nodes
   - 42.6% of indirect paths involve latent confounders

3. **Random Baseline (`compute_random_baseline`)**
   - Expected F1 if same number of edges placed randomly
   - Formula: `random_precision = n_struct_edges / n_possible`
   - Key finding: Only 41.7% of configs beat random baseline

4. **Permutation Test (`run_permutation_test`)**
   - 500 permutations of functional adjacency
   - Tests H0: observed F1 no better than chance
   - Reports p-value and significance at α=0.05, α=0.01

5. **Cross-Stimulus Consistency (`compute_cross_stimulus_consistency`)**
   - Compares edge predictions across NaCl and Pentanedione
   - Reports mean ± std F1 per configuration
   - Best: feature_bilinear/per_stimulus_prefix: F1 = 0.356 ± 0.001

**Best Results:**
- **F1 Score:** 0.356 (Pentanedione + Feature-Bilinear + per_stimulus_prefix)
- **Recall:** 61.7% (same config)
- **Path-2 F1:** 0.703 (when counting indirect paths as TPs)
- **FPs Explained by Path-2:** 76.8%

**Key Functions Added (v2):**
```python
def compute_path2_analysis(A_struct):
    """Compute 2-hop reachability matrix and indirect connection stats."""

def compare_with_path2_ground_truth(A_func, A_struct, A_reachable_2):
    """Evaluate against A + A² ground truth."""

def compute_random_baseline(n_func_edges, n_struct_edges, n_possible):
    """Expected metrics for random edge predictor."""

def compute_cross_stimulus_consistency(comparisons):
    """Analyze edge consistency across stimuli."""

def run_permutation_test(A_func, A_struct, n_permutations=500):
    """Statistical significance via permutation test."""
```

**Interpretation:**
- **Recall 61.7%:** SBTG detects majority of structural edges
- **Precision ~25%:** Many functional edges lack direct structural counterparts
- **Path-2 explains FPs:** 76.8% of "false positives" are actually indirect connections
- **Latent confounders:** 43 unmeasured neurons create hidden pathways

---

### Script 07: `07_weighted_connectome_analysis.py`

**Purpose:** Weighted correlation analysis (mu_hat vs synapse counts)

**Inputs:**
- `results/comprehensive_sbtg/all_results.csv` - Config metadata
- `results/comprehensive_sbtg/models/{stim}/{config}/result.npz` - Model outputs (with mu_hat)
- `results/intermediate/connectome/A_struct.npy` - Structural weights (synapse counts)

**Outputs:**
- `results/comprehensive_sbtg/weighted_analysis/weight_correlations.csv` - Correlation metrics
- `results/comprehensive_sbtg/weighted_analysis/weighted_edge_details.json` - Edge-level data
- `results/comprehensive_sbtg/weighted_analysis/figures/` - Distribution plots (72 total)

**Metrics Computed:**
```python
# For edges present in both networks (overlap)
overlap_edges = (sign_adj != 0) & (A_struct > 0)

# Extract weights
func_weights = |mu_hat[overlap_edges]|  # Absolute coupling
struct_weights = A_struct[overlap_edges]  # Synapse counts

# Correlations
pearson_r, pearson_p = pearsonr(func_weights, struct_weights)
spearman_r, spearman_p = spearmanr(func_weights, struct_weights)
```

**Weight Extraction:**
```python
def extract_wparam_weights(result_npz):
    """Extract mu_hat from result file"""
    # Priority: mu_hat > W_param > sign_adj
    if 'mu_hat' in data:
        return data['mu_hat']
    elif 'W_param' in data:
        return data['W_param']
    else:
        return data['sign_adj']  # Fallback
```

**Key Findings (Verified):**
- **Best Spearman ρ:** 0.112 (p=0.0015) - Pentanedione + Feature-Bilinear + Random
- **Average ρ:** 0.01 ± 0.04 - Essentially no correlation
- **Weight Scales:** 
  - Functional: 0.05 ± 0.06
  - Structural: 12.1 ± 28.1
  - **240× difference!**

**Interpretation:**
- **Weak correlations:** Functional coupling strength ≠ anatomical synapse count
- **Biological reasons:**
  - **Activity-dependent weighting:** Synapses modulated by neuromodulators, plasticity
  - **Functional vs anatomical:** Not all synapses active during stimulus
  - **Scale mismatch:** SBTG estimates effective coupling, not synapse number
  - **Indirect paths:** Functional connections may reflect multi-synapse paths

**Visualizations:**
1. **Scatter plots:** |mu_hat| vs synapse count (per config)
2. **Distribution histograms:** Functional vs structural weight distributions
3. **Overlap analysis:** Venn diagrams of edge sets

**Assumptions:**
- mu_hat represents functional coupling strength
- Stronger coupling should correlate with more synapses (weak assumption!)
- Overlapping edges are correctly aligned

---

### Script 08: `08_generate_network_visualizations.py`

**Purpose:** Publication-quality network diagrams (AGGREGATE graphs)

**Inputs:**
- `results/comprehensive_sbtg/all_results.csv` - Config list
- `results/comprehensive_sbtg/models/{stim}/{config}/result.npz` - Adjacencies, mu_hat

**Outputs:**
- `results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/network_graph.png`
- `results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/degree_distributions.png`
- `results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/edge_properties.png`
- `results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/node_statistics.csv`
- `results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/graph_statistics.json`

**Network Visualization:**
```python
# NetworkX directed graph
G = nx.DiGraph()

# Nodes
- All neurons (n=109 or subset)
- Size: 100 + 400 * (total_strength / max_strength)
- Color: Gray (#34495e)

# Edges (top 200 strongest shown)
- Color: RdBu_r colormap
  - Blue (+): Excitatory (sign_adj > 0)
  - Red (-): Inhibitory (sign_adj < 0)
- Width: 0.5 + 2.5 * (|mu_hat| / max_|mu_hat|)
- Alpha: 0.6 (transparency)

# Layout
- Spring layout (NetworkX)
- Seed: 42 (reproducibility)
- k=0.5, iterations=50
```

**Degree Distributions:**
- In-degree histogram
- Out-degree histogram
- Total degree histogram
- Top 10 hub neurons (bar chart)

**Edge Properties:**
- Signed weight distribution
- Absolute weight distribution
- Excitatory vs Inhibitory counts (bar chart)
- Cumulative weight distribution (80% line)

**Node Statistics (CSV):**
```
neuron | in_degree | out_degree | total_degree | in_strength | out_strength | 
       | total_strength | in_excitatory | in_inhibitory | out_excitatory | out_inhibitory
```

**Graph Statistics (JSON):**
```json
{
  "stimulus": "nacl",
  "model_type": "feature_bilinear",
  "train_split": "per_stimulus_prefix",
  "n_nodes": 109,
  "n_edges": 4513,
  "density": 0.384,
  "avg_in_degree": 41.4,
  "avg_out_degree": 41.4,
  "n_excitatory": 2456,
  "n_inhibitory": 2057
}
```

**Assumptions:**
- Top 200 edges sufficient for visualization clarity
- Node size by strength indicates importance
- Edge color by sign reveals circuit function

---

### Script 09: `09_parameter_sweep.py` ⚠️ IN DEVELOPMENT

**Purpose:** Parameter optimization to improve connectome matching

**Target Parameters:**
- `fdr_alpha`: [0.01, 0.05, 0.1, 0.15, 0.2]
- `sign_threshold`: [0.0, 0.1, 0.2, 0.3, 0.5]
- `l1_lambda`: [0.0, 0.0001, 0.001, 0.01]

**Total Combinations:** 5 × 5 × 4 = 100 parameter sets

**Test Configurations:**
- **Models:** feature_bilinear, linear (best 2)
- **Splits:** per_stimulus_prefix, random (best 2)
- **Stimuli:** nacl, pentanedione

**Outputs (Planned):**
- `results/comprehensive_sbtg/parameter_sweep/parameter_sweep_results.csv`
- `results/comprehensive_sbtg/parameter_sweep/best_parameters.csv`
- `results/comprehensive_sbtg/parameter_sweep/heatmap_*.png`

**Status:** Data loading needs refinement for NeuroPAL structure

---

### Script 10: `10_generate_summary.py`

**Purpose:** Comprehensive summary report with overfitting and baseline analysis

**Inputs:**
- `results/comprehensive_sbtg/all_results.csv`
- `results/comprehensive_sbtg/connectome_comparison/edge_comparisons.csv`
- `results/comprehensive_sbtg/connectome_comparison/path2_comparisons.csv` - **NEW**
- `results/comprehensive_sbtg/connectome_comparison/permutation_tests.csv` - **NEW**
- `results/comprehensive_sbtg/connectome_comparison/cross_stimulus_consistency.json` - **NEW**
- `results/comprehensive_sbtg/weighted_analysis/weight_correlations.csv`

**Outputs:**
- `results/comprehensive_sbtg/summary/SUMMARY_REPORT.md` - **Master report with extended analyses**
- `results/comprehensive_sbtg/summary/summary_statistics.csv`
- `results/comprehensive_sbtg/summary/best_configurations.csv`
- `results/comprehensive_sbtg/summary/model_comparison.png`
- `results/comprehensive_sbtg/summary/stimulus_comparison.png`

**Extended Analyses (v2):**

1. **Overfitting Analysis (`compute_overfitting_analysis`)**
   - Compares random vs prefix split F1 scores
   - Flags when random/prefix ratio > 1.5x
   - Key finding: Average ratio 2.19x, max 3.80x (overfitting warning!)

2. **Random Baseline Summary (`compute_random_baseline_summary`)**
   - Aggregates F1 vs random baseline across all configs
   - Reports % of configs beating random
   - Key finding: Only 41.7% beat random, 0% are >1.5x better

**Report Sections (Updated):**
1. Summary statistics (total configs, avg edges, best metrics)
2. Best configurations (top per metric)
3. Edge detection performance (top 5 by F1)
4. Weighted correlation analysis (top 5 by Spearman ρ)
5. **⚠️ Overfitting Analysis** - NEW
6. **Random Baseline Comparison** - NEW
7. **Path-2 (Indirect Connection) Analysis** - NEW
8. **Cross-Stimulus Consistency** - NEW
9. Key findings (interpretation)
10. File structure overview

**Best Configurations Table:**
```
metric      | value  | stimulus      | model_type        | train_split
------------|--------|---------------|-------------------|------------------
f1_score    | 0.356  | pentanedione  | feature_bilinear  | per_stimulus_prefix
recall      | 0.617  | pentanedione  | feature_bilinear  | per_stimulus_prefix
precision   | 0.284  | nacl          | linear            | odd_even
spearman_r  | 0.095  | pentanedione  | regime_gated      | prefix
```

**Key Warnings from Report:**
- ⚠️ Potential overfitting: random/prefix ratio = 2.19x average
- ⚠️ Marginal improvement over random: only 1.16x at best
- ✅ Path-2 explains 76.8% of false positives
- ✅ Cross-stimulus consistency high for best config (F1 = 0.356 ± 0.001)

---

## Data Flow

### End-to-End Pipeline

```
Raw Data (data/)
    ↓
[01_prepare_data.py]
    ↓
Intermediate Data (results/intermediate/)
    ├── connectome/ (A_struct.npy, nodes.json)
    └── datasets/{stim}/ (Z_std.npy, metadata.json)
    ↓
[02_train_models.py] (Single config)
    ↓
Trained Models (results/models/sbtg/best/{stim}/)
    ├── result.npz (sign_adj, volatility_adj, p_mean, p_volatility, mu_hat)
    └── config.json
    ↓
[03_evaluate_results.py]
    ↓
Evaluation Results (results/tables/evaluation/, results/figures/evaluation/)
    ↓
[04_analyze_circuits.py]
    ↓
Circuit Analysis (results/tables/circuit_dynamics/, results/figures/circuits/)

OR (PRODUCTION PATH)

Raw Data (data/)
    ↓
[05_comprehensive_sbtg_analysis.py] (36 configs)
    ↓
Comprehensive Results (results/comprehensive_sbtg/)
    ├── all_results.csv
    └── models/{stim}/{config}/result.npz
    ↓
[06_connectome_comparison.py]
    ↓
Binary Edge Evaluation (results/comprehensive_sbtg/connectome_comparison/)
    ↓
[07_weighted_connectome_analysis.py]
    ↓
Weighted Correlation (results/comprehensive_sbtg/weighted_analysis/)
    ↓
[08_generate_network_visualizations.py]
    ↓
Network Diagrams (results/comprehensive_sbtg/visualizations/)
    ↓
[10_generate_summary.py]
    ↓
Final Report (results/comprehensive_sbtg/summary/SUMMARY_REPORT.md)
```

### File Format Specifications

**result.npz (NumPy archive):**
```python
{
    'sign_adj': np.ndarray(n, n, dtype=int8),     # {-1, 0, +1}
    'volatility_adj': np.ndarray(n, n, dtype=int8),  # {0, 1}
    'p_mean': np.ndarray(n, n, dtype=float32),    # [0, 1]
    'p_volatility': np.ndarray(n, n, dtype=float32),  # [0, 1]
    'mu_hat': np.ndarray(n, n, dtype=float32)     # Real-valued coupling weights
}
```

**Loading Example:**
```python
import numpy as np
data = np.load('result.npz')
sign_adj = data['sign_adj']      # Extract adjacency
mu_hat = data['mu_hat']          # Extract weights
data.close()                      # Close file
```

**config.json (Model metadata):**
```json
{
    "model_type": "feature_bilinear",
    "train_split": "per_stimulus_prefix",
    "window_length": 5,
    "fdr_alpha": 0.05,
    "fdr_method": "fdr_by",
    "dsm_epochs": 100,
    "n_neurons": 109,
    "n_timepoints": 20,
    "timestamp": "2025-12-24T19:40:51.365172"
}
```

**neuron_names.json (List):**
```json
[
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR",
    ...
]
```

---

## Results Organization

### Directory Structure

```
results/
├── comprehensive_sbtg/                    # Main results (scripts 05-10)
│   ├── all_results.csv                    # 36-row summary
│   ├── models/                            # Individual model outputs
│   │   ├── nacl/
│   │   │   ├── linear_prefix/
│   │   │   │   ├── result.npz
│   │   │   │   ├── neuron_names.json
│   │   │   │   └── graph_statistics.json
│   │   │   ├── linear_per_stimulus_prefix/
│   │   │   ├── ... (12 configs total for nacl)
│   │   ├── pentanedione/
│   │   │   └── ... (12 configs)
│   │   └── butanone/
│   │       └── ... (12 configs, most failed)
│   ├── connectome_comparison/             # Script 06 outputs (extended v2)
│   │   ├── edge_comparisons.csv           # Core metrics + random baseline ratio
│   │   ├── path2_comparisons.csv          # **NEW:** Indirect connection analysis
│   │   ├── permutation_tests.csv          # **NEW:** Statistical significance
│   │   ├── cross_stimulus_consistency.json # **NEW:** Edge reproducibility
│   │   ├── reports/
│   │   │   └── extended_analysis.md       # **NEW:** Comprehensive report
│   │   └── figures/
│   │       ├── nacl/
│   │       │   ├── linear_prefix/
│   │       │   │   ├── confusion_matrix.png
│   │       │   │   └── weight_correlation.png
│   │       │   └── ... (12 configs)
│   │       └── pentanedione/
│   ├── weighted_analysis/                 # Script 07 outputs
│   │   ├── weight_correlations.csv
│   │   ├── weighted_edge_details.json
│   │   ├── weight_correlation_summary.png
│   │   └── figures/
│   │       ├── nacl/
│   │       │   ├── linear_prefix/
│   │       │   │   ├── weight_scatter.png
│   │       │   │   ├── weight_distributions.png
│   │       │   │   └── overlap_analysis.png
│   │       │   └── ... (12 configs)
│   │       └── pentanedione/
│   ├── visualizations/                    # Script 08 outputs
│   │   └── figures/
│   │       ├── nacl/
│   │       │   ├── linear_prefix/
│   │       │   │   ├── network_graph.png
│   │       │   │   ├── degree_distributions.png
│   │       │   │   ├── edge_properties.png
│   │       │   │   ├── node_statistics.csv
│   │       │   │   └── graph_statistics.json
│   │       │   └── ... (12 configs)
│   │       └── pentanedione/
│   ├── parameter_sweep/                   # Script 09 (in development)
│   └── summary/                           # Script 10 outputs
│       ├── SUMMARY_REPORT.md              # **READ THIS FIRST** (includes overfitting analysis)
│       ├── summary_statistics.csv
│       ├── best_configurations.csv
│       ├── model_comparison.png
│       └── stimulus_comparison.png
│
├── models/sbtg/best/                      # Scripts 02-04 outputs
│   ├── nacl/
│   ├── pentanedione/
│   └── butanone/
│
├── intermediate/                          # Script 01 outputs
│   ├── connectome/
│   │   ├── A_chem.npy
│   │   ├── A_gap.npy
│   │   ├── A_struct.npy
│   │   └── nodes.json
│   └── datasets/
│       ├── nacl/
│       ├── pentanedione/
│       └── butanone/
│
├── figures/                               # Visualizations (scripts 01-04)
│   ├── connectome/
│   ├── sbtg_training/
│   ├── evaluation/
│   └── circuits/
│
├── tables/                                # Tables (scripts 01-04)
│   ├── connectome/
│   ├── evaluation/
│   └── circuit_dynamics/
│
└── reports/                               # Markdown reports (scripts 03-04)
    ├── evaluation_summary.md
    └── circuit_analysis.md
```

### Key Result Files

**Must Read:**
1. `results/comprehensive_sbtg/summary/SUMMARY_REPORT.md` - Overall summary
2. `results/comprehensive_sbtg/all_results.csv` - Config performance table

**Best Results:**
3. `results/comprehensive_sbtg/models/pentanedione/feature_bilinear_per_stimulus_prefix/result.npz` - Best F1
4. `results/comprehensive_sbtg/weighted_analysis/weight_correlations.csv` - Correlation analysis

**Visualizations:**
5. `results/comprehensive_sbtg/summary/model_comparison.png` - Model rankings
6. `results/comprehensive_sbtg/visualizations/figures/pentanedione/feature_bilinear_per_stimulus_prefix/network_graph.png` - Best network

---

## Assumptions and Limitations

### Data Assumptions

1. **NeuroPAL Neuron Identification:**
   - Neuron IDs are accurate (NeuroPAL atlas-based)
   - Bilateral neurons (L/R) can be averaged
   - Minimum 18 worms required per neuron

2. **Calcium Imaging:**
   - ∆F/F0 reflects neural activity (spike rate proxy)
   - 4 Hz sampling sufficient for dynamics (250 ms resolution)
   - Activity approximately stationary within stimulus

3. **Connectome Alignment:**
   - Neuron names match between datasets (after normalization)
   - Structural connectome (Cook 2019) is ground truth
   - 57-neuron overlap is representative

### Model Assumptions

1. **SBTG Framework:**
   - Score-based models can capture volatility structure
   - Denoising score matching provides good estimates
   - Structured energy form (g0 + g1 + x_t+1^T W x_t) is appropriate

2. **Statistical Testing:**
   - HAC variance estimation handles autocorrelation
   - FDR control (Benjamini-Yekutieli) is conservative enough
   - Transfer tests detect directed effects

3. **Causality:**
   - Granger causality (predictive) relates to structural connectivity
   - Time-lagged coupling W[i,j] indicates j → i
   - Confounders are minimal (stimulus-controlled)

### Known Limitations

1. **Timepoints:**
   - **NaCl/Pentanedione:** 20 timepoints (adequate)
   - **Butanone:** 1 timepoint (insufficient, all configs fail)
   - Minimum ~10-15 timepoints recommended

2. **Functional vs Anatomical:**
   - Weak weight correlations (ρ ≈ 0.01)
   - Functional coupling ≠ synapse count
   - Activity-dependent modulation not modeled

3. **Model Complexity:**
   - Feature-Bilinear best but more parameters (risk of overfitting)
   - Regime-Gated has convergence issues on small data
   - Linear may be too simple for complex dynamics

4. **Evaluation:**
   - Binary edge metrics ignore edge weights
   - Weighted correlation limited by scale mismatch
   - No temporal dynamics evaluation (aggregate graphs only)

5. **Computational:**
   - DSM training slow on CPU (~5-15 min per config)
   - Memory usage scales with O(n²) for adjacency matrices
   - Parameter sweep incomplete (data loading issues)

6. **Statistical Validity (v2 findings):**
   - ⚠️ Only 41.7% of configurations beat random baseline
   - ⚠️ Overfitting detected: random/prefix ratio = 2.19x average
   - ⚠️ Best F1 improvement over random: only 1.16x
   - ⚠️ No comparison against standard baseline methods yet

7. **Latent Confounders:**
   - 43 neurons in connectome not measured
   - 1,117 indirect paths through unmeasured intermediaries
   - Cannot distinguish mediated influence from confounding

---

## Baseline Methods (TODO - HIGH PRIORITY)

### Motivation

The current pipeline lacks comparison against standard functional connectivity methods. **Before publication, SBTG must be compared against:**

### Required Baselines

| Method | Type | Direction | Implementation |
|--------|------|-----------|----------------|
| **Pearson Correlation** | Correlation | Undirected | `np.corrcoef(X.T)` |
| **Partial Correlation** | Conditional | Undirected | `sklearn.covariance.EmpiricalCovariance` → invert |
| **Cross-Correlation (lag-1)** | Temporal | Directed | `scipy.signal.correlate` |
| **Granger Causality** | VAR | Directed | `statsmodels.tsa.stattools.grangercausalitytests` |
| **Transfer Entropy** | Information | Directed | Custom or `pyinform` |
| **GLASSO** | Sparse Covariance | Undirected | `sklearn.covariance.GraphicalLasso` |

### Implementation Plan

**Add to Script 06 or create new Script 11:**

```python
def compute_baseline_methods(X, A_struct, overlap_idx):
    """
    Compute baseline connectivity methods and compare to SBTG.
    
    Parameters:
    -----------
    X : ndarray (T, n)
        Time series data
    A_struct : ndarray (n, n)
        Ground truth structural adjacency
    overlap_idx : list
        Indices of neurons in both datasets
    
    Returns:
    --------
    baselines : dict
        {method_name: {adjacency, f1, precision, recall}}
    """
    
    results = {}
    
    # 1. Pearson Correlation
    corr = np.corrcoef(X.T)
    corr_adj = (np.abs(corr) > threshold)  # threshold at same density as SBTG
    results['pearson'] = evaluate_adjacency(corr_adj, A_struct)
    
    # 2. Partial Correlation
    from sklearn.covariance import EmpiricalCovariance
    cov = EmpiricalCovariance().fit(X)
    prec = np.linalg.inv(cov.covariance_)
    partial_corr = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    results['partial_corr'] = evaluate_adjacency(partial_corr, A_struct)
    
    # 3. Cross-Correlation (lag-1)
    # For each pair (i, j), compute corr(X[:-1, i], X[1:, j])
    xcorr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xcorr[i, j] = np.corrcoef(X[:-1, i], X[1:, j])[0, 1]
    results['crosscorr_lag1'] = evaluate_adjacency(xcorr, A_struct)
    
    # 4. Granger Causality
    from statsmodels.tsa.stattools import grangercausalitytests
    # ... (pairwise tests)
    
    # 5. GLASSO
    from sklearn.covariance import GraphicalLasso
    model = GraphicalLasso(alpha=0.1)
    model.fit(X)
    results['glasso'] = evaluate_adjacency(model.precision_, A_struct)
    
    return results
```

### Expected Outcome

**SBTG should significantly outperform simple baselines:**
- Target: SBTG F1 > 1.5× Pearson correlation F1
- Target: SBTG F1 > 1.5× Cross-correlation F1
- If SBTG ≈ Granger causality, justify complexity with other benefits

**Current Concern:**
- Best SBTG F1 = 0.356
- Random baseline F1 = 0.337 (best config)
- Improvement ratio = 1.01-1.16x (marginal)

**Action Required:**
- Implement baselines before drawing conclusions
- If baselines match SBTG, reconsider model complexity
- If SBTG >> baselines, highlight in manuscript

---

### Future Improvements

1. **Data:**
   - Longer recordings (100+ timepoints)
   - Higher sampling rate (10-20 Hz)
   - More stimuli (6-8 conditions)

2. **Models:**
   - Time-varying W (temporal phases)
   - Nonlinear coupling functions
   - Multivariate volatility models

3. **Evaluation:**
   - Path-based metrics (indirect connections)
   - Temporal correlation (phase-resolved)
   - Causal inference (intervention analysis)

---

## Troubleshooting

### Common Issues

**1. "No module named 'sbtg'"**
```bash
# Solution: Ensure sbtg/ directory in same level as pipeline/
cd CURRENT/
ls sbtg/sbtg_main.py  # Should exist
python -c "import sys; sys.path.insert(0, '.'); from sbtg import SBTGStructuredVolatilityEstimator"
```

**2. "File not found: data/Head_Activity_OH16230.mat"**
```bash
# Solution: Data files must be in data/ directory
cd CURRENT/
ls data/*.mat  # Should show MAT files
```

**3. "Connectome files not found"**
```bash
# Solution: Run 01_prepare_data.py first
python pipeline/01_prepare_data.py
ls results/intermediate/connectome/A_struct.npy  # Should exist
```

**4. "ValueError: Not enough timepoints"**
```
# Cause: Butanone has only 1 timepoint
# Solution: Expected behavior, skip butanone or use longer recordings
```

**5. "KeyError: 'mu_hat'"**
```bash
# Solution: Re-run 05_comprehensive_sbtg_analysis.py with updated version
# Old results don't have mu_hat saved
python pipeline/05_comprehensive_sbtg_analysis.py
```

**6. "CUDA out of memory"**
```python
# Solution: Reduce batch size
DEFAULT_DSM_BATCH_SIZE = 64  # Instead of 128
```

**7. "JSON serialization error (numpy bool)"**
```python
# Fixed in current versions
# If encountered, update to latest 07_weighted_connectome_analysis.py
```

### Performance Optimization

**Speed up training:**
```python
# Reduce epochs (may hurt accuracy)
dsm_epochs = 50  # Instead of 100

# Use GPU if available
import torch
assert torch.cuda.is_available()

# Smaller hidden dimensions
structured_hidden_dim = 32  # Instead of 64
```

**Reduce memory:**
```python
# Smaller batch size
dsm_batch_size = 64  # Instead of 128

# Process fewer neurons
neuron_subset = neuron_names[:50]  # Use first 50
```

**Parallel processing:**
```bash
# Run multiple stimuli in parallel (separate terminals)
python pipeline/05_comprehensive_sbtg_analysis.py --stimulus nacl &
python pipeline/05_comprehensive_sbtg_analysis.py --stimulus pentanedione &
# Note: Current version doesn't support --stimulus flag, modify manually
```

### Data Validation

**Check data integrity:**
```python
import numpy as np
from scipy.io import loadmat

# Load MAT file
mat = loadmat('data/Head_Activity_OH16230.mat')

# Check structure
print("Keys:", mat.keys())
print("Neurons:", len(mat['neurons']))
print("Stimuli:", [str(s[0]) for s in mat['stim_names']])

# Check for NaN
traces = mat['norm_traces']
for i, neuron_traces in enumerate(traces):
    for j, trace in enumerate(neuron_traces[0]):
        if np.any(np.isnan(trace)):
            print(f"NaN in neuron {i}, worm {j}")
```

**Verify connectome alignment:**
```python
import json
import numpy as np

# Load structural connectome
A_struct = np.load('results/intermediate/connectome/A_struct.npy')
with open('results/intermediate/connectome/nodes.json') as f:
    nodes = json.load(f)

print(f"Structural neurons: {len(nodes)}")
print(f"Edges: {(A_struct > 0).sum()}")
print(f"Density: {(A_struct > 0).sum() / (len(nodes) ** 2):.3f}")
print("Sample neurons:", nodes[:10])
```

### Debugging Tools

**Enable verbose logging:**
```python
# In SBTG estimator
estimator = SBTGStructuredVolatilityEstimator(
    model_type='linear',
    verbose=True  # Prints training progress
)
```

**Save intermediate results:**
```python
# After each major step
np.save('debug_sign_adj.npy', result.sign_adj)
print(f"Edges: {(result.sign_adj != 0).sum()}")
```

**Profile runtime:**
```python
import time
start = time.time()
result = estimator.fit(X)
print(f"Training took {time.time() - start:.1f}s")
```

---

## Quick Reference

### Essential Commands

**Run full pipeline:**
```bash
cd diffusionCircuit/
python pipeline/01_prepare_data.py
python pipeline/02_hyperparameter_search.py --quick
python pipeline/03_comprehensive_analysis.py
python pipeline/04_connectome_comparison.py
python pipeline/05_weighted_analysis.py
python pipeline/06_network_visualizations.py
python pipeline/07_generate_summary.py
```

**Or use the shell script:**
```bash
./run_pipeline.sh              # Full pipeline
./run_pipeline.sh quick        # Quick mode (reduced HP search)
./run_pipeline.sh eval         # Eval only (skip training)
```

**Check results:**
```bash
cat results/comprehensive_sbtg/summary/SUMMARY_REPORT.md
head results/comprehensive_sbtg/all_results.csv
```

**Load best model:**
```python
import numpy as np
import json

# Best configuration
stimulus = 'pentanedione'
config = 'feature_bilinear_per_stimulus_prefix'

# Load results
result = np.load(f'results/comprehensive_sbtg/models/{stimulus}/{config}/result.npz')
with open(f'results/comprehensive_sbtg/models/{stimulus}/{config}/neuron_names.json') as f:
    neurons = json.load(f)

sign_adj = result['sign_adj']
mu_hat = result['mu_hat']

print(f"Neurons: {len(neurons)}")
print(f"Edges: {(sign_adj != 0).sum()}")
print(f"Excitatory: {(sign_adj > 0).sum()}")
print(f"Inhibitory: {(sign_adj < 0).sum()}")
```

---

## Contact & Support

**Documentation:** This README and README.md  
**Code:** `pipeline/` and `sbtg/`  
**Results:** `results/comprehensive_sbtg/`  
**Issues:** Check Troubleshooting section above

**Last Updated:** December 25, 2025 (v2 with extended analyses)

---

## Changelog

### v2 (December 25, 2025)
- Added path-2 (indirect connection) analysis to script 06
- Added random baseline comparison
- Added permutation tests for statistical significance
- Added cross-stimulus consistency analysis
- Added overfitting analysis (random vs prefix splits) to script 10
- Added latent confounder detection (unmeasured intermediate neurons)
- Identified critical finding: only 1.16x improvement over random baseline
- Documented high-priority need for baseline method comparisons

### v1 (December 24, 2025)
- Initial production pipeline (scripts 01-10)
- All scripts tested and verified
- Best F1 = 0.356 (pentanedione/feature_bilinear/per_stimulus_prefix)
