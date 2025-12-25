# Functional Connectome Inference via Score-Based Temporal Graphical Models

**Project:** Inferring C. elegans functional connectivity from calcium imaging using SBTG models  
**Status:** Production pipeline complete, extended validation implemented  
**Last Updated:** December 25, 2025

---

## Overview

This repository implements a complete pipeline for inferring **functional connectivity** from C. elegans calcium imaging data using **Score-Based Temporal Graphical (SBTG) models**. The inferred networks are validated against the **structural connectome** from Cook et al. (2019).

### Key Results

| Metric | Best Value | Configuration |
|--------|------------|---------------|
| **F1 Score** | 0.337 | pentanedione / regime_gated / odd_even |
| **Recall** | 49.8% | pentanedione / regime_gated / odd_even |
| **Precision** | 26.6% | pentanedione / regime_gated / prefix |
| **Spearman ρ** | 0.108 | pentanedione / regime_gated / per_stimulus_prefix |

### Scientific Context

- **Input:** NeuroPAL calcium imaging (∆F/F0 traces)
- **Method:** SBTG with denoising score matching
- **Validation:** Binary edge matching + weight correlation vs. Cook et al. 2019 connectome
- **Overlap:** 51 neurons shared between functional (109) and structural (94) datasets

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test suite
python test_pipeline.py

# 4. Run production pipeline
./run_pipeline.sh              # Full pipeline with hyperparameter search
./run_pipeline.sh quick        # Quick mode (reduced HP search)
./run_pipeline.sh eval         # Eval only (skip training)

# Or run scripts individually:
python pipeline/01_prepare_data.py
python pipeline/02_hyperparameter_search.py --quick
python pipeline/03_comprehensive_analysis.py
python pipeline/04_connectome_comparison.py
python pipeline/05_weighted_analysis.py
python pipeline/06_network_visualizations.py
python pipeline/07_generate_summary.py

# 5. View results
cat results/comprehensive_sbtg/summary/SUMMARY_REPORT.md
```

---

## Pipeline Scripts (01-07)

| Script | Description | Status | Runtime |
|--------|-------------|--------|---------|
| `01_prepare_data.py` | Connectome alignment & lag-window dataset prep | ✅ Production | ~2 min |
| `02_hyperparameter_search.py` | **Grid search for optimal hyperparameters per dataset** | ✅ Production | ~1-4 hr |
| `03_comprehensive_analysis.py` | **Multi-config training (3 models × 4 splits × 2 stimuli)** | ✅ Production | ~15 min |
| `04_connectome_comparison.py` | Binary edge evaluation + extended analyses | ✅ Production | ~3 min |
| `05_weighted_analysis.py` | Weight correlation analysis | ✅ Production | ~2 min |
| `06_network_visualizations.py` | Network diagrams & degree distributions | ✅ Production | ~5 min |
| `07_generate_summary.py` | Comprehensive summary with **overfitting analysis** | ✅ Production | ~30 sec |

---

## Extended Analyses (New in v2)

Script 06 now includes advanced validation:

| Analysis | Description | Output |
|----------|-------------|--------|
| **Path-2 Connections** | Identifies indirect (2-hop) structural paths | `path2_comparisons.csv` |
| **Latent Confounder Detection** | Flags unmeasured intermediate neurons | 42.6% of path-2 via latent nodes |
| **Random Baseline** | Expected F1 if edges randomly assigned | `edge_comparisons.csv` (new columns) |
| **Permutation Tests** | Statistical significance (500 permutations) | `permutation_tests.csv` |
| **Cross-Stimulus Consistency** | Edge reproducibility across stimuli | `cross_stimulus_consistency.json` |
| **Overfitting Analysis** | Random vs prefix split comparison | In summary report |

---

## Critical Findings

### ⚠️ Concerns

1. **Marginal improvement over random baseline**
   - Only 41.7% of configurations beat random baseline
   - Best improvement: 1.16x (nacl/linear/odd_even)
   - Average F1/Random ratio: 0.95x

2. **Overfitting signal detected**
   - Random splits outperform prefix splits by 2.19x on average
   - Max ratio: 3.80x (nacl/linear)
   - Suggests temporal leakage in random splits

3. **Latent confounders**
   - 43 neurons unmeasured but in structural connectome
   - 1,117 path-2 connections go through unmeasured intermediaries
   - These appear as "false positives" but may be real influence

### ✅ Positive Findings

1. **Path-2 explains most "false positives"**
   - 76.8% of FPs are actually indirect (2-hop) connections
   - Suggests model captures multi-synaptic pathways

2. **Cross-stimulus consistency**
   - Best config: F1 = 0.356 ± 0.001 across stimuli
   - Feature-bilinear model most stable

3. **Permutation test significance**
   - nacl/linear/odd_even: p < 0.0001 (significant)
   - Some configurations show real signal

---

## Immediate Next Steps

### 🔴 HIGH PRIORITY: Implement Baseline Methods

The current evaluation lacks comparison against standard connectivity inference methods. **Before publication, we must implement:**

| Baseline Method | Description | Complexity | Implementation |
|-----------------|-------------|------------|----------------|
| **Pearson Correlation** | Simple pairwise correlation of time series | Easy | `np.corrcoef()` |
| **Granger Causality** | VAR-based temporal precedence test | Medium | `statsmodels.tsa.stattools.grangercausalitytests` |
| **Transfer Entropy** | Information-theoretic directional coupling | Medium | Custom or `pyinform` |
| **Partial Correlation** | Correlation controlling for other nodes | Easy | Inverse covariance |
| **GLASSO** | Sparse inverse covariance (undirected) | Medium | `sklearn.covariance.GraphicalLasso` |
| **Cross-Correlation (lag-1)** | Time-lagged pairwise correlation | Easy | `scipy.signal.correlate` |

**Implementation plan:**
1. Add `pipeline/XX_baseline_methods.py` or integrate into script 06
2. Compute each baseline's adjacency matrix
3. Threshold at same edge density as SBTG
4. Compare F1/precision/recall for each baseline
5. Report improvement ratio: `SBTG_F1 / Baseline_F1`

**Expected outcome:** SBTG should significantly outperform simple baselines (>1.5x F1) to justify model complexity.

### 🟡 MEDIUM PRIORITY

- [ ] Add edge directionality analysis (currently evaluating undirected)
- [ ] Implement neuron-type stratified evaluation (sensory → interneuron → motor)
- [ ] Test with different FDR thresholds (0.01, 0.05, 0.2)
- [ ] Add temporal resolution sensitivity analysis
- [ ] Run full permutation tests on all configurations (currently only first 4)

### 🟢 LOW PRIORITY

- [ ] Publication-quality figures
- [ ] Supplementary data tables
- [ ] Cross-validation with held-out worms
- [ ] Add butanone stimulus (currently only 1 timepoint, insufficient)

---

## Project Structure

```
diffusionCircuit/
├── pipeline/                    # Analysis scripts (01-07)
│   ├── 01_prepare_data.py      # Data loading & connectome alignment
│   ├── 02_hyperparameter_search.py  # Grid search for optimal hyperparams
│   ├── 03_comprehensive_analysis.py  # Multi-config SBTG training
│   ├── 04_connectome_comparison.py   # Extended validation vs structural
│   ├── 05_weighted_analysis.py       # Weight correlations
│   ├── 06_network_visualizations.py  # Network plots
│   ├── 07_generate_summary.py        # Final report
│   └── archive/                # Old/superseded scripts (gitignored)
├── sbtg/                        # SBTG implementation
│   ├── sbtg.py                 # Core model classes
│   └── __init__.py
├── data/                        # Input data (not tracked in git)
│   ├── Head_Activity_OH15500.mat
│   ├── Head_Activity_OH16230.mat
│   └── *.xlsx                  # Cook connectome
├── results/                     # Output (not tracked in git)
│   ├── hyperparameter_search/  # Script 02 outputs
│   │   ├── search_results.csv
│   │   ├── best_hyperparameters.json  # Used by script 03
│   │   └── loss_curves/        # Training loss per config
│   ├── comprehensive_sbtg/     # Main results
│   │   ├── all_results.csv
│   │   ├── models/             # Trained models per config
│   │   ├── connectome_comparison/
│   │   │   ├── edge_comparisons.csv
│   │   │   ├── path2_comparisons.csv      # Indirect connections
│   │   │   ├── permutation_tests.csv      # Statistical significance
│   │   │   └── cross_stimulus_consistency.json
│   │   ├── weighted_analysis/
│   │   └── summary/
│   │       └── SUMMARY_REPORT.md
│   └── intermediate/
│       ├── connectome/         # Aligned structural matrices
│       └── datasets/           # Lag-window datasets
├── run_pipeline.sh              # One-command pipeline execution
├── test_pipeline.py             # Test suite
├── requirements.txt
├── .gitignore
├── README.md                    # This file
└── README_PIPELINE.md           # Detailed technical documentation
```

---

## Data Requirements

| File | Description | Size |
|------|-------------|------|
| `Head_Activity_OH15500.mat` | NaCl stimulus recordings | ~50 MB |
| `Head_Activity_OH16230.mat` | Pentanedione/Butanone recordings | ~50 MB |
| `SI 6 Cell class lists.xlsx` | Neuron metadata | ~100 KB |
| `SI 7 Cell class connectome....xlsx` | Structural adjacency matrices | ~500 KB |

**Note:** Data files are not included in the repository. Place them in the `data/` directory.

---

## Dataset Details

### NeuroPAL Calcium Imaging Data

The calcium imaging data comes from NeuroPAL transgenic C. elegans, which enables whole-brain neuronal identification via multicolor fluorescence (Yemini et al., Cell 2021).

**Source Files:**
| MAT File | Strain | Stimuli | Worms | Neurons |
|----------|--------|---------|-------|---------|
| `Head_Activity_OH15500.mat` | OH15500 | butanone, pentanedione, nacl | 7 | 108 |
| `Head_Activity_OH16230.mat` | OH16230 | butanone, pentanedione, nacl | 21 | 109 |

**Recording Parameters:**
- **Frame rate:** 4 Hz (250 ms temporal resolution)
- **Stimulus windows:** ~10 seconds (40 frames per worm per stimulus)
- **Traces:** ∆F/F₀ normalized fluorescence

**MAT File Structure:**
```
neurons          (n,) cell array      Neuron names (e.g., 'AVAL', 'AVBR')
norm_traces      (n,) cell array      ∆F/F₀ traces per neuron
stim_names       (3,) cell array      ['butanone', 'pentanedione', 'nacl']
stim_times       (3,2) array          [start_sec, end_sec] per stimulus
stims            (worms,) cell array  Stimulus order per worm (1-indexed)
fps              scalar               Frame rate (4 Hz)
files            (worms,) cell array  Worm IDs
```

### Prepared Datasets

After running `01_prepare_data.py`, standardized datasets are created:

```
results/intermediate/datasets/{stimulus}/
├── X_segments.npy     # (n_worms, n_timepoints, n_neurons) individual worm recordings
├── Z_std.npy          # (n_windows, 2×n_neurons) standardized lag windows
├── Z_raw.npy          # (n_windows, 2×n_neurons) raw lag windows  
├── standardization.json  # Mean/std per neuron, node order
└── segments.csv       # Worm metadata (indices, frame ranges)
```

**Per-Stimulus Dataset Dimensions:**
| Stimulus | Worms | Timepoints | Neurons | Lag Windows |
|----------|-------|------------|---------|-------------|
| NaCl | 12 | 40 | 51 | 468 |
| Pentanedione | 12 | 40 | 51 | 468 |
| Butanone | 12 | 40 | 51 | 468 |

**Key Properties:**
- **51 neurons:** Intersection of NeuroPAL recordings with Cook et al. structural connectome
- **468 windows:** 12 worms × 39 windows/worm (T-1 lag windows per worm)
- **No cross-worm transitions:** Lag windows built within each worm separately
- **Standardized:** Zero mean, unit variance per neuron

### Structural Connectome (Cook et al. 2019)

Ground truth from electron microscopy reconstruction of the complete C. elegans connectome.

**Source:** Cook, S.J. et al. (2019). Nature, 571(7763), 63-71.

**Aligned Connectome:**
```
results/intermediate/connectome/
├── A_struct.npy    # Combined adjacency (chemical + 0.5×gap junctions)
├── A_chem.npy      # Chemical synapse counts
├── A_gap.npy       # Gap junction counts
└── nodes.json      # 51 aligned neuron names
```

**Properties:**
- **51 neurons:** Overlap with functional data
- **808 edges:** Direct structural connections (edge if synapse count > 0)
- **Density:** 31.1%

---

## Dependencies

```
numpy>=1.21
scipy>=1.7
pandas>=1.3
matplotlib>=3.4
seaborn>=0.11
networkx>=2.6
torch>=2.0
scikit-learn>=1.0
statsmodels>=0.13
openpyxl>=3.0
h5py>=3.0
```

Install with: `pip install -r requirements.txt`

---

## SBTG Model Details

### Model Types

| Model | Description | Parameters | Best Use Case |
|-------|-------------|------------|---------------|
| **Linear** | Cross-lag linear: $x_{t+1} = Wx_t + \epsilon$ | ~n² | Fast baseline |
| **Feature-Bilinear** | Nonlinear feature transforms + bilinear coupling | ~2n² | Best F1 (0.356) |
| **Regime-Gated** | State-dependent connectivity switching | ~3n² | Best weight correlation |

### Train/Test Splits

| Split | Description | Leakage Risk | Recommendation |
|-------|-------------|--------------|----------------|
| **prefix** | First 70% train, last 30% test | Low | ✅ Use for final evaluation |
| **per_stimulus_prefix** | Per-stimulus temporal split | Lowest | ✅ Most rigorous |
| **random** | Random 70/30 split | ⚠️ High | ❌ Avoid (shows overfitting) |
| **odd_even** | Alternating timepoints | Medium | ⚠️ Use with caution |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fdr_alpha` | 0.1 | False discovery rate threshold |
| `dsm_epochs` | 100 | Denoising score matching epochs |
| `dsm_noise_std` | 0.1 | Noise level for DSM training |
| `structured_l1_lambda` | 0.0 | L1 regularization on W |
| `hac_max_lag` | 5 | HAC variance estimation lags |

---

## Validation Metrics

### Binary Edge Detection

- **Precision:** TP / (TP + FP) — fraction of predicted edges that are correct
- **Recall:** TP / (TP + FN) — fraction of true edges that are detected
- **F1 Score:** 2 × Precision × Recall / (Precision + Recall)

Ground truth: Binary structural connectome (edge exists if synapse count > 0)

### Weighted Correlation

- **Spearman ρ:** Rank correlation between functional weights (μ̂) and structural weights
- **Pearson r:** Linear correlation (sensitive to outliers)

Note: Weight correlation is weak (ρ ≈ 0.01) due to fundamental differences between functional and anatomical connectivity.

### Extended Metrics (v2)

- **Path-2 F1:** F1 when counting 2-hop paths as true positives
- **F1 vs Random Ratio:** Improvement over random edge predictor
- **Permutation p-value:** Statistical significance of F1

---

## Citation

If you use this code, please cite:

```bibtex
@article{sbtg2025,
  title={Score-Based Temporal Graphical Models for C. elegans Functional Connectome Inference},
  author={...},
  journal={...},
  year={2025}
}
```

**Structural connectome reference:**
> Cook, S.J. et al. (2019). Whole-animal connectomes of both Caenorhabditis elegans sexes. Nature, 571(7763), 63-71.

---

## See Also

- [README_PIPELINE.md](README_PIPELINE.md) — Detailed technical documentation for each script
- [results/comprehensive_sbtg/summary/SUMMARY_REPORT.md](results/comprehensive_sbtg/summary/SUMMARY_REPORT.md) — Latest results

---

## License

[Add license information]

## Contact

[Add contact information]
