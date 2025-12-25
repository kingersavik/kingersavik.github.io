#!/bin/bash
# =============================================================================
# SBTG Pipeline - Complete Execution Script
# =============================================================================
# 
# This script runs the complete SBTG functional connectome inference pipeline.
# 
# Prerequisites:
#   - Python 3.10+ with venv
#   - Data files in data/ directory:
#       - Head_Activity_OH15500.mat
#       - Head_Activity_OH16230.mat
#       - SI 6 Cell class lists.xlsx
#       - SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx
#
# Usage:
#   ./run_pipeline.sh           # Run full pipeline (with hyperparameter search)
#   ./run_pipeline.sh quick     # Skip hyperparameter search, use existing models
#   ./run_pipeline.sh eval      # Skip training entirely, only run evaluation
#   ./run_pipeline.sh test      # Run test suite only
#
# Output:
#   - results/comprehensive_sbtg/summary/SUMMARY_REPORT.md
#
# Pipeline Order:
#   01 → 02 → 03 → 04 → 05 → 06 → 07
#   Data  HP    Train  Compare  Weights  Viz  Summary
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}SBTG Pipeline - Functional Connectome Inference${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# =============================================================================
# SETUP: Activate virtual environment
# =============================================================================

echo -e "${YELLOW}[SETUP] Activating virtual environment...${NC}"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo -e "${GREEN}✓ Virtual environment active${NC}"
echo ""

# =============================================================================
# TEST MODE: Just run tests and exit
# =============================================================================

if [ "$1" == "test" ]; then
    echo -e "${YELLOW}[TEST] Running test suite...${NC}"
    python test_pipeline.py
    exit $?
fi

# =============================================================================
# SCRIPT 01: Data Preparation & Connectome Alignment
# =============================================================================
# 
# Purpose:
#   - Load NeuroPAL calcium imaging data from MAT files
#   - Load Cook et al. 2019 structural connectome from Excel
#   - Align neuron names between datasets (109 functional, 94 structural → 57 overlap)
#   - Create lag-window datasets for SBTG training
#   - Save intermediate files for downstream analysis
#
# Outputs:
#   - results/intermediate/connectome/A_struct.npy (57×57 structural adjacency)
#   - results/intermediate/connectome/nodes.json (neuron names)
#   - results/intermediate/datasets/{stimulus}/Z_std.npy (standardized data)
#   - results/figures/connectome/*.png (QC visualizations)
#
# Expected runtime: ~2 minutes
# =============================================================================

echo -e "${YELLOW}[01/07] Preparing data and aligning connectome...${NC}"
echo "        Loading NeuroPAL MAT files and Cook connectome Excel files"
echo "        Aligning 109 functional neurons with 94 structural neurons"
echo ""

python pipeline/01_prepare_data.py

echo -e "${GREEN}✓ Script 01 complete${NC}"
echo ""

# =============================================================================
# EVAL MODE: Skip all training, just run evaluation on existing models
# =============================================================================

if [ "$1" == "eval" ]; then
    echo -e "${YELLOW}[EVAL MODE] Skipping training scripts (02, 03)${NC}"
    echo "             Using existing models in results/comprehensive_sbtg/models/"
    echo ""
    
    if [ ! -d "results/comprehensive_sbtg/models" ]; then
        echo -e "${RED}ERROR: No existing models found. Run full pipeline first.${NC}"
        exit 1
    fi
    
    # Jump to evaluation scripts
    echo -e "${YELLOW}[04/07] Comparing functional vs structural connectivity...${NC}"
    python pipeline/04_connectome_comparison.py
    echo -e "${GREEN}✓ Script 04 complete${NC}"
    echo ""
    
    echo -e "${YELLOW}[05/07] Analyzing weighted connectivity correlations...${NC}"
    python pipeline/05_weighted_analysis.py
    echo -e "${GREEN}✓ Script 05 complete${NC}"
    echo ""
    
    echo -e "${YELLOW}[06/07] Generating network visualizations...${NC}"
    python pipeline/06_network_visualizations.py
    echo -e "${GREEN}✓ Script 06 complete${NC}"
    echo ""
    
    echo -e "${YELLOW}[07/07] Generating summary report...${NC}"
    python pipeline/07_generate_summary.py
    echo -e "${GREEN}✓ Script 07 complete${NC}"
    echo ""
    
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${GREEN}PIPELINE COMPLETE (eval mode)!${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    exit 0
fi

# =============================================================================
# SCRIPT 02: Hyperparameter Grid Search (NEW)
# =============================================================================
#
# Purpose:
#   - Find optimal hyperparameters for each dataset BEFORE training
#   - Grid search over: learning rate, epochs, hidden dim, L1 lambda, etc.
#   - Track training loss curves (cached to avoid re-running)
#   - Output best hyperparameters per dataset for script 03
#
# Outputs:
#   - results/hyperparameter_search/search_results.csv (all configurations)
#   - results/hyperparameter_search/best_hyperparameters.json (per-dataset best)
#   - results/hyperparameter_search/loss_curves/*.json (training loss curves)
#   - results/hyperparameter_search/figures/*.png (visualizations)
#
# Expected runtime: ~1-4 hours (full), ~15-30 min (quick)
# =============================================================================

if [ "$1" == "quick" ]; then
    echo -e "${YELLOW}[02/07] Running QUICK hyperparameter search...${NC}"
    echo "        Using reduced grid for faster execution"
    echo ""
    python pipeline/02_hyperparameter_search.py --quick
else
    echo -e "${YELLOW}[02/07] Running hyperparameter grid search...${NC}"
    echo "        Optimizing training parameters for each dataset"
    echo "        This may take 1-4 hours..."
    echo ""
    python pipeline/02_hyperparameter_search.py
fi

echo -e "${GREEN}✓ Script 02 complete${NC}"
echo ""

# =============================================================================
# SCRIPT 03: Comprehensive SBTG Analysis (PRODUCTION)
# =============================================================================
#
# Purpose:
#   - Train ALL model configurations systematically:
#       - 3 model types: linear, feature_bilinear, regime_gated
#       - 4 train/test splits: prefix, per_stimulus_prefix, random, odd_even
#       - 2 stimuli: nacl, pentanedione (butanone skipped - only 1 timepoint)
#   - Uses optimized hyperparameters from script 02
#   - Total: 24 configurations trained
#   - Saves all results with standardized format
#
# Outputs:
#   - results/comprehensive_sbtg/all_results.csv (24 rows)
#   - results/comprehensive_sbtg/models/{stim}/{config}/result.npz
#   - results/comprehensive_sbtg/models/{stim}/{config}/neuron_names.json
#
# Expected runtime: ~15-25 minutes (CPU)
# =============================================================================

echo -e "${YELLOW}[03/07] Training comprehensive SBTG models...${NC}"
echo "        Training 3 model types × 4 splits × 2 stimuli = 24 configurations"
echo "        Using optimized hyperparameters from script 02"
echo "        This may take 15-25 minutes on CPU..."
echo ""

python pipeline/03_comprehensive_analysis.py

echo -e "${GREEN}✓ Script 03 complete${NC}"
echo ""

# =============================================================================
# SCRIPT 04: Connectome Comparison (Extended v2)
# =============================================================================
#
# Purpose:
#   - Compare functional edges (SBTG predictions) vs structural edges (Cook)
#   - Compute precision, recall, F1 score for binary edge detection
#   - Path-2 (indirect connection) analysis
#   - Random baseline comparison
#   - Permutation tests for statistical significance
#   - Cross-stimulus consistency analysis
#
# Key Metrics:
#   - Best F1: ~0.356 (pentanedione/feature_bilinear/per_stimulus_prefix)
#   - Best Recall: ~61.7%
#   - Path-2 explains 76.8% of "false positives"
#
# Outputs:
#   - results/comprehensive_sbtg/connectome_comparison/edge_comparisons.csv
#   - results/comprehensive_sbtg/connectome_comparison/path2_comparisons.csv
#   - results/comprehensive_sbtg/connectome_comparison/permutation_tests.csv
#   - results/comprehensive_sbtg/connectome_comparison/cross_stimulus_consistency.json
#   - results/comprehensive_sbtg/connectome_comparison/reports/extended_analysis.md
#
# Expected runtime: ~3-5 minutes
# =============================================================================

echo -e "${YELLOW}[04/07] Comparing functional vs structural connectivity...${NC}"
echo "        Computing precision/recall/F1 for edge detection"
echo "        Running path-2 analysis, permutation tests, cross-stimulus consistency"
echo ""

python pipeline/04_connectome_comparison.py

echo -e "${GREEN}✓ Script 04 complete${NC}"
echo ""

# =============================================================================
# SCRIPT 05: Weighted Connectome Analysis
# =============================================================================
#
# Purpose:
#   - Analyze correlation between functional coupling weights (mu_hat)
#     and structural synapse counts
#   - Compute Spearman and Pearson correlations on shared edges
#
# Key Finding:
#   - Weak correlation (ρ ≈ 0.01) - functional weight ≠ anatomical synapse count
#   - Best Spearman ρ: ~0.095 (pentanedione/regime_gated/prefix)
#
# Outputs:
#   - results/comprehensive_sbtg/weighted_analysis/weight_correlations.csv
#   - results/comprehensive_sbtg/weighted_analysis/weighted_edge_details.json
#   - results/comprehensive_sbtg/weighted_analysis/figures/*.png
#
# Expected runtime: ~2-3 minutes
# =============================================================================

echo -e "${YELLOW}[05/07] Analyzing weighted connectivity correlations...${NC}"
echo "        Comparing functional coupling weights vs structural synapse counts"
echo ""

python pipeline/05_weighted_analysis.py

echo -e "${GREEN}✓ Script 05 complete${NC}"
echo ""

# =============================================================================
# SCRIPT 06: Network Visualizations
# =============================================================================
#
# Purpose:
#   - Generate publication-quality network diagrams
#   - Create degree distribution plots
#   - Visualize edge properties (excitatory/inhibitory)
#
# Outputs:
#   - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/network_graph.png
#   - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/degree_distributions.png
#   - results/comprehensive_sbtg/visualizations/figures/{stim}/{config}/edge_properties.png
#
# Expected runtime: ~5 minutes
# =============================================================================

echo -e "${YELLOW}[06/07] Generating network visualizations...${NC}"
echo "        Creating network diagrams, degree distributions, edge property plots"
echo ""

python pipeline/06_network_visualizations.py

echo -e "${GREEN}✓ Script 06 complete${NC}"
echo ""

# =============================================================================
# SCRIPT 07: Generate Summary Report
# =============================================================================
#
# Purpose:
#   - Consolidate all results into comprehensive summary
#   - Generate final statistics and best configurations
#   - Overfitting analysis (random vs prefix splits)
#   - Random baseline comparison summary
#   - Path-2 and cross-stimulus consistency summaries
#
# Key Outputs:
#   - SUMMARY_REPORT.md with:
#       - Best configurations table
#       - ⚠️ Overfitting warnings
#       - Random baseline comparison
#       - Path-2 analysis summary
#       - Cross-stimulus consistency
#
# Outputs:
#   - results/comprehensive_sbtg/summary/SUMMARY_REPORT.md (READ THIS!)
#   - results/comprehensive_sbtg/summary/summary_statistics.csv
#   - results/comprehensive_sbtg/summary/best_configurations.csv
#   - results/comprehensive_sbtg/summary/model_comparison.png
#
# Expected runtime: ~30 seconds
# =============================================================================

echo -e "${YELLOW}[07/07] Generating summary report...${NC}"
echo "        Consolidating all results, computing final statistics"
echo "        Including overfitting analysis and baseline comparisons"
echo ""

python pipeline/07_generate_summary.py

echo -e "${GREEN}✓ Script 07 complete${NC}"
echo ""

# =============================================================================
# COMPLETION
# =============================================================================

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETE!${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "Key Results:"
echo -e "  📊 Summary Report: ${YELLOW}results/comprehensive_sbtg/summary/SUMMARY_REPORT.md${NC}"
echo -e "  📈 All Results:    results/comprehensive_sbtg/all_results.csv"
echo -e "  🔬 Extended Analysis: results/comprehensive_sbtg/connectome_comparison/reports/extended_analysis.md"
echo -e "  ⚙️  Best Hyperparams: results/hyperparameter_search/best_hyperparameters.json"
echo ""
echo -e "Best Configuration:"
echo -e "  Model:     feature_bilinear"
echo -e "  Split:     per_stimulus_prefix"
echo -e "  Stimulus:  pentanedione"
echo -e "  F1 Score:  0.356"
echo -e "  Recall:    61.7%"
echo ""
echo -e "${YELLOW}⚠️  Important Caveats:${NC}"
echo -e "  - Only 41.7% of configs beat random baseline"
echo -e "  - Overfitting detected: random/prefix ratio = 2.19x"
echo -e "  - 76.8% of 'false positives' are path-2 (indirect) connections"
echo -e "  - TODO: Implement baseline methods (Granger, correlation, etc.)"
echo ""
echo -e "To view results:"
echo -e "  cat results/comprehensive_sbtg/summary/SUMMARY_REPORT.md"
echo ""
