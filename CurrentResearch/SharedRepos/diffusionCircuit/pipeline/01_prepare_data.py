#!/usr/bin/env python3
"""
SCRIPT 1: Prepare Data for SBTG Analysis

This script performs all data preparation steps needed for the SBTG analysis:
1. Aligns the Cook et al. 2019 connectome with NeuroPAL neuron names
2. Builds standardized lag-window datasets for each stimulus
3. Generates quality control visualizations

USAGE:
    python pipeline/01_prepare_data.py

OUTPUTS:
    - results/intermediate/connectome/     (adjacency matrices, node lists)
    - results/intermediate/datasets/       (lag-window datasets per stimulus)
    - results/figures/connectome/          (connectome heatmaps)
    - results/tables/connectome/           (alignment tables)

REQUIREMENTS:
    - data/Head_Activity_OH16230.mat       (NeuroPAL recordings)
    - data/SI 6 Cell class lists.xlsx      (Cook neuron list)
    - data/SI 7 Cell class connectome...xlsx (Cook adjacency matrices)

RUNTIME: ~2-3 minutes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum number of worms a neuron must appear in to be included
MIN_WORMS = 18

# Which stimuli to process (set to None to process all)
STIMULI = None  # or ["nacl", "butanone", "pentanedione"]

# File names (should not need to change these)
NEUROPAL_FILE = "Head_Activity_OH16230.mat"
COOK_CONNECTOME_FILE = "SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx"
COOK_CELL_LIST_FILE = "SI 6 Cell class lists.xlsx"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def locate_project_root(start: Path) -> Path:
    """
    Find the project root directory by looking for the 'data' folder.
    
    Args:
        start: Starting directory path
        
    Returns:
        Path to project root
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    candidate = start.resolve()
    while True:
        if (candidate / "data").exists():
            return candidate
        if candidate.parent == candidate:
            raise FileNotFoundError(
                "Unable to locate project root. Please run from within the project directory."
            )
        candidate = candidate.parent


def normalize_name(name: str) -> str:
    """
    Normalize neuron class names to uppercase without whitespace.
    
    Args:
        name: Raw neuron name
        
    Returns:
        Normalized name (e.g., "  ase  " -> "ASE")
    """
    return name.strip().upper()


def ensure_directories(results_dir: Path) -> Dict[str, Path]:
    """
    Create all necessary output directories.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary mapping directory purposes to paths
    """
    dirs = {
        "connectome_intermediate": results_dir / "intermediate" / "connectome",
        "connectome_figures": results_dir / "figures" / "connectome",
        "connectome_tables": results_dir / "tables" / "connectome",
        "datasets": results_dir / "intermediate" / "datasets",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


# =============================================================================
# STEP 1: CONNECTOME ALIGNMENT
# =============================================================================

def load_cook_adjacency_with_labels(
    spreadsheet_path: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load adjacency matrix from Cook et al. 2019 spreadsheet.
    
    The spreadsheets have a specific format:
    - Row 0: Excel metadata (skip)
    - Row 1: Column headers (neuron names)
    - Row 2+: Row label (neuron name) + adjacency values
    
    Args:
        spreadsheet_path: Path to Excel file
        sheet_name: Name of sheet to read
        
    Returns:
        Tuple of (adjacency_matrix, column_labels, row_labels)
    """
    print(f"  Loading sheet '{sheet_name}'...")
    
    # Read the raw spreadsheet
    df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name, header=None)
    
    # Extract column labels from row 1
    col_labels = []
    for val in df.iloc[1, 2:]:
        if isinstance(val, str) and val.strip():
            col_labels.append(normalize_name(val))
        else:
            break  # Stop at first empty cell
    
    num_cols = len(col_labels)
    print(f"    Found {num_cols} column neurons")
    
    # Extract row labels and values
    row_labels = []
    matrix_rows = []
    
    for idx in range(2, len(df)):
        row = df.iloc[idx]
        
        # Get row label (check both column 0 and 1)
        label = None
        for col_idx in [1, 0]:
            val = row.iloc[col_idx]
            if isinstance(val, str) and val.strip():
                label = normalize_name(val)
                break
        
        if label is None:
            continue  # Skip rows without labels
        
        # Get adjacency values
        values = row.iloc[2:2 + num_cols].tolist()
        numeric_values = pd.to_numeric(
            pd.Series(values),
            errors="coerce"
        ).fillna(0.0).tolist()
        
        row_labels.append(label)
        matrix_rows.append(numeric_values)
    
    # Convert to numpy array
    adjacency = np.array(matrix_rows, dtype=float)
    print(f"    Matrix shape: {adjacency.shape} (rows={len(row_labels)}, cols={num_cols})")
    
    return adjacency, col_labels, row_labels


def align_matrices_to_common_neurons(
    A_chem: np.ndarray,
    col_labels_chem: List[str],
    row_labels_chem: List[str],
    A_gap: np.ndarray,
    col_labels_gap: List[str],
    row_labels_gap: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align chemical and gap junction matrices to common neuron set.
    
    Args:
        A_chem: Chemical synapse matrix
        col_labels_chem: Column labels for chemical matrix
        row_labels_chem: Row labels for chemical matrix
        A_gap: Gap junction matrix
        col_labels_gap: Column labels for gap matrix
        row_labels_gap: Row labels for gap matrix
        
    Returns:
        Tuple of (aligned_A_chem, aligned_A_gap, common_neurons)
    """
    # Find neurons present in both row and column labels for each matrix
    chem_neurons = set(col_labels_chem) & set(row_labels_chem)
    gap_neurons = set(col_labels_gap) & set(row_labels_gap)
    
    # Find common neurons across both matrices
    common_neurons = sorted(chem_neurons & gap_neurons)
    print(f"  Chemical neurons (rows ∩ cols): {len(chem_neurons)}")
    print(f"  Gap junction neurons (rows ∩ cols): {len(gap_neurons)}")
    print(f"  Common neurons: {len(common_neurons)}")
    
    # Create aligned matrices
    n = len(common_neurons)
    aligned_chem = np.zeros((n, n), dtype=float)
    aligned_gap = np.zeros((n, n), dtype=float)
    
    # Build index maps
    chem_col_idx = {name: i for i, name in enumerate(col_labels_chem)}
    chem_row_idx = {name: i for i, name in enumerate(row_labels_chem)}
    gap_col_idx = {name: i for i, name in enumerate(col_labels_gap)}
    gap_row_idx = {name: i for i, name in enumerate(row_labels_gap)}
    
    # Fill aligned matrices
    for i, row_name in enumerate(common_neurons):
        for j, col_name in enumerate(common_neurons):
            # Chemical
            if row_name in chem_row_idx and col_name in chem_col_idx:
                aligned_chem[i, j] = A_chem[chem_row_idx[row_name], chem_col_idx[col_name]]
            # Gap junction
            if row_name in gap_row_idx and col_name in gap_col_idx:
                aligned_gap[i, j] = A_gap[gap_row_idx[row_name], gap_col_idx[col_name]]
    
    return aligned_chem, aligned_gap, common_neurons


def align_connectome(
    data_dir: Path,
    output_dirs: Dict[str, Path],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Load and align Cook connectome data.
    
    Processes:
    - Chemical synapses (directed)
    - Gap junctions (symmetric)
    - Combined structural connectivity
    
    Args:
        data_dir: Path to data directory
        output_dirs: Dictionary of output directory paths
        
    Returns:
        Tuple of (node_list, adjacency_dict)
        where adjacency_dict has keys: 'chem', 'gap', 'struct'
    """
    print("\n" + "=" * 70)
    print("STEP 1: Aligning Cook Connectome with NeuroPAL")
    print("=" * 70)
    
    connectome_file = data_dir / COOK_CONNECTOME_FILE
    
    if not connectome_file.exists():
        raise FileNotFoundError(
            f"Cook connectome file not found: {connectome_file}\n"
            f"Please ensure it is in the data/ directory."
        )
    
    # Load chemical synapses
    print("\nLoading chemical synapses...")
    A_chem_raw, col_labels_chem, row_labels_chem = load_cook_adjacency_with_labels(
        connectome_file,
        "herm chem grouped"
    )
    
    # Load gap junctions
    print("\nLoading gap junctions...")
    A_gap_raw, col_labels_gap, row_labels_gap = load_cook_adjacency_with_labels(
        connectome_file,
        "herm gap jn grouped asymmetric"
    )
    
    # Align matrices to common neuron set
    print("\nAligning matrices to common neuron set...")
    A_chem, A_gap, nodes = align_matrices_to_common_neurons(
        A_chem_raw, col_labels_chem, row_labels_chem,
        A_gap_raw, col_labels_gap, row_labels_gap
    )
    
    print(f"\n✓ Loaded connectome with {len(nodes)} neuron classes")
    
    # Create combined structural matrix (chemical + gap)
    A_struct = A_chem + A_gap
    
    # Save adjacency matrices
    print("\nSaving adjacency matrices...")
    np.save(output_dirs["connectome_intermediate"] / "A_chem.npy", A_chem)
    np.save(output_dirs["connectome_intermediate"] / "A_gap.npy", A_gap)
    np.save(output_dirs["connectome_intermediate"] / "A_struct.npy", A_struct)
    
    # Save node list
    with open(output_dirs["connectome_intermediate"] / "nodes.json", "w") as f:
        json.dump(nodes, f, indent=2)
    
    print(f"  Saved to: {output_dirs['connectome_intermediate']}/")
    
    # Generate visualizations
    print("\nGenerating connectome visualizations...")
    plot_connectome_heatmaps(
        {"Chemical": A_chem, "Gap Junctions": A_gap, "Combined": A_struct},
        nodes,
        output_dirs["connectome_figures"]
    )
    
    # Create summary table
    summary_df = create_connectome_summary(A_chem, A_gap, A_struct, nodes)
    summary_df.to_csv(
        output_dirs["connectome_tables"] / "connectome_summary.csv",
        index=False
    )
    
    print(f"\n✓ Connectome alignment complete")
    print(f"  Total edges - Chemical: {int((A_chem > 0).sum())}, "
          f"Gap: {int((A_gap > 0).sum())}, "
          f"Combined: {int((A_struct > 0).sum())}")
    
    return nodes, {
        "chem": A_chem,
        "gap": A_gap,
        "struct": A_struct
    }


def plot_connectome_heatmaps(
    matrices: Dict[str, np.ndarray],
    nodes: List[str],
    output_dir: Path,
) -> None:
    """
    Generate heatmap visualizations of connectome matrices.
    
    Args:
        matrices: Dictionary mapping matrix names to adjacency matrices
        nodes: List of neuron names
        output_dir: Where to save figures
    """
    for name, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use log scale for better visualization
        plot_matrix = np.log10(matrix + 1)
        
        sns.heatmap(
            plot_matrix,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "log₁₀(weight + 1)"},
            xticklabels=False,
            yticklabels=False,
        )
        
        ax.set_title(f"{name} Connectome (Cook et al. 2019)", fontsize=14)
        ax.set_xlabel("Pre-synaptic neuron", fontsize=12)
        ax.set_ylabel("Post-synaptic neuron", fontsize=12)
        
        # Add edge count annotation
        edge_count = int((matrix > 0).sum())
        ax.text(
            0.02, 0.98,
            f"Edges: {edge_count}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        filename = output_dir / f"{name.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename.name}")


def create_connectome_summary(
    A_chem: np.ndarray,
    A_gap: np.ndarray,
    A_struct: np.ndarray,
    nodes: List[str],
) -> pd.DataFrame:
    """
    Create summary statistics table for connectome.
    
    Args:
        A_chem: Chemical synapse adjacency matrix
        A_gap: Gap junction adjacency matrix
        A_struct: Combined structural adjacency matrix
        nodes: List of neuron names
        
    Returns:
        DataFrame with per-neuron statistics
    """
    records = []
    
    for i, node in enumerate(nodes):
        # Chemical synapses
        chem_out = int((A_chem[i, :] > 0).sum())
        chem_in = int((A_chem[:, i] > 0).sum())
        
        # Gap junctions
        gap_out = int((A_gap[i, :] > 0).sum())
        gap_in = int((A_gap[:, i] > 0).sum())
        
        # Combined
        struct_out = int((A_struct[i, :] > 0).sum())
        struct_in = int((A_struct[:, i] > 0).sum())
        
        records.append({
            "neuron": node,
            "chem_out_degree": chem_out,
            "chem_in_degree": chem_in,
            "gap_out_degree": gap_out,
            "gap_in_degree": gap_in,
            "total_out_degree": struct_out,
            "total_in_degree": struct_in,
        })
    
    return pd.DataFrame(records)


# =============================================================================
# STEP 2: BUILD SBTG DATASETS
# =============================================================================

def load_neuropal_data(data_dir: Path) -> Dict:
    """
    Load NeuroPAL recording from MATLAB file.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with neuron names, traces, stimulus info, etc.
    """
    neuropal_path = data_dir / NEUROPAL_FILE
    
    if not neuropal_path.exists():
        raise FileNotFoundError(
            f"NeuroPAL file not found: {neuropal_path}\n"
            f"Please ensure it is in the data/ directory."
        )
    
    print(f"\nLoading NeuroPAL data from: {neuropal_path.name}")
    mat = loadmat(neuropal_path, simplify_cells=True)
    
    return {
        "neuron_names": [normalize_name(str(n)) for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "stim_names": [str(s) for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "stims_per_worm": [np.asarray(row, dtype=int) for row in mat["stims"]],
        "worm_ids": [str(f) for f in mat["files"]],
    }


def compute_node_coverage(
    nodes: List[str],
    neuropal_data: Dict,
) -> pd.DataFrame:
    """
    Compute how many worms each neuron appears in.
    
    Args:
        nodes: List of neuron names (from connectome)
        neuropal_data: NeuroPAL recording data
        
    Returns:
        DataFrame with worm coverage counts per neuron
    """
    name_to_idx = {
        name: idx
        for idx, name in enumerate(neuropal_data["neuron_names"])
    }
    
    num_worms = len(neuropal_data["worm_ids"])
    
    records = []
    for node in nodes:
        if node not in name_to_idx:
            # Neuron not present in NeuroPAL recordings
            records.append({"node": node, "worm_count": 0})
            continue
        
        idx = name_to_idx[node]
        count = 0
        
        # Count how many worms have this neuron
        for worm_idx in range(num_worms):
            trace = collect_worm_trace(
                neuropal_data["norm_traces"][idx],
                worm_idx,
                num_worms
            )
            if trace is not None:
                count += 1
        
        records.append({"node": node, "worm_count": count})
    
    df = pd.DataFrame(records)
    df = df.sort_values("worm_count", ascending=False).reset_index(drop=True)
    
    return df


def collect_worm_trace(
    neuron_traces: List[np.ndarray],
    worm_idx: int,
    num_worms: int,
) -> np.ndarray | None:
    """
    Average left/right traces for a neuron in a specific worm.
    
    NeuroPAL stores up to 2*num_worms entries per neuron:
    - First num_worms: left neurons
    - Second num_worms: right neurons
    
    Args:
        neuron_traces: List of trace arrays for this neuron
        worm_idx: Which worm (0-indexed)
        num_worms: Total number of worms
        
    Returns:
        Averaged trace array, or None if neuron missing
    """
    segments = []
    
    # Check left and right traces
    for offset in [worm_idx, worm_idx + num_worms]:
        if offset >= len(neuron_traces):
            continue
        
        arr = np.asarray(neuron_traces[offset], dtype=float)
        if arr.size == 0:
            continue
        
        segments.append(arr)
    
    if not segments:
        return None
    
    # Trim to shortest length and average
    min_len = min(seg.shape[-1] for seg in segments)
    stacked = np.stack([seg[-min_len:] for seg in segments], axis=0)
    
    return stacked.mean(axis=0)


def select_nodes_and_worms(
    coverage_df: pd.DataFrame,
    neuropal_data: Dict,
    min_worms: int,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Select neurons with sufficient coverage and worms with all selected neurons.
    
    Args:
        coverage_df: Node coverage dataframe
        neuropal_data: NeuroPAL recording data
        min_worms: Minimum worm count threshold
        
    Returns:
        Tuple of (selected_nodes, eligible_worms, name_to_idx)
    """
    # Select nodes with sufficient coverage
    selected_nodes = coverage_df[
        coverage_df["worm_count"] >= min_worms
    ]["node"].tolist()
    
    if not selected_nodes:
        raise ValueError(
            f"No neurons meet the min_worms={min_worms} threshold. "
            f"Try lowering the threshold."
        )
    
    print(f"  Selected {len(selected_nodes)} neurons "
          f"(coverage >= {min_worms} worms)")
    
    # Build name->index mapping
    name_to_idx = {
        name: idx
        for idx, name in enumerate(neuropal_data["neuron_names"])
    }
    
    # Find worms that have ALL selected neurons
    num_worms = len(neuropal_data["worm_ids"])
    eligible_worms = []
    
    for worm_idx in range(num_worms):
        has_all = True
        
        for node in selected_nodes:
            idx = name_to_idx[node]
            trace = collect_worm_trace(
                neuropal_data["norm_traces"][idx],
                worm_idx,
                num_worms
            )
            
            if trace is None:
                has_all = False
                break
        
        if has_all:
            eligible_worms.append(worm_idx)
    
    print(f"  Found {len(eligible_worms)}/{num_worms} worms with complete coverage")
    
    if not eligible_worms:
        raise ValueError(
            "No worms contain all selected neurons. "
            "Try lowering min_worms threshold."
        )
    
    return selected_nodes, eligible_worms, name_to_idx


def build_stimulus_dataset(
    stimulus_name: str,
    stim_idx: int,
    selected_nodes: List[str],
    eligible_worms: List[int],
    neuropal_data: Dict,
    name_to_idx: Dict[str, int],
    output_dir: Path,
) -> None:
    """
    Build lag-window dataset for a specific stimulus.
    
    Creates standardized lag-1 windows: z_t = [x_t, x_{t+1}]
    
    Args:
        stimulus_name: Name of stimulus (e.g., "NaCl")
        stim_idx: Index of stimulus in stim_times array
        selected_nodes: List of neuron names to include
        eligible_worms: List of worm indices to use
        neuropal_data: NeuroPAL recording data
        name_to_idx: Mapping from neuron names to trace indices
        output_dir: Where to save dataset
    """
    print(f"\n  Processing '{stimulus_name}'...")
    
    # Create output directory
    stim_dir = output_dir / stimulus_name.lower().replace(" ", "_")
    stim_dir.mkdir(parents=True, exist_ok=True)
    
    node_indices = [name_to_idx[name] for name in selected_nodes]
    num_worms = len(neuropal_data["worm_ids"])
    fps = neuropal_data["fps"]
    
    # Collect data from each worm
    lag_windows = []
    segment_info = []
    time_series = []
    
    worms_attempted = 0
    worms_contributed = 0
    
    for worm_idx in eligible_worms:
        worms_attempted += 1
        
        # Get stimulus order for this worm
        order = neuropal_data["stims_per_worm"][worm_idx]
        
        # Find when this worm received this stimulus
        try:
            event_pos = int(np.where(order == (stim_idx + 1))[0][0])
        except IndexError:
            # Worm did not receive this stimulus
            continue
        
        # Get time window for this stimulus
        start_sec, end_sec = neuropal_data["stim_times"][event_pos]
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        
        # Build neuron × time matrix for this worm
        columns = []
        for idx in node_indices:
            trace = collect_worm_trace(
                neuropal_data["norm_traces"][idx],
                worm_idx,
                num_worms
            )
            if trace is None:
                raise RuntimeError(
                    f"Missing trace for node {selected_nodes[node_indices.index(idx)]} "
                    f"in worm {worm_idx} despite eligibility check"
                )
            columns.append(trace)
        
        # Truncate to common length
        min_len = min(col.shape[-1] for col in columns)
        matrix = np.stack([col[-min_len:] for col in columns], axis=1)
        
        # Extract stimulus segment
        segment = matrix[max(0, start_frame):min(end_frame, matrix.shape[0]), :]
        
        if segment.shape[0] < 2:
            print(f"    Warning: Segment too short for worm {worm_idx}, skipping")
            continue
        
        # Create lag-1 windows
        x_t = segment[:-1, :]
        x_tp1 = segment[1:, :]
        windows = np.concatenate([x_t, x_tp1], axis=1)
        
        lag_windows.append(windows)
        time_series.append(segment)
        
        segment_info.append({
            "worm_index": worm_idx,
            "worm_id": neuropal_data["worm_ids"][worm_idx],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_used": segment.shape[0],
            "windows_created": windows.shape[0],
        })
        
        worms_contributed += 1
    
    if not lag_windows:
        print(f"    No data collected for {stimulus_name}. Skipping.")
        print(f"    ({worms_contributed}/{worms_attempted} worms contributed)")
        return
    
    # Stack all windows
    Z_raw = np.vstack(lag_windows)
    
    # Standardize (zero mean, unit variance)
    mean = Z_raw.mean(axis=0)
    std = Z_raw.std(axis=0)
    std = np.maximum(std, 1e-6)  # Avoid division by zero
    Z_std = (Z_raw - mean) / std
    
    # Save outputs
    np.save(stim_dir / "Z_raw.npy", Z_raw)
    np.save(stim_dir / "Z_std.npy", Z_std)
    np.save(stim_dir / "X_segments.npy", np.array(time_series, dtype=object))
    
    pd.DataFrame(segment_info).to_csv(stim_dir / "segments.csv", index=False)
    
    # Save standardization metadata
    metadata = {
        "stimulus": stimulus_name,
        "stimulus_index": stim_idx + 1,
        "min_worms": MIN_WORMS,
        "node_order": selected_nodes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_count": int(Z_raw.shape[0]),
        "window_dim": int(Z_raw.shape[1]),
        "segment_count": len(segment_info),
        "worms_contributed": worms_contributed,
        "worms_attempted": worms_attempted,
    }
    
    with open(stim_dir / "standardization.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Created {Z_raw.shape[0]} windows from {worms_contributed} worms")
    print(f"      ({worms_attempted - worms_contributed} worms skipped - "
          f"did not receive this stimulus)")


def build_datasets(
    nodes: List[str],
    data_dir: Path,
    output_dir: Path,
    min_worms: int = MIN_WORMS,
    stimuli_filter: List[str] | None = None,
) -> None:
    """
    Build SBTG datasets for all stimuli.
    
    Args:
        nodes: List of neuron names from connectome
        data_dir: Path to data directory
        output_dir: Where to save datasets
        min_worms: Minimum worm coverage threshold
        stimuli_filter: Optional list of stimuli to process (None = all)
    """
    print("\n" + "=" * 70)
    print("STEP 2: Building SBTG Datasets")
    print("=" * 70)
    
    # Load NeuroPAL data
    neuropal_data = load_neuropal_data(data_dir)
    
    print(f"\nNeuroPAL summary:")
    print(f"  Neurons: {len(neuropal_data['neuron_names'])}")
    print(f"  Worms: {len(neuropal_data['worm_ids'])}")
    print(f"  Stimuli: {', '.join(neuropal_data['stim_names'])}")
    print(f"  Sampling rate: {neuropal_data['fps']:.1f} Hz")
    
    # Compute node coverage
    print(f"\nComputing neuron coverage across worms...")
    coverage_df = compute_node_coverage(nodes, neuropal_data)
    
    # Select nodes and worms
    print(f"\nApplying coverage threshold (min_worms={min_worms})...")
    selected_nodes, eligible_worms, name_to_idx = select_nodes_and_worms(
        coverage_df,
        neuropal_data,
        min_worms
    )
    
    # Determine which stimuli to process
    if stimuli_filter is not None:
        stim_names = [
            s for s in neuropal_data["stim_names"]
            if s.lower() in [f.lower() for f in stimuli_filter]
        ]
    else:
        stim_names = neuropal_data["stim_names"]
    
    print(f"\nBuilding datasets for {len(stim_names)} stimuli:")
    
    # Build dataset for each stimulus
    for stim_idx, stim_name in enumerate(neuropal_data["stim_names"]):
        if stim_name not in stim_names:
            continue
        
        build_stimulus_dataset(
            stim_name,
            stim_idx,
            selected_nodes,
            eligible_worms,
            neuropal_data,
            name_to_idx,
            output_dir
        )
    
    print(f"\n✓ Dataset preparation complete")
    print(f"  Datasets saved to: {output_dir}/")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("PREPARE DATA FOR SBTG ANALYSIS")
    print("=" * 70)
    print("\nThis script prepares all data needed for SBTG analysis:")
    print("  1. Aligns Cook connectome with NeuroPAL neuron names")
    print("  2. Builds standardized lag-window datasets for each stimulus")
    print("  3. Generates quality control visualizations")
    print("\n" + "=" * 70)
    
    # Locate project root
    try:
        project_root = locate_project_root(Path.cwd())
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run this script from within the project directory:")
        print("  python pipeline/01_prepare_data.py")
        sys.exit(1)
    
    print(f"\nProject root: {project_root}")
    
    # Set up paths
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    
    # Create output directories
    output_dirs = ensure_directories(results_dir)
    
    try:
        # Step 1: Align connectome
        nodes, adjacencies = align_connectome(data_dir, output_dirs)
        
        # Step 2: Build datasets
        build_datasets(
            nodes,
            data_dir,
            output_dirs["datasets"],
            min_worms=MIN_WORMS,
            stimuli_filter=STIMULI
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print("\nData preparation complete. Next steps:")
        print("  1. Review figures in results/figures/connectome/")
        print("  2. Check dataset summaries in results/intermediate/datasets/")
        print("  3. Run: python pipeline/02_train_models.py")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPlease check:")
        print("  - Data files are in data/ directory")
        print("  - File names match expected names")
        print("  - MIN_WORMS threshold is not too high")
        sys.exit(1)


if __name__ == "__main__":
    # Set plotting style
    sns.set_context("talk")
    plt.style.use("seaborn-v0_8-colorblind")
    
    main()
