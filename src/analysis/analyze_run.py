#!/usr/bin/env python3
import argparse
import ast
import json
import logging
import math
import os
import re
from typing import Optional, Union, Sequence, Tuple, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.stats import linregress

from analysis.analyze_cross_network_results import read_input_tsv, add_marginal_means_bars, annotate_heatmap, \
    HEATMAP_TIGHT_RECT

DEFAULT_SIGNIFICANCE_LEVEL = 0.05
HUMAN_DISEASE_MONDO = "mondo.0700096"  #

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)
SHOW_PLOTS = True

_BASE_MD = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}


def plot_category_significance_by_children(
        results_df,
        children,
        parent_to_samples,
        plots_dir,
        *,
        significance_col="significant",
        id_to_name=None,
):
    """
    Heatmap: rows = method combos, cols = MONDO children categories.
    Cell = percent of diseases in that category where the combo is significant at least once.

    Implementation details matching requested spec:
      - "Children": use the list of MONDO IDs *directly* below HUMAN_DISEASE (argument `children`).
      - For each child, take *all* diseases from `results_df` whose `parents` list contains that child
        (provided via `parent_to_samples[child]`). This full set size is the denominator.
      - A disease counts as significant for a `(algorithm | prioritization)` combo if `significant` is True
        in any repeat for that disease under that combo. Missing values are treated as not significant.
      - Drop columns with zero diseases, sort columns by set size (desc), and label with resolved names.
    """
    # Prepare per-(combo, sample) significance, treating missing as False
    df = results_df[["algorithm", "prioritization_algorithm", "sample", significance_col]].copy()
    if significance_col not in df.columns:
        logger.error("Column '%s' not found in results_df.", significance_col)
        return
    df[significance_col] = df[significance_col].fillna(False).astype(bool)
    df["_combo"] = df["algorithm"].astype(str) + " | " + df["prioritization_algorithm"].astype(str)

    # Reduce to disease level within combo: any() significant over repeats
    per = (
        df.groupby(["_combo", "sample"], as_index=False)[significance_col]
        .any()
        .rename(columns={significance_col: "sig_for_combo"})
    )

    # Build usable children list with sizes, then sort by size desc then by name
    child_info = []
    denom_map = {}
    for child in children:
        # full set of diseases in results having this child anywhere in `parents`
        disease_set = set(parent_to_samples.get(child, []))
        if disease_set:
            n = len(disease_set)
            denom_map[child] = float(n)
            nm = id_to_name.get(child, child) if id_to_name else child
            child_info.append((child, n, nm))
    if not child_info:
        logger.warning("No MONDO child categories contain any diseases after filtering. Skipping heatmap.")
        return

    child_info.sort(key=lambda t: (-t[1], t[2]))
    col_order = [c for c, _, _ in child_info]
    col_labels = [f"{nm} ({n})" for _, n, nm in child_info]

    # All combos
    all_combos = sorted(per["_combo"].unique())

    # Numerators per (combo, child)
    heat_counts = pd.DataFrame(0.0, index=all_combos, columns=col_order)
    for child in col_order:
        cat_samples = set(parent_to_samples.get(child, []))
        if not cat_samples:
            continue
        tmp = per[per["sample"].isin(cat_samples)]
        num = tmp.groupby("_combo")["sig_for_combo"].sum()
        for combo, k in num.items():
            heat_counts.at[combo, child] = float(k)

    # Percentages with the *full* category size as denominator
    heat = heat_counts.copy()
    for child in col_order:
        heat[child] = (heat[child] / denom_map[child]) * 100.0

    # Plot
    fig_width = max(8, heat.shape[1] * 0.8)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect="auto", vmin=0.0, vmax=100.0)

    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100), right_size="2%")
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel("Disease category")
    ax.set_ylabel("Module | Prioritization")
    fig.suptitle("Percent significant by MONDO category\nDenominator = diseases in category")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% significant")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_path = os.path.join(plots_dir, "category_significance_pct_heatmap.pdf")
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved category significance heatmap to %s", out_path)


# Keys are case-sensitive; use exactly these names. :contentReference[oaicite:2]{index=2}
# --- New function: plot_cumcount_by_stratify ---
def plot_cumcount_by_stratify(df: pd.DataFrame, column: str, stratify: Optional[Union[str, Sequence[str]]] = None,
                              xlim: Tuple[float, float] = (0, 1), title: Optional[str] = None,
                              xlabel: Optional[str] = None, ylabel: str = 'Cumulative count',
                              save_path: Optional[str] = None,
                              significance_level: Optional[float] = None,
                              bh_sig_level: Optional[float] = None) -> None:
    """
    Plot empirical cumulative *count* (cumulative frequency) of a column, optionally stratified by
    one or two grouping columns. This is like an ECDF but with counts on the y-axis instead of
    relative probability.

    Parameters:
    df          : DataFrame with data
    column      : Column to plot
    stratify    : Column name or sequence of up to two column names for stratification
    xlim        : x-axis limits
    title       : Custom plot title
    xlabel      : x-axis label
    ylabel      : y-axis label (defaults to 'Cumulative count')
    save_path   : Path to save the plot file
    significance_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x.
    bh_sig_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x (recommended: max p_value_DCG where df['significant'] == True).
    """
    # Determine grouping
    groups = [] if stratify is None else ([stratify] if isinstance(stratify, str) else list(stratify))

    # Setup color and style maps (consistent with other stratified plots)
    if len(groups) >= 1:
        unique1 = df[groups[0]].unique()
        cmap = plt.get_cmap('tab10')
        colors = {val: cmap(i) for i, val in enumerate(unique1)}
    if len(groups) == 2:
        unique2 = df[groups[1]].unique()
        styles = ['solid', 'dashed', 'dashdot', 'dotted']
        dash_styles = {val: styles[i % len(styles)] for i, val in enumerate(unique2)}

    def _cumcount(series: pd.Series):
        x = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        if x.size == 0:
            return None, None
        x = np.sort(x)
        y = np.arange(1, x.size + 1)
        return x, y

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    max_count = 0
    if not groups:
        x, y = _cumcount(df[column])
        if y is not None:
            ax.step(x, y, where='post', label='All')
            max_count = max(max_count, int(y[-1]))
    elif len(groups) == 1:
        for val in unique1:
            subset = df[df[groups[0]] == val][column]
            x, y = _cumcount(subset)
            if y is not None:
                ax.step(x, y, where='post', label=str(val), color=colors[val])
                max_count = max(max_count, int(y[-1]))
    else:
        for val1 in unique1:
            for val2 in unique2:
                subset = df[(df[groups[0]] == val1) & (df[groups[1]] == val2)][column]
                x, y = _cumcount(subset)
                if y is not None:
                    ax.step(x, y, where='post', color=colors[val1], linestyle=dash_styles[val2],
                            label=f"{val1} | {val2}")
                    max_count = max(max_count, int(y[-1]))

    # Finalize
    ax.set_xlim(*xlim)
    if max_count > 0:
        ax.set_ylim(0, max_count)
    plot_title = title or f"Cumulative count of {column}" + (f" by {groups}" if groups else "")
    ax.set_title(plot_title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

    # Optional dashed significance lines for p_value_DCG
    added_lines = False
    if column == 'p_value_DCG':
        if significance_level is not None:
            ax.axvline(significance_level, linestyle='--', linewidth=1.5, label=f"alpha={significance_level:g}")
            added_lines = True
        if bh_sig_level is not None and np.isfinite(bh_sig_level):
            ax.axvline(bh_sig_level, linestyle='--', linewidth=1.5, label=f"BH cutoff≈{bh_sig_level:.3g}")
            added_lines = True

    if groups or added_lines:
        legend_title = (' | '.join(groups)) if groups else 'Thresholds'
        ax.legend(title=legend_title, loc='best', fontsize='small')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --- add this small helper just above the savefig patches ---
def _with_net_suffix(fname):
    """Append NETWORK_NAME to output filenames just before the extension."""
    try:
        path = os.fspath(fname)  # supports str and PathLike
    except TypeError:
        return fname  # file-like objects, leave unchanged
    base, ext = os.path.splitext(path)
    return f"{base}_{NETWORK_NAME}{ext}" if NETWORK_NAME else fname


# Patch plt.savefig
_orig_plt_savefig = plt.savefig


def _plt_savefig(fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    fname = _with_net_suffix(fname)  # <-- append network here
    return _orig_plt_savefig(fname, *a, **k)


plt.savefig = _plt_savefig

# Patch Figure.savefig (covers fig.savefig)
_orig_fig_savefig = Figure.savefig


def _fig_savefig(self, fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    fname = _with_net_suffix(fname)  # <-- append network here
    return _orig_fig_savefig(self, fname, *a, **k)


Figure.savefig = _fig_savefig


# --- helpers for bounds and annotation placement ---

def _infer_y_bounds(y: pd.Series, y_name: str) -> Tuple[float, float]:
    """Infer sensible y-limits from data and common metric names.
    - p-values: [0, 1]
    - percentages: [0, 100]
    - otherwise: min/max of finite values.
    """
    name = (y_name or "").lower()
    finite = pd.to_numeric(y, errors='coerce').dropna()
    if "p_value" in name or name == "p" or name.endswith("_p"):
        return 0.0, 1.0
    if "percent" in name or name.endswith("_pct") or name.endswith("_percent"):
        return 0.0, 100.0
    if finite.empty:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def _pick_annotation_corner(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str, str]:
    """Choose a corner in axes coords (x_ax, y_ax) and ha/va based on sparsest quadrant.
    Split by medians; place text near the corner with least points.
    Returns (x_ax, y_ax, ha, va).
    """
    if x.size == 0 or y.size == 0:
        return 0.98, 0.02, 'right', 'bottom'
    x_med = np.nanmedian(x)
    y_med = np.nanmedian(y)
    q_counts = {'bl': np.sum((x <= x_med) & (y <= y_med)), 'br': np.sum((x > x_med) & (y <= y_med)),
                'tl': np.sum((x <= x_med) & (y > y_med)), 'tr': np.sum((x > x_med) & (y > y_med)), }
    corner = min(q_counts, key=q_counts.get)
    mapping = {'bl': (0.02, 0.02, 'left', 'bottom'), 'br': (0.98, 0.02, 'right', 'bottom'),
               'tl': (0.02, 0.98, 'left', 'top'), 'tr': (0.98, 0.98, 'right', 'top'), }
    return mapping[corner]


# Global network name used to annotate plot titles

NETWORK_NAME = ""

# Candidate columns that might carry the network/ppi identifier
_NETWORK_COL_CANDIDATES = ("ppi", "network", "PPI", "Network")


def _find_network_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching network column present in df, or None."""
    for col in _NETWORK_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _apply_network_selection(df: pd.DataFrame, requested: Optional[str]):
    """Apply network selection logic.

    If `requested` is provided, verify that a network column exists and that the
    requested value is present. Filter to that network and return (df_filtered, name).

    If `requested` is None, infer the network context:
      - If no network column exists, return (df, "").
      - If exactly one unique value exists, return (df, that value).
      - Otherwise return (df, "multiple").
    """
    col = _find_network_column(df)
    if requested:
        if col is None:
            logger.error("--network was provided ('%s') but no network column found. Tried columns: %s", requested,
                         _NETWORK_COL_CANDIDATES)
            raise SystemExit(2)
        uniq = df[col].dropna().astype(str).unique()
        if requested not in set(uniq):
            logger.error("Requested network '%s' not found in column '%s'. Available: %s", requested, col,
                         ", ".join(map(str, uniq)))
            raise SystemExit(2)
        df2 = df[df[col].astype(str) == requested].copy()
        logger.info("Using requested network '%s' (column '%s'); %d rows after filtering.", requested, col, len(df2))
        return df2, requested
    else:
        if col is None:
            logger.info(
                "No network column found. Proceeding without network context; outputs won't be suffixed by network.")
            return df, ""
        vals = df[col].dropna().astype(str).unique()
        if len(vals) == 1:
            name = str(vals[0])
            logger.info("Detected single network '%s' from column '%s'.", name, col)
            return df, name
        else:
            shown = ", ".join(map(str, vals[:8])) + (" …" if len(vals) > 8 else "")
            logger.info("Detected multiple networks (%d) from column '%s': %s. Using 'multiple'.", len(vals), col,
                        shown)
            return df, "multiple"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze a run of drug prioritization results.")
    parser.add_argument("--input_tsv", default="../../data/input/cross_network_data.tsv.gz",
                        help="Path to the input TSV file containing run data.")
    parser.add_argument("--plots_dir", default="../../plots/results", help="Directory to save analysis results.")
    parser.add_argument("--significance_level", type=float, default=DEFAULT_SIGNIFICANCE_LEVEL,
                        help=f"Significance level for p-value analysis (default: {DEFAULT_SIGNIFICANCE_LEVEL}).")
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='Suppress display of plots; only save them to files.')
    parser.add_argument('--network', type=str, default="string_min700",
                        help='Optional network/ppi name to analyze. If provided, the data is filtered to this network; error if not present.')
    parser.add_argument("--disorder_is_subtype_of_disorder",
                        default="../../data/nedrexDB/disorder_is_subtype_of_disorder.csv",
                        help="CSV disorder_is_subtype_of_disorder")
    parser.add_argument("--disorders_csv", default="../../data/nedrexDB/disorder.csv",
                        help="CSV mapping primaryDomainId to displayName")
    return parser.parse_args()



def plot_correlation_matrix(results_df, plots_dir, plot_title, file_name, ignore_cols, annot_fontsize=10):
    """
    column_order: list[str] or None
        If provided, columns are reordered to this list first, then any remaining numeric columns are appended.
        If None, uses COLUMN_ORDER above. If both None, uses dataframe's current order.
    annot_fontsize: int
        Font size for the numbers shown inside each cell.
        Increase this to make the in-cell labels larger without changing figure size.
    """

    # 1. Numeric selection with optional drops
    numerical_df = results_df.select_dtypes(include=['number']).drop(columns=(ignore_cols or []), errors='ignore')
    # 2. Column reordering
    column_order = ['seeds','approved_drugs_with_targets','num_children','num_parents','seed_entropy','seed_gene_score_mean','seed_specificity_mean','seed_specificity_adjusted_mean','jaccard_max','jaccard_mean','seeds_not_in_network','nodes','edges','max_dist_to_seed','diameter','components','largest_component','isolated_nodes','candidate_count','p_value_DCG','p_value_without_ranks','percent_true_drugs_found']
    present = [c for c in column_order if c in numerical_df.columns]
    missing = [c for c in column_order if c not in numerical_df.columns]  # likely typos / not numeric
    extras = [c for c in numerical_df.columns if c not in column_order]  # numeric cols not in requested list
    desired = present + extras

    if missing:
        logger.warning("Requested numeric columns not found (possible misspellings): %s", ", ".join(missing))
    if extras:
        logger.debug("Numeric columns that will be appended (not in requested order): %s", ", ".join(extras))

    # Only perform a reorder if the desired order differs from current
    if list(numerical_df.columns) != desired and desired:
        logger.info("Reordering numeric columns to requested order (present ones first). New order: %s",
                    ", ".join(desired))
        numerical_df = numerical_df.loc[:, desired]
    else:
        logger.debug("Numeric columns already in desired order or no numeric columns found.")

    # 3. Correlation — compute and enforce same ordering on both axes so heatmap matches
    corr = numerical_df.corr(method='spearman')
    corr = corr.reindex(index=desired, columns=desired)



    # 4. Plot (figure size unchanged)
    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    # 5. Colorbar
    cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Spearman correlation', rotation=270, labelpad=15)

    # 6. Tick labels
    labels = corr.columns
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # 7. Cell outlines via minor grid
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.grid(False, which='major')
    ax.tick_params(which='minor', bottom=False, left=False)

    # 8. In-cell annotations (slightly larger via annot_fontsize)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iat[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha='center', va='center',
                fontsize=annot_fontsize,
                color='white' if abs(val) > 0.5 else 'black'
            )

    # 9. Titles
    ax.set_title(f'{plot_title}' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=20)
    ax.set_xlabel('Numeric Features')
    ax.set_ylabel('Numeric Features')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, file_name), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_distribution_by_stratify(df: pd.DataFrame, column: str, stratify: Optional[Union[str, Sequence[str]]] = None,
                                  xlim: Tuple[float, float] = (0, 1), title: Optional[str] = None,
                                  xlabel: Optional[str] = None, ylabel: str = 'Density',
                                  save_path: Optional[str] = None,
                                  significance_level: Optional[float] = None,
                                  bh_sig_level: Optional[float] = None) -> None:
    """
    Plot KDE distribution of a column, optionally stratified by one or two grouping columns.

    Parameters:
    df          : DataFrame with data
    column      : Column to plot
    stratify    : Column name or sequence of up to two column names for stratification
    xlim        : x-axis limits
    title       : Custom plot title
    xlabel      : x-axis label
    ylabel      : y-axis label
    save_path   : Path to save the plot file
    significance_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x.
    bh_sig_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x (recommended: max p_value_DCG where df['significant'] == True).
    """
    # Determine grouping
    groups = [] if stratify is None else ([stratify] if isinstance(stratify, str) else list(stratify))

    # Setup color and style maps
    if len(groups) >= 1:
        unique1 = df[groups[0]].unique()
        cmap = plt.get_cmap('tab10')
        colors = {val: cmap(i) for i, val in enumerate(unique1)}
    if len(groups) == 2:
        unique2 = df[groups[1]].unique()
        styles = ['solid', 'dashed', 'dashdot', 'dotted']
        dash_styles = {val: styles[i % len(styles)] for i, val in enumerate(unique2)}

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if not groups:
        df[column].plot.kde(ax=ax)
    elif len(groups) == 1:
        for val in unique1:
            subset = df[df[groups[0]] == val][column]
            subset.plot.kde(ax=ax, label=str(val), color=colors[val])
    else:
        for val1 in unique1:
            for val2 in unique2:
                subset = df[(df[groups[0]] == val1) & (df[groups[1]] == val2)][column]
                if not subset.empty:
                    subset.plot.kde(ax=ax, color=colors[val1], linestyle=dash_styles[val2], label=f"{val1} | {val2}")

    # Finalize
    ax.set_xlim(*xlim)
    plot_title = title or f"KDE of {column}" + (f" by {groups}" if groups else "")
    ax.set_title(plot_title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

    # Optional dashed significance lines for p_value_DCG
    added_lines = False
    if column == 'p_value_DCG':
        if significance_level is not None:
            ax.axvline(significance_level, linestyle='--', linewidth=1.5, label=f"alpha={significance_level:g}")
            added_lines = True
        if bh_sig_level is not None and np.isfinite(bh_sig_level):
            ax.axvline(bh_sig_level, linestyle='--', linewidth=1.5, label=f"BH cutoff≈{bh_sig_level:.3g}")
            added_lines = True

    if groups or added_lines:
        legend_title = (' | '.join(groups)) if groups else 'Thresholds'
        ax.legend(title=legend_title, loc='best', fontsize='small')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --- ECDF plot for p-value distributions (used in main) ---
def plot_cdf_by_stratify(df: pd.DataFrame, column: str, stratify: Optional[Union[str, Sequence[str]]] = None,
                         xlim: Tuple[float, float] = (0, 1), title: Optional[str] = None,
                         xlabel: Optional[str] = None, ylabel: str = 'Cumulative probability',
                         save_path: Optional[str] = None,
                         significance_level: Optional[float] = None,
                         bh_sig_level: Optional[float] = None,
                         overlay_values: Optional[pd.Series] = None,
                         overlay_label: str = 'Best per sample') -> None:
    """
    Plot empirical cumulative distribution (ECDF) of a column, optionally stratified by one or two grouping columns.

    Parameters:
    df          : DataFrame with data
    column      : Column to plot
    stratify    : Column name or sequence of up to two column names for stratification
    xlim        : x-axis limits
    title       : Custom plot title
    xlabel      : x-axis label
    ylabel      : y-axis label (defaults to 'Cumulative probability')
    save_path   : Path to save the plot file
    significance_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x.
    bh_sig_level : Optional[float]
        If provided and column == 'p_value_DCG', draw a dashed vertical line at this x (recommended: max p_value_DCG where df['significant'] == True).
    overlay_values : Optional[pd.Series]
        If provided, overlays an additional ECDF line computed from these values (e.g., per-sample minimum p-values across all combinations).
    overlay_label : str
        Legend label for the overlay ECDF line.
    """
    # Determine grouping
    groups = [] if stratify is None else ([stratify] if isinstance(stratify, str) else list(stratify))

    # Setup color and style maps (same policy as KDE variant)
    if len(groups) >= 1:
        unique1 = df[groups[0]].unique()
        cmap = plt.get_cmap('tab10')
        colors = {val: cmap(i) for i, val in enumerate(unique1)}
    if len(groups) == 2:
        unique2 = df[groups[1]].unique()
        styles = ['solid', 'dashed', 'dashdot', 'dotted']
        dash_styles = {val: styles[i % len(styles)] for i, val in enumerate(unique2)}

    def _ecdf(series: pd.Series):
        x = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        if x.size == 0:
            return None, None
        x = np.sort(x)
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if not groups:
        x, y = _ecdf(df[column])
        if y is not None:
            ax.step(x, y, where='post', label='All')
    elif len(groups) == 1:
        for val in unique1:
            subset = df[df[groups[0]] == val][column]
            x, y = _ecdf(subset)
            if y is not None:
                ax.step(x, y, where='post', label=str(val), color=colors[val])
    else:
        for val1 in unique1:
            for val2 in unique2:
                subset = df[(df[groups[0]] == val1) & (df[groups[1]] == val2)][column]
                x, y = _ecdf(subset)
                if y is not None:
                    ax.step(x, y, where='post', color=colors[val1], linestyle=dash_styles[val2],
                            label=f"{val1} | {val2}")

    # Optional overlay ECDF (e.g., best-per-sample across all)
    overlay_added = False
    if overlay_values is not None:
        x_o, y_o = _ecdf(pd.to_numeric(overlay_values, errors='coerce'))
        if y_o is not None:
            ax.step(x_o, y_o, where='post', label=overlay_label, linewidth=2.0, color='black', alpha=0.9)
            overlay_added = True

    # Finalize
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    plot_title = title or f"ECDF of {column}" + (f" by {groups}" if groups else "")
    ax.set_title(plot_title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

    # Optional dashed significance lines for p_value_DCG
    added_lines = False
    if column == 'p_value_DCG':
        if significance_level is not None:
            ax.axvline(significance_level, linestyle='--', linewidth=1.5, label=f"alpha={significance_level:g}")
            added_lines = True
        if bh_sig_level is not None and np.isfinite(bh_sig_level):
            ax.axvline(bh_sig_level, linestyle='--', linewidth=1.5, label=f"BH cutoff≈{bh_sig_level:.3g}")
            added_lines = True

    if groups or added_lines or overlay_added:
        legend_title = (' | '.join(groups)) if groups else 'Reference'
        ax.legend(title=legend_title, loc='best', fontsize='small')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# Additional plotting functions
def plot_scatter(df: pd.DataFrame, plot_dir: str, x_col: str, y_col: str, title: str, filename: str,
                 color_by: Optional[str] = None, alpha: float = 0.7) -> None:
    """Plot scatter of x_col vs y_col, colored by a third variable if provided."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if color_by:
        scatter = ax.scatter(df[x_col], df[y_col], c=df[color_by], alpha=alpha, cmap='viridis')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=alpha)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_hexbin(df: pd.DataFrame, plot_dir: str, x_col: str, y_col: str, title: str, filename: str, gridsize: int = 50,
                cmap: str = 'Blues') -> None:
    """Plot 2D hexbin of x_col vs y_col."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hexbin = ax.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap)
    cbar = fig.colorbar(hexbin, ax=ax)
    cbar.set_label('counts')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --- New function: plot_scatter_grid ---
def plot_scatter_grid(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], plots_dir: str, title: Optional[str] = None,
                      filename: Optional[str] = None, color_by: Optional[str] = None, alpha: float = 0.5,
                      add_trend: bool = True,
                      show_p: bool = True, ) -> None:
    """
    Create a grid (matrix) of scatter plots for a single dependent variable against multiple predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the columns to plot.
    y_col : str
        Column name placed on the y-axis for all subplots.
    x_cols : Sequence[str]
        Iterable of column names to place on the x-axes (one subplot per column).
    plots_dir : str
        Directory to save the resulting figure.
    title : Optional[str]
        Optional figure-level title. If None, a descriptive title is generated.
    filename : Optional[str]
        Output filename (PDF). If None, uses `scatter_grid_{y_col}_vs_{x_cols}.pdf`.
    color_by : Optional[str]
        Optional column to map to point color. If provided, a single shared colorbar is added.
    alpha : float
        Point transparency.
    add_trend : bool
        If True, overlay a least-squares regression line per subplot.
    show_p : bool
        If True, annotate each subplot with the p-value (slope \u2260 0) and R.

    Notes
    -----
    - The grid shape is chosen to be as square as possible, using
      ncols = ceil(sqrt(n_plots)) and nrows = ceil(n_plots / ncols).
    - Axes are *not* shared, so each subplot auto-scales independently, which
      is helpful when distributions differ across variables.
    """
    n_plots = len(x_cols)
    if n_plots == 0:
        logger.warning("plot_scatter_grid called with empty x_cols; nothing to plot.")
        return

    # Compute a near-square grid
    ncols = int(math.ceil(math.sqrt(n_plots)))
    nrows = int(math.ceil(n_plots / ncols))

    # Figure size scales with grid dimensions
    # Width ~4 per col, height ~3.2 per row (tuned for readability)
    fig_w = max(8.0, 4.0 * ncols)
    fig_h = max(6.0, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=False, sharey=False)

    # Track a single mappable for a shared colorbar if color_by is provided
    mappable = None

    for i, x in enumerate(x_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        cols_needed = [x, y_col]
        if color_by is not None:
            cols_needed.append(color_by)
        data = df[cols_needed].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        # Infer and apply y-axis bounds to avoid implying invalid values
        y_min, y_max = _infer_y_bounds(data[y_col], y_col)
        ax.set_ylim(y_min, y_max)
        if color_by is not None:
            sc = ax.scatter(data[x], data[y_col], c=data[color_by], cmap='viridis', alpha=alpha)
            if mappable is None:
                mappable = sc
        else:
            ax.scatter(data[x], data[y_col], alpha=alpha)

        # Add OLS trend line and p-value annotation if requested and enough data
        if add_trend and len(data) >= 2 and data[x].nunique() >= 2:
            try:
                lr = linregress(data[x].to_numpy(), data[y_col].to_numpy())
                x_min, x_max = np.nanmin(data[x].to_numpy()), np.nanmax(data[x].to_numpy())
                x_line = np.linspace(x_min, x_max, 100)
                y_line = lr.slope * x_line + lr.intercept
                y_line = np.clip(y_line, y_min, y_max)
                ax.plot(x_line, y_line, linewidth=2, alpha=0.9, color='red')
                if show_p:
                    # Dynamic placement: pick emptiest corner, place in data coords, then repel from points
                    x_ax, y_ax, ha, va = _pick_annotation_corner(data[x].to_numpy(), data[y_col].to_numpy())
                    # Convert corner in axes coords -> data coords with small padding
                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()
                    pad_x = 0.02 * (x1 - x0)
                    pad_y = 0.02 * (y1 - y0)
                    x_data = x0 + pad_x if ha == 'left' else x1 - pad_x
                    y_data = y0 + pad_y if va == 'bottom' else y1 - pad_y
                    txt = ax.text(x_data, y_data, f"p={lr.pvalue:.2e}\nR={lr.rvalue:.2f}", ha=ha, va=va, fontsize=9,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
                    try:
                        adjust_text([txt], x=data[x].to_numpy(), y=data[y_col].to_numpy(), ax=ax,
                                    expand_points=(1.2, 1.2), expand_text=(1.05, 1.2), force_points=(0.5, 0.5),
                                    force_text=(0.3, 0.3), only_move={'points': 'xy', 'text': 'xy'}, lim=100, )
                    except Exception as e:
                        logger.debug(f"adjust_text failed for {x} vs {y_col}: {e}")
            except Exception as e:
                logger.debug(f"Trend line failed for {x} vs {y_col}: {e}")
        ax.set_xlabel(x)
        ax.set_ylabel(y_col)  # ax.set_title(f"{y_col} vs {x}")

    # Hide any unused axes
    for j in range(n_plots, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(title + (f" ({NETWORK_NAME})" if NETWORK_NAME else ""), y=0.995, fontsize=14)

    # Shared colorbar if applicable
    if mappable is not None:
        # Attach one shared colorbar spanning all axes
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label(color_by)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save and optionally show
    if filename is None:
        x_stub = "_".join(x_cols[:4])
        if len(x_cols) > 4:
            x_stub += f"_plus{len(x_cols) - 4}"
        out_name = f"scatter_grid_{y_col}_vs_{x_stub}.pdf"
    else:
        out_name = filename
    fig.savefig(os.path.join(plots_dir, out_name), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# --- New function: plot_hexbin_grid_log ---
def plot_hexbin_grid_log(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], plots_dir: str,
                         title: Optional[str] = None, filename: Optional[str] = None, gridsize: int = 50,
                         cmap: str = 'viridis',
                         mincnt: int = 1, ) -> None:
    """
    Grid of hexbin density plots with a single shared log-scaled colorbar.

    - One subplot per x in `x_cols`, plotting (x, y_col).
    - Uses a global LogNorm so densities are comparable across panels.
    - Removes subplot titles (keeps axis labels for readability).
    """
    n_plots = len(x_cols)
    if n_plots == 0:
        logger.warning("plot_hexbin_grid_log called with empty x_cols; nothing to plot.")
        return

    # Layout
    ncols = int(math.ceil(math.sqrt(n_plots)))
    nrows = int(math.ceil(n_plots / ncols))
    fig_w = max(8.0, 4.0 * ncols)
    fig_h = max(6.0, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=False, sharey=False)
    # Reserve right margin for a dedicated colorbar axis
    fig.subplots_adjust(right=0.86)

    # First pass: draw hexbins, track global max count
    collections = []
    global_max = 1.0
    for i, x in enumerate(x_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        data = df[[x, y_col]].dropna()
        if data.empty:
            ax.set_visible(False)
            collections.append(None)
            continue
        # Infer and apply y-axis bounds
        y_min, y_max = _infer_y_bounds(data[y_col], y_col)
        ax.set_ylim(y_min, y_max)
        hb = ax.hexbin(data[x].to_numpy(), data[y_col].to_numpy(), gridsize=gridsize, mincnt=mincnt, cmap=cmap, )
        # Optional OLS trend line and annotation
        if len(data) >= 2 and data[x].nunique() >= 2:
            try:
                lr = linregress(data[x].to_numpy(), data[y_col].to_numpy())
                x_min, x_max = np.nanmin(data[x].to_numpy()), np.nanmax(data[x].to_numpy())
                x_line = np.linspace(x_min, x_max, 100)
                y_line = lr.slope * x_line + lr.intercept
                y_line = np.clip(y_line, y_min, y_max)
                ax.plot(x_line, y_line, linewidth=2, alpha=0.9, color='red')
                # Dynamic placement: pick emptiest corner, place in data coords, then repel from points
                x_ax, y_ax, ha, va = _pick_annotation_corner(data[x].to_numpy(), data[y_col].to_numpy())
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                pad_x = 0.02 * (x1 - x0)
                pad_y = 0.02 * (y1 - y0)
                x_data = x0 + pad_x if ha == 'left' else x1 - pad_x
                y_data = y0 + pad_y if va == 'bottom' else y1 - pad_y
                txt = ax.text(x_data, y_data, f"p={lr.pvalue:.2e}\nR={lr.rvalue:.2f}", ha=ha, va=va, fontsize=9,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
                try:
                    adjust_text([txt], x=data[x].to_numpy(), y=data[y_col].to_numpy(), ax=ax, expand_points=(1.2, 1.2),
                                expand_text=(1.05, 1.2), force_points=(0.5, 0.5), force_text=(0.3, 0.3),
                                only_move={'points': 'xy', 'text': 'xy'}, lim=100, )
                except Exception as e:
                    logger.debug(f"adjust_text failed for {x} vs {y_col}: {e}")
            except Exception as e:
                logger.debug(f"Trend line failed for {x} vs {y_col}: {e}")
        # Track max raw count from this panel
        arr = hb.get_array()
        if arr.size:
            panel_max = float(np.nanmax(arr))
            if panel_max > global_max:
                global_max = panel_max
        collections.append(hb)
        # Labels, but no per-panel title
        ax.set_xlabel(x)
        ax.set_ylabel(y_col)

    # Hide any unused axes
    for j in range(n_plots, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    # Apply shared log normalization across all panels
    norm = LogNorm(vmin=max(1, mincnt), vmax=global_max if global_max >= 1 else 1)
    mappable = None
    for hb in collections:
        if hb is None:
            continue
        hb.set_norm(norm)
        if mappable is None:
            mappable = hb

    # Figure-level title
    if title is None:
        if n_plots <= 4:
            x_list_text = ", ".join(x_cols)
        else:
            x_list_text = ", ".join(x_cols[:4]) + f" … (+{n_plots - 4} more)"
        title = f"Hexbin grid (log density): {y_col} vs {x_list_text}"
    fig.suptitle(title + (f" ({NETWORK_NAME})" if NETWORK_NAME else ""), y=0.995, fontsize=14)

    # One shared colorbar on the side for the whole grid
    if mappable is not None:
        # Place colorbar in a dedicated axis outside the grid
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("count (log scale)")

    fig.tight_layout(rect=[0, 0, 0.86, 0.97])

    # Save
    if filename is None:
        x_stub = "_".join(x_cols[:4])
        if len(x_cols) > 4:
            x_stub += f"_plus{len(x_cols) - 4}"
        out_name = f"hexgrid_log_{y_col}_vs_{x_stub}.pdf"
    else:
        out_name = filename
    fig.savefig(os.path.join(plots_dir, out_name), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def get_disease_sets_by_algo_combination(df: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    """
    Return a dict mapping (algorithm, prioritization_algorithm) to set of samples (diseases) in df.
    """
    sets_dict: Dict[Tuple[str, str], Set[str]] = {}
    grouped = df.groupby(['algorithm', 'prioritization_algorithm'])
    for (algo, prio_algo), group in grouped:
        sets_dict[(algo, prio_algo)] = set(group['sample'])
    return sets_dict


def plot_jaccard_matrix_from_sets(sets_dict: Dict[Tuple[str, str], Set[str]],
                                  plots_dir: str,
                                  title: str,
                                  filename: str) -> None:
    """
    Plot a heatmap of Jaccard indices for each algorithm/prioritization_algorithm combination.
    Uses annotate_heatmap for contrast-aware cell labels and larger axis tick labels.
    """
    # Preserve insertion order of keys for deterministic labeling
    keys = list(sets_dict.keys())
    labels = [f"{algo} | {prio}" for algo, prio in keys]
    n = len(labels)

    # Build Jaccard matrix
    matrix = np.zeros((n, n), dtype=float)
    for i, key_i in enumerate(keys):
        set_i = sets_dict[key_i]
        for j, key_j in enumerate(keys):
            set_j = sets_dict[key_j]
            union = set_i | set_j
            matrix[i, j] = (len(set_i & set_j) / len(union)) if union else 0.0

    # Figure size scales with matrix size; keep it readable
    fig_width = max(13, n * 0.6)
    fig_height = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Heatmap
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    # Axis ticks and larger labels for readability
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Module | Prioritization", fontsize=12)
    ax.set_ylabel("Module | Prioritization", fontsize=12)

    # Title (include network name if available)
    ax.set_title(title + (f" ({NETWORK_NAME})" if NETWORK_NAME else ""), pad=20)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Jaccard Index", rotation=270, labelpad=15)

    # Contrast-aware per-cell annotations
    annotate_heatmap(
        ax,
        im,
        matrix,
        fmt="{:.2f}",
        fontsize=10,
        textcolors=("black", "white"),
        min_contrast=3.0,
        use_outline=False,
        outline_width=0.8,
        bbox=False,
    )

    # Layout (reuse your shared tight rect if available)
    try:
        plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    except NameError:
        plt.tight_layout()

    # Save and show
    out_path = os.path.join(plots_dir, filename)
    plt.savefig(out_path, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    logger.info("Saved Jaccard heatmap to %s", out_path)


def plot_delivery_and_significance(df: pd.DataFrame, plots_dir: str, significance_threshold: float = 0.05) -> None:
    """
    Plot three bar charts showing, for each algorithm/prioritization_algorithm combination:
    1) The percentage of diseases for which a result was delivered.
    2) The percentage of diseases with significant p_value_DCG < significance_threshold.
    3) The percentage of diseases with significant p_value_DCG < significance_threshold divided by the number of delivered results for each algorithm/prioritization combination.
    """
    # Determine total number of unique diseases
    total_diseases = df['sample'].nunique()

    # Count deliveries and significant results
    grouped = df.groupby(['algorithm', 'prioritization_algorithm'])
    counts = grouped['sample'].nunique().unstack(fill_value=0)
    # Use precomputed boolean significance
    sig_counts = df[df['significant']].groupby(['algorithm', 'prioritization_algorithm'])['sample'].nunique().unstack(
        fill_value=0)

    # Convert to percentages
    delivery_pct = counts / total_diseases * 100
    sig_pct = sig_counts / total_diseases * 100

    # Setup bar plot parameters
    algorithms = delivery_pct.index.tolist()
    prio_algs = delivery_pct.columns.tolist()
    x = np.arange(len(algorithms))
    width = 0.8 / len(prio_algs)

    # Map each prioritization_algorithm to a consistent bar offset and color
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(prio_algs)]

    # Plot delivery rates
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, prio in enumerate(prio_algs):
        ax.bar(x + i * width, delivery_pct[prio], width, label=prio, color=colors[i])
    ax.set_ylabel('Delivery Rate (%)')
    ax.set_title('Percentage of Diseases with Delivered Results' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''))
    ax.set_xticks(x + width * (len(prio_algs) - 1) / 2)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(title='Prioritization Algorithm', loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'delivery_rate_barplot.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Plot significant delivery rates
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, prio in enumerate(prio_algs):
        ax.bar(x + i * width, sig_pct[prio], width, label=prio, color=colors[i])
    ax.set_ylabel(f'Significant Rate (%) (p < {significance_threshold})')
    ax.set_title('Percentage of Diseases with Significant Results' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''))
    ax.set_xticks(x + width * (len(prio_algs) - 1) / 2)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(title='Prioritization Algorithm', loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'significant_rate_barplot.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Plot significant rate among delivered results
    conditional_pct = (sig_counts / counts * 100).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, prio in enumerate(prio_algs):
        ax.bar(x + i * width, conditional_pct[prio], width, label=prio, color=colors[i])
    ax.set_ylabel(f'Significant Rate Among Delivered (%) (p < {significance_threshold})')
    ax.set_title(
        'Percentage of Delivered Diseases with Significant Results' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''))
    ax.set_xticks(x + width * (len(prio_algs) - 1) / 2)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(title='Prioritization Algorithm', loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'conditional_significant_rate_barplot.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_minimum_significance_by_category(df: pd.DataFrame, plots_dir: str,
                                          significance_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL) -> None:
    """
    Plot three bar charts showing, for each algorithm and each prioritization algorithm separately:
    1) The percentage of diseases with p_value_DCG < significance_threshold by taking the minimum
       p-value across the other axis.
    2) An overall bar using the minimum p-value across all algorithm/prioritization combinations for
       each disease.
    3) The percentage from (1) divided by the number of delivered results for each category.
    """
    total_diseases = df['sample'].nunique()
    results = {}

    # Percentage by algorithm: pick row with min p per sample within the algorithm, then check 'significant'
    for algo in sorted(df['algorithm'].unique()):
        sub = df[df['algorithm'] == algo]
        if sub.empty:
            results[algo] = 0.0
            continue
        idx = sub.groupby('sample')['p_value_DCG'].idxmin()
        min_rows = sub.loc[idx]
        results[algo] = min_rows['significant'].sum() / total_diseases * 100

    # Percentage by prioritization algorithm: pick row with min p per sample within the prio, then check 'significant'
    for prio in sorted(df['prioritization_algorithm'].unique()):
        sub = df[df['prioritization_algorithm'] == prio]
        if sub.empty:
            results[prio] = 0.0
            continue
        idx = sub.groupby('sample')['p_value_DCG'].idxmin()
        min_rows = sub.loc[idx]
        results[prio] = min_rows['significant'].sum() / total_diseases * 100

    # Overall percentage: min p per sample across all combinations, then check 'significant'
    idx_all = df.groupby('sample')['p_value_DCG'].idxmin()
    min_rows_all = df.loc[idx_all]
    results['Total'] = min_rows_all['significant'].sum() / total_diseases * 100

    # Prepare for plotting
    labels = list(results.keys())
    values = list(results.values())
    x = np.arange(len(labels))

    # Assign colors
    algo_labels = sorted(df['algorithm'].unique())
    prio_labels = sorted(df['prioritization_algorithm'].unique())
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_alg = cycle[0]
    color_prio = cycle[1] if len(cycle) > 1 else cycle[0]
    color_total = cycle[2] if len(cycle) > 2 else cycle[-1]
    bar_colors = [color_alg if lbl in algo_labels else color_prio if lbl in prio_labels else color_total for lbl in
                  labels]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f'Significance Rate (%) (p < {significance_threshold})')
    ax.set_title('Minimum p-value Significance Rates by Category' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''))

    # Legend
    legend_handles = [Patch(facecolor=color_alg, label='Algorithm'),
                      Patch(facecolor=color_prio, label='Prioritization Algorithm'),
                      Patch(facecolor=color_total, label='Total')]
    ax.legend(handles=legend_handles, loc='best')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'min_significance_by_category_barplot.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Conditional minimum significance rates by category
    delivered_per_algo = df.groupby('algorithm')['sample'].nunique()
    delivered_per_prio = df.groupby('prioritization_algorithm')['sample'].nunique()
    conditional_results = {}

    for algo in sorted(df['algorithm'].unique()):
        sub = df[df['algorithm'] == algo]
        if sub.empty:
            conditional_results[algo] = 0.0
            continue
        idx = sub.groupby('sample')['p_value_DCG'].idxmin()
        min_rows = sub.loc[idx]
        count_sig = min_rows['significant'].sum()
        denom = delivered_per_algo.get(algo, 1)
        conditional_results[algo] = count_sig / denom * 100

    for prio in sorted(df['prioritization_algorithm'].unique()):
        sub = df[df['prioritization_algorithm'] == prio]
        if sub.empty:
            conditional_results[prio] = 0.0
            continue
        idx = sub.groupby('sample')['p_value_DCG'].idxmin()
        min_rows = sub.loc[idx]
        count_sig = min_rows['significant'].sum()
        denom = delivered_per_prio.get(prio, 1)
        conditional_results[prio] = count_sig / denom * 100

    # Keep the original denominator behavior for Total
    conditional_results['Total'] = min_rows_all['significant'].sum() / total_diseases * 100

    labels_cond = list(conditional_results.keys())
    values_cond = [conditional_results[l] for l in labels_cond]
    x_cond = np.arange(len(labels_cond))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_cond, values_cond, color=bar_colors)
    ax.set_xticks(x_cond)
    ax.set_xticklabels(labels_cond, rotation=45, ha='right')
    ax.set_ylabel(f'Conditional Minimum p-value Significance Rate (%) (p < {significance_threshold})')
    ax.set_title('Minimum p-value Significance Rates by Category (Conditional on Delivered Results)' + (
        f' ({NETWORK_NAME})' if NETWORK_NAME else ''))
    ax.legend(handles=legend_handles, loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'conditional_min_significance_by_category_barplot.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_faceted_violins(df: pd.DataFrame, plots_dir: str, facet_by_network: bool = True,
                         no_show: bool = False) -> None:
    """
    Faceted small multiples of violin plots.
    Columns = main algorithms; within each subplot, three colored violins for prioritization algorithms.
    If a 'network' column exists and facet_by_network=True, create one row per network; otherwise a single row.
    Saves a PDF named `violin_faceted_by_algorithm[ _and_network].pdf`.
    """
    algorithms = sorted(df['algorithm'].unique())
    prio_algs = sorted(df['prioritization_algorithm'].unique())
    has_network = facet_by_network and ('network' in df.columns)
    networks = sorted(df['network'].unique()) if has_network else [None]

    # Add one extra combined facet: Total & Best
    facet_names = algorithms + ["Total & Best"]
    n_rows = len(networks)
    n_cols = len(facet_names)
    # scalable figure size
    fig_w = max(12, 3 * n_cols)
    fig_h = max(4, 2.8 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharey=True)

    # consistent colors for prioritization algorithms
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(prio_algs)]
    color_map = {p: colors[i] for i, p in enumerate(prio_algs)}

    for r, net in enumerate(networks):
        df_net = df if net is None else df[df['network'] == net]
        for c, facet in enumerate(facet_names):
            ax = axes[r, c]
            if facet in algorithms:
                # Prioritization violins for this algorithm
                data = [df_net[(df_net['algorithm'] == facet) & (df_net['prioritization_algorithm'] == p)][
                            'p_value_DCG'].dropna() for p in prio_algs]
                positions = np.arange(len(prio_algs))
                parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.8)
                # Color as before
                for j, body in enumerate(parts['bodies']):
                    body.set_facecolor(color_map[prio_algs[j]])
                    body.set_edgecolor('black')
                    body.set_alpha(0.7)
            elif facet == "Total & Best":
                # Combined facet for Total and Best
                total = df_net['p_value_DCG'].dropna()
                best = df_net.groupby('sample')['p_value_DCG'].min()
                data = [total, best]
                positions = [0, 1]
                parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, widths=0.8)
                # Style both violins white with black edge
                for body in parts['bodies']:
                    body.set_facecolor('white')
                    body.set_edgecolor('black')
                    body.set_alpha(0.9)
                # X-axis labels for combined facet
                ax.set_xticks(positions)
                ax.set_xticklabels(["Total", "Best"], rotation=45, ha='right', fontsize=9)
            # Style median line
            if 'cmedians' in parts:
                parts['cmedians'].set_color('black')
                parts['cmedians'].set_linewidth(2)
            # X-axis labels for algorithms
            if facet in algorithms:
                ax.set_xticks(positions)
                ax.set_xticklabels(prio_algs, rotation=45, ha='right', fontsize=9)
            # Titles and y-labels
            if r == 0:
                ax.set_title(facet, pad=8)
            if c == 0:
                ylabel = 'p_value_DCG' if net is None else f"{net}\n(p_value_DCG)"
                ax.set_ylabel(ylabel)

    # Common legend
    handles = [Patch(facecolor=color_map[p], edgecolor='black', label=p, alpha=0.7) for p in prio_algs]
    handles += [Patch(facecolor='white', edgecolor='black', label='Total', alpha=0.9),
                Patch(facecolor='white', edgecolor='black', label='Best', alpha=0.9)]
    fig.legend(handles=handles, loc='upper right', title='Prioritization')
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.suptitle('p_value_DCG distributions by algorithm and prioritization (with Total & Best per algorithm)' + (
        '' if not has_network else ' and network') + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), y=0.995,
                 fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])

    out_name = 'violin_faceted_by_algorithm.pdf' if not has_network else 'violin_faceted_by_algorithm_and_network.pdf'
    fig.savefig(os.path.join(plots_dir, out_name), bbox_inches='tight')
    if not no_show:
        plt.show()


def main():
    args = parse_arguments()
    # Suppress plot display if --no-show is set
    if args.no_show:
        global SHOW_PLOTS
        SHOW_PLOTS = False

    results_df = read_input_tsv(args.input_tsv)
    significance_level = args.significance_level

    # Determine/validate network context
    global NETWORK_NAME
    results_df, NETWORK_NAME = _apply_network_selection(results_df, args.network)

    disorders_df = pd.read_csv(args.disorders_csv, usecols=["primaryDomainId", "displayName"])
    id_to_name = dict(zip(disorders_df["primaryDomainId"], disorders_df["displayName"]))

    plots_dir = os.path.join(args.plots_dir, NETWORK_NAME)
    logger.info("Writing plots to: %s", plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    ignore_cols = ['dcg_exceed_count', 'observed_DCG', 'observed_overlap', 'overlap_exceed_count', 'jaccard_median',
                   'seed_specificity_median', 'seed_specificity_adjusted_median', 'num_drugs', 'num_targets',
                   'num_approved_drugs', 'p_value_bh']
    plot_correlation_matrix(results_df, plots_dir, "Correlation Heatmap of Numeric Metrics",
                            "prioritization_evaluation_correlation_heatmap.pdf", ignore_cols)
    plot_correlation_matrix(results_df[results_df['p_value_DCG'] < 1], plots_dir,
                            "Correlation Heatmap of Numeric Metrics, excluding p_value_DCG = 1",
                            "prioritization_evaluation_correlation_heatmap_excluding_p_value_DCG_1.pdf", ignore_cols)

    # Find all diseases directly below human
    # Load in parents from argument
    parents_df = pd.read_csv(args.disorder_is_subtype_of_disorder, usecols=["sourceDomainId", "targetDomainId"])
    children = parents_df.loc[
        parents_df["targetDomainId"] == HUMAN_DISEASE_MONDO, "sourceDomainId"
    ].tolist()

    # ensure one row per sample and only needed columns
    df = results_df.drop_duplicates(subset="sample")[["sample", "parents"]].copy()

    # Robustly parse list-like strings in `parents` into real Python lists
    # Handles formats like "['mondo.0002254', 'mondo.0006510']", JSON lists, and common nulls
    def _parse_parents_cell(v):
        if isinstance(v, list):
            return v
        if pd.isna(v):
            return []
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null", "[]"}:
            return []
        # Try safe Python literal, then JSON
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(s)
                if isinstance(out, (list, tuple, set)):
                    return [str(x).strip().strip('"').strip("'") for x in list(out) if str(x).strip()]
            except Exception:
                pass
        # Fallback: extract quoted tokens like 'mondo.XXXX' or "mondo.XXXX"
        toks = re.findall(r"[\"']\s*([^\"']+?)\s*[\"']", s)
        if toks:
            return [t.strip() for t in toks if t.strip()]
        # Last resort: strip brackets and split by comma
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(',') if p.strip()]
            return parts
        return [s]

    df["parents"] = df["parents"].apply(_parse_parents_cell)
    logger.info("parents parsed types: %s", df["parents"].map(type).value_counts().to_dict())

    # --- Propagate to full ancestors: build parents_complete ---
    # Build graph: child -> set(immediate parents)
    parent_map = parents_df.groupby("sourceDomainId")["targetDomainId"].apply(set).to_dict()

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _all_ancestors(node: str) -> set:
        acc = set()
        for par in parent_map.get(node, set()):
            acc.add(par)
            acc |= _all_ancestors(par)
        return acc

    def _expand_parents_to_ancestors(plist):
        if not isinstance(plist, (list, tuple, set)):
            return []
        out = set()
        for p in plist:
            if not p:
                continue
            out.add(p)
            out |= _all_ancestors(str(p))
        return sorted(out)

    df["parents_complete"] = df["parents"].apply(_expand_parents_to_ancestors)
    # Some quick diagnostics
    try:
        sizes = df["parents_complete"].map(len)
        logger.info(
            "parents_complete sizes: min=%s, median=%s, max=%s",
            int(sizes.min()) if len(sizes) else 0,
            int(sizes.median()) if len(sizes) else 0,
            int(sizes.max()) if len(sizes) else 0,
        )
    except Exception:
        pass

    # Build category membership using the COMPLETE ancestor lists
    parent_to_samples = (
        df.explode("parents_complete")
        .dropna(subset=["parents_complete"])  # drop NaNs from empty ancestor lists
        .query("parents_complete in @children")
        .groupby("parents_complete")["sample"]
        .unique()
        .apply(list)
        .reindex(children, fill_value=[])
        .to_dict()
    )

    # Diagnostics for category coverage and anchor sanity
    try:
        union_size = len(set().union(*parent_to_samples.values())) if parent_to_samples else 0
    except TypeError:
        union_size = 0
    sum_lengths = sum(len(v) for v in parent_to_samples.values())
    logger.info(
        "category coverage: union_unique_diseases=%d, sum_over_categories=%d",
        union_size, sum_lengths
    )

    total_children = len(parent_to_samples)
    non_empty_sizes = [len(v) for v in parent_to_samples.values() if v]
    num_with_samples = len(non_empty_sizes)
    num_empty = total_children - num_with_samples
    avg_excluding_zero = (sum(non_empty_sizes) / num_with_samples) if num_with_samples else 0.0
    logger.info(
        "children_total=%d, with_samples=%d, empty=%d, avg_samples_excluding_zero=%.2f",
        total_children,
        num_with_samples,
        num_empty,
        avg_excluding_zero,
    )
    plot_category_significance_by_children(
        results_df,
        children,
        parent_to_samples,
        plots_dir,
        id_to_name=id_to_name,
    )

    # Best per sample across all rows
    idx_best_all = results_df.groupby('sample')['p_value_DCG'].idxmin()
    best_all_for_corr = results_df.loc[idx_best_all]

    plot_correlation_matrix(
        best_all_for_corr, plots_dir,
        "Correlation Heatmap of Numeric Metrics (best per sample)",
        "prioritization_evaluation_correlation_heatmap_best_per_sample.pdf",
        ignore_cols,
    )

    # Best per sample within p_value_DCG < 1
    df_lt1 = results_df[results_df['p_value_DCG'] < 1]
    idx_best_lt1 = df_lt1.groupby('sample')['p_value_DCG'].idxmin()
    best_lt1 = df_lt1.loc[idx_best_lt1]

    plot_correlation_matrix(
        best_lt1, plots_dir,
        "Correlation Heatmap of Numeric Metrics (best per sample, excluding p_value_DCG = 1)",
        "prioritization_evaluation_correlation_heatmap_best_per_sample_excluding_p_value_DCG_1.pdf",
        ignore_cols,
    )

    # Compute BH-like significance cutoff once: max p_value_DCG among rows marked significant
    bh_sig_cutoff = None
    if 'significant' in results_df.columns:
        sig_rows = results_df[results_df['significant']]
        if not sig_rows.empty:
            bh_sig_cutoff = float(sig_rows['p_value_DCG'].max())
            logger.info("BH-like cutoff (max p where significant=True): %.6g", bh_sig_cutoff)
        else:
            logger.info("No rows with significant==True; BH-like cutoff not set.")

    plot_cdf_by_stratify(results_df, column='p_value_DCG', title='ECDF of DCG p-values',
                         save_path=f"{plots_dir}/cdf_p_value_DCG_overall.pdf",
                         significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    # 1) Summary: pick the best p<1 per sample, then tabulate
    df_lt1 = results_df[results_df['p_value_DCG'] < 1]
    idx = df_lt1.groupby('sample')['p_value_DCG'].idxmin()
    best_lt1 = df_lt1.loc[idx]

    summary = (best_lt1.groupby(['algorithm', 'prioritization_algorithm']).size().reset_index(name='count'))
    summary['percent'] = summary['count'] / summary['count'].sum() * 100
    # Sort by percent descending
    summary = summary.sort_values(by='percent', ascending=False)
    print(summary)

    # 2) Distribution: pick the best per sample (including p=1) and plot
    idx_all = results_df.groupby('sample')['p_value_DCG'].idxmin()
    best_all = results_df.loc[idx_all]

    plot_cdf_by_stratify(best_all, column='p_value_DCG',
                         title='ECDF of DCG p-values (best per sample, including p=1)',
                         save_path=f"{plots_dir}/cdf_p_value_DCG_overall_only_best.pdf",
                         significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_cdf_by_stratify(results_df, column='p_value_DCG', stratify='algorithm',
                         title='ECDF of DCG p-values by algorithm',
                         save_path=f"{plots_dir}/cdf_p_value_DCG_by_algorithm.pdf",
                         significance_level=significance_level, bh_sig_level=bh_sig_cutoff,
                         overlay_values=best_all['p_value_DCG'],
                         overlay_label='Best per sample (min across all)')

    plot_cdf_by_stratify(results_df, column='p_value_DCG', stratify='prioritization_algorithm',
                         title='ECDF of DCG p-values by prioritization_algorithm',
                         save_path=f"{plots_dir}/cdf_p_value_DCG_by_prioritization_algorithm.pdf",
                         significance_level=significance_level, bh_sig_level=bh_sig_cutoff,
                         overlay_values=best_all['p_value_DCG'],
                         overlay_label='Best per sample (min across all)')

    plot_cdf_by_stratify(results_df, column='p_value_DCG', stratify=['algorithm', 'prioritization_algorithm'],
                         title='ECDF of DCG p-values by algorithm and prioritization_algorithm',
                         save_path=f"{plots_dir}/cdf_p_value_DCG_by_algorithm_and_prioritization_algorithm.pdf",
                         significance_level=significance_level, bh_sig_level=bh_sig_cutoff,
                         overlay_values=best_all['p_value_DCG'],
                         overlay_label='Best per sample (min across all)')

    # Absolute cumulative-count variants of the p-value distributions
    plot_cumcount_by_stratify(results_df, column='p_value_DCG',
                              title='Cumulative count of DCG p-values',
                              save_path=f"{plots_dir}/cumcount_p_value_DCG_overall.pdf",
                              significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_cumcount_by_stratify(best_all, column='p_value_DCG',
                              title='Cumulative count of DCG p-values (best per sample, including p=1)',
                              save_path=f"{plots_dir}/cumcount_p_value_DCG_overall_only_best.pdf",
                              significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_cumcount_by_stratify(results_df, column='p_value_DCG', stratify='algorithm',
                              title='Cumulative count of DCG p-values by algorithm',
                              save_path=f"{plots_dir}/cumcount_p_value_DCG_by_algorithm.pdf",
                              significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_cumcount_by_stratify(results_df, column='p_value_DCG', stratify='prioritization_algorithm',
                              title='Cumulative count of DCG p-values by prioritization_algorithm',
                              save_path=f"{plots_dir}/cumcount_p_value_DCG_by_prioritization_algorithm.pdf",
                              significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_cumcount_by_stratify(results_df, column='p_value_DCG', stratify=['algorithm', 'prioritization_algorithm'],
                              title='Cumulative count of DCG p-values by algorithm and prioritization_algorithm',
                              save_path=f"{plots_dir}/cumcount_p_value_DCG_by_algorithm_and_prioritization_algorithm.pdf",
                              significance_level=significance_level, bh_sig_level=bh_sig_cutoff)

    plot_distribution_by_stratify(results_df, column='observed_DCG', title='KDE of Observed DCG',
                                  save_path=f"{plots_dir}/kde_observed_DCG_overall.pdf")

    plot_distribution_by_stratify(results_df, column='percent_true_drugs_found', stratify='algorithm',
                                  title='KDE of percent true drugs found by algorithm',
                                  save_path=f"{plots_dir}/kde_percent_true_drugs_found_by_algorithm.pdf", xlim=(0, 100))

    plot_distribution_by_stratify(results_df, column='percent_true_drugs_found', stratify='prioritization_algorithm',
                                  title='KDE of percent true drugs found by prioritization_algorithm',
                                  save_path=f"{plots_dir}/kde_percent_true_drugs_found_by_prioritization_algorithm.pdf",
                                  xlim=(0, 100))

    plot_distribution_by_stratify(results_df, column='percent_true_drugs_found',
                                  stratify=['algorithm', 'prioritization_algorithm'],
                                  title='KDE of percent true drugs found by algorithm and prioritization_algorithm',
                                  save_path=f"{plots_dir}/kde_percent_true_drugs_found_by_algorithm_and_prioritization_algorithm.pdf",
                                  xlim=(0, 100))

    plot_distribution_by_stratify(results_df.query("percent_true_drugs_found != 0"), column='percent_true_drugs_found',
                                  stratify=['algorithm', 'prioritization_algorithm'],
                                  title='KDE of percent true drugs found by algorithm and prioritization_algorithm, excluding 0',
                                  save_path=f"{plots_dir}/kde_percent_true_drugs_found_by_algorithm_and_prioritization_algorithm_excluding_0.pdf",
                                  xlim=(0, 100))

    plot_faceted_violins(results_df, plots_dir, facet_by_network=True, no_show=args.no_show)
    # Recovered drugs (overlap) vs DCG p-value — include all rows
    plot_scatter(
        results_df,
        plots_dir,
        x_col='observed_overlap',
        y_col='p_value_DCG',
        title='Recovered drugs (overlap) vs DCG p-value',
        filename='overlap_vs_p_value_DCG.pdf'
    )

    plot_hexbin_log(
        results_df,
        plots_dir,
        x_col='observed_overlap',
        y_col='p_value_DCG',
        title='Hexbin of recovered drugs (overlap) vs DCG p-value (log density)',
        filename='hexbin_overlap_vs_p_value_DCG.pdf',
        gridsize=50,
        mincnt=1
    )
    logger.info("Plotted overlap vs p_value_DCG")
    plot_scatter(results_df, plots_dir, 'num_drugs', 'p_value_DCG', "numdrugs vs p value DCG",
                 "num_drugs_vs_p_value_DCG.pdf")
    plot_hexbin(results_df, plots_dir, x_col='num_drugs', y_col='p_value_DCG',
                title='Hexbin of num_drugs vs p_value_DCG', filename='hexbin_num_drugs_vs_p_value_DCG.pdf')
    # Jaccard index plots for different p_value_DCG thresholds
    # Jaccard index plots
    subset_df = results_df[results_df['p_value_DCG'] < 1]
    sets_dict = get_disease_sets_by_algo_combination(subset_df)
    plot_jaccard_matrix_from_sets(
        sets_dict,
        plots_dir,
        title="Jaccard Index Matrix (p_value_DCG < 1)",
        filename="jaccard_p_value_DCG_lt1.pdf",
    )

    subset_sig = results_df[results_df['significant']]
    sets_dict_sig = get_disease_sets_by_algo_combination(subset_sig)
    plot_jaccard_matrix_from_sets(
        sets_dict_sig,
        plots_dir,
        title="Jaccard Index Matrix (significant results)",
        filename="jaccard_significant.pdf",
    )

    plot_delivery_and_significance(results_df, plots_dir, significance_threshold=significance_level)
    plot_minimum_significance_by_category(results_df, plots_dir, significance_threshold=significance_level)
    plot_stacked_delivery_breakdown(results_df, plots_dir, significance_threshold=significance_level)

    # Example usage (uncomment and customize):

    p_val_comparison = ["seeds", "nodes", "edges", "components", "num_parents", "diameter", "num_children",
                        "seed_entropy", "seed_specificity_mean", "seed_specificity_adjusted_mean", "jaccard_max",
                        "jaccard_mean", "candidate_count", "p_value_without_ranks", "approved_drugs_with_targets",
                        "percent_true_drugs_found"]
    percent_true_comparison = ["seeds", "nodes", "edges", "components", "num_parents", "diameter", "num_children",
                               "seed_entropy", "seed_specificity_mean", "seed_specificity_adjusted_mean", "jaccard_max",
                               "jaccard_mean", "candidate_count", "p_value_DCG", "approved_drugs_with_targets",
                               "largest_component"]
    # P value DCG vs predictors, excluding p_value = 1
    plot_scatter_grid(results_df.query("p_value_DCG < 1"), y_col="p_value_DCG", x_cols=p_val_comparison,
                      plots_dir=plots_dir, title="Key outcomes vs predictors, excluding p_value = 1",
                      filename="scatter_grid_p_value_DCG_vs_predictors_excluding_1.pdf")
    plot_hexbin_grid_log(results_df.query("p_value_DCG < 1"), y_col="p_value_DCG", x_cols=p_val_comparison,
                         plots_dir=plots_dir,
                         title="Hexbin density (log) of outcomes vs predictors, excluding p_value = 1", gridsize=50,
                         mincnt=1, filename="hexbin_grid_log_p_value_DCG_vs_predictors_excluding_1.pdf")
    # P value DCG vs predictors, including p_value = 1
    plot_scatter_grid(results_df, y_col="p_value_DCG", x_cols=p_val_comparison, plots_dir=plots_dir,
                      title="Key outcomes vs predictors", filename="scatter_grid_p_value_DCG_vs_predictors.pdf")
    plot_hexbin_grid_log(results_df, y_col="p_value_DCG", x_cols=p_val_comparison, plots_dir=plots_dir,
                         title="Hexbin density (log) of outcomes vs predictors", gridsize=50, mincnt=1,
                         filename="hexbin_grid_log_p_value_DCG_vs_predictors.pdf")

    # Percent true drugs found vs predictors, excluding 0 true drugs found
    plot_scatter_grid(results_df.query("percent_true_drugs_found > 0"), y_col="percent_true_drugs_found",
                      x_cols=percent_true_comparison, plots_dir=plots_dir,
                      title="Key outcomes vs predictors, excluding 0 true drugs found",
                      filename="scatter_grid_percent_true_drugs_found_vs_predictors_excluding_0.pdf")
    plot_hexbin_grid_log(results_df.query("percent_true_drugs_found > 0"), y_col="percent_true_drugs_found",
                         x_cols=percent_true_comparison, plots_dir=plots_dir,
                         title="Hexbin density (log) of outcomes vs predictors, excluding 0 true drugs found",
                         gridsize=50, mincnt=1,
                         filename="hexbin_grid_log_percent_true_drugs_found_vs_predictors_excluding_0.pdf")
    # Percent true drugs found vs predictors, including 0 true drugs found
    plot_scatter_grid(results_df, y_col="percent_true_drugs_found", x_cols=percent_true_comparison, plots_dir=plots_dir,
                      title="Key outcomes vs predictors",
                      filename="scatter_grid_percent_true_drugs_found_vs_predictors.pdf")
    plot_hexbin_grid_log(results_df, y_col="percent_true_drugs_found", x_cols=percent_true_comparison,
                         plots_dir=plots_dir, title="Hexbin density (log) of outcomes vs predictors", gridsize=50,
                         mincnt=1,
                         filename="hexbin_grid_log_percent_true_drugs_found_vs_predictors.pdf")


def plot_stacked_delivery_breakdown(df: pd.DataFrame, plots_dir: str,
                                    significance_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL) -> None:
    """
    Stacked bars per algorithm, with one bar per prioritization_algorithm.
    Stacks (bottom to top), as % of all diseases:
      1) Significant (from df['significant'] and candidate_count > 0)
      2) Delivered but non-significant (candidate_count > 0)
      3) Delivered with candidate_count == 0
      4) Not delivered for that algorithm/prioritization pair

    Visuals:
      - Colors encode segments (legend: right, top)
      - Hatches encode prioritization (legend: right, below segments)
      - First prioritization uses solid fill (no hatch)
      - Small spacing between bars within each algorithm group
      - Y ticks every 10%
    """
    total_diseases = df['sample'].nunique()

    algorithms = sorted(df['algorithm'].unique())
    prios = sorted(df['prioritization_algorithm'].unique())

    # Containers
    sig_pct = {p: np.zeros(len(algorithms)) for p in prios}
    delivered_non_sig_gt0_pct = {p: np.zeros(len(algorithms)) for p in prios}
    cand0_pct = {p: np.zeros(len(algorithms)) for p in prios}
    not_delivered_pct = {p: np.zeros(len(algorithms)) for p in prios}

    # Aggregate one representative row per disease-combination
    for ai, algo in enumerate(algorithms):
        for p in prios:
            sub = df[(df['algorithm'] == algo) & (df['prioritization_algorithm'] == p)]
            if sub.empty:
                not_delivered_pct[p][ai] = 100.0
                continue

            delivered = sub['sample'].nunique()
            not_delivered = max(0, total_diseases - delivered)

            cand0 = (sub['candidate_count'] == 0).sum()
            delivered_gt0 = max(0, delivered - cand0)

            sig = (sub['significant']).sum()
            non_sig_gt0 = max(0, delivered_gt0 - sig)

            sig_pct[p][ai] = sig / total_diseases * 100.0
            delivered_non_sig_gt0_pct[p][ai] = non_sig_gt0 / total_diseases * 100.0
            cand0_pct[p][ai] = cand0 / total_diseases * 100.0
            not_delivered_pct[p][ai] = not_delivered / total_diseases * 100.0

    # Plot
    fig, ax = plt.subplots(figsize=(18, 8))
    # Reserve a generous right margin for two legends so nothing is cropped
    fig.subplots_adjust(left=0.08, right=0.60, bottom=0.14, top=0.92)

    x = np.arange(len(algorithms))
    total_width = 0.8
    bar_spacing = 0.05
    width = (total_width - bar_spacing * (len(prios) - 1)) / len(prios)

    # Colors per segment (color legend must stay visible)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col_sig = colors[0]
    col_deliv_non_sig = colors[1] if len(colors) > 1 else colors[0]
    col_cand0 = colors[2] if len(colors) > 2 else colors[-1]
    col_not_delivered = colors[3] if len(colors) > 3 else colors[-1]

    # Hatch patterns per prioritization algorithm; first is solid
    hatch_cycle = ['', '///', '\\\\\\', 'xx', '..', '++', '--', '||', 'oo', '**']
    prio_hatch = {p: hatch_cycle[i % len(hatch_cycle)] for i, p in enumerate(prios)}

    # Draw bars
    edge_kw = dict(edgecolor='black', linewidth=0.6)
    for i, p in enumerate(prios):
        xpos = x - total_width / 2 + i * (width + bar_spacing) + width / 2
        b1 = sig_pct[p]
        b2 = b1 + delivered_non_sig_gt0_pct[p]
        b3 = b2 + cand0_pct[p]
        hatch = prio_hatch[p]

        ax.bar(xpos, sig_pct[p], width, color=col_sig, hatch=hatch, **edge_kw)
        ax.bar(xpos, delivered_non_sig_gt0_pct[p], width, bottom=b1, color=col_deliv_non_sig, hatch=hatch, **edge_kw)
        ax.bar(xpos, cand0_pct[p], width, bottom=b2, color=col_cand0, hatch=hatch, **edge_kw)
        ax.bar(xpos, not_delivered_pct[p], width, bottom=b3, color=col_not_delivered, hatch=hatch, **edge_kw)

    # Axes, ticks, labels
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_ylabel('Percentage of diseases (%)')
    ax.set_title(
        'Delivery breakdown by algorithm and prioritization (stacked)' + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''))

    # Legend 1: Segments (right, top) - colored patches
    seg_handles = [Patch(facecolor=col_sig, edgecolor='black', linewidth=0.6, label='Significant'),
                   Patch(facecolor=col_deliv_non_sig, edgecolor='black', linewidth=0.6,
                         label='Delivered non-significant'),
                   Patch(facecolor=col_cand0, edgecolor='black', linewidth=0.6, label='Candidate count = 0'),
                   Patch(facecolor=col_not_delivered, edgecolor='black', linewidth=0.6, label='Not delivered'), ]
    fig.legend(handles=seg_handles, loc='upper left', bbox_to_anchor=(0.62, 0.92), borderaxespad=0.0, title='Segments')

    # Legend 2: Prioritization (right, below segments) - hatch keys
    prio_handles = []
    for idx, p in enumerate(prios):
        if idx == 0:
            prio_handles.append(Patch(facecolor='0.85', edgecolor='black', linewidth=0.6, label=p))
        else:
            prio_handles.append(
                Patch(facecolor='white', edgecolor='black', linewidth=0.6, hatch=prio_hatch[p], label=p))
    fig.legend(handles=prio_handles, loc='upper left', bbox_to_anchor=(0.62, 0.48), borderaxespad=0.0,
               title='Prioritization', ncol=1)

    # Save
    plt.savefig(os.path.join(plots_dir, 'stacked_delivery_breakdown.pdf'), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# --- New helper: single-panel hexbin with log-scaled density ---
def plot_hexbin_log(df: pd.DataFrame, plot_dir: str, x_col: str, y_col: str,
                    title: str, filename: str, gridsize: int = 50,
                    mincnt: int = 1, cmap: str = 'viridis') -> None:
    """Hexbin with a log-scaled colorbar for density."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[[x_col, y_col]].dropna()
    if not data.empty:
        y_min, y_max = _infer_y_bounds(data[y_col], y_col)
        ax.set_ylim(y_min, y_max)
        hb = ax.hexbin(
            data[x_col].to_numpy(),
            data[y_col].to_numpy(),
            gridsize=gridsize,
            mincnt=mincnt,
            cmap=cmap,
            norm=LogNorm(vmin=max(1, mincnt))
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('count (log scale)')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title + (f' ({NETWORK_NAME})' if NETWORK_NAME else ''), pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
