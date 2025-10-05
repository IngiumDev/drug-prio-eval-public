#!/usr/bin/env python3
import argparse
import logging
import os
import re

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullLocator  # NEW/UPDATED
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

HEATMAP_TIGHT_RECT = [0, 0, 1, 0.985]
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
INFECTIOUS_DISEASE_PARENT = "mondo.0005550"
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)
SHOW_PLOTS = True

_BASE_MD = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}

# Patch plt.savefig
_orig_plt_savefig = plt.savefig


def _plt_savefig(fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    return _orig_plt_savefig(fname, *a, **k)


plt.savefig = _plt_savefig

# Patch Figure.savefig (covers fig.savefig)
_orig_fig_savefig = Figure.savefig


def _fig_savefig(self, fname, *a, **k):
    meta = {**_BASE_MD, **(k.get("metadata") or {})}
    k["metadata"] = meta
    return _orig_fig_savefig(self, fname, *a, **k)


Figure.savefig = _fig_savefig


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze cross-network results",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--input_tsv", default="../../data/input/cross_network_data.tsv.gz",
                        help="Path to the input TSV file containing run data.")
    parser.add_argument("--plots_dir", default="../../plots/cross_network", help="Directory to save analysis results.")
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='Suppress display of plots; only save them to files.')
    parser.add_argument("--networks", type=str, default=None,
                        help="Comma-separated list of networks to analyze. If not provided, all networks will be analyzed.")
    parser.add_argument("--module_algorithms", type=str,
                        default="diamond,domino,firstneighbor,no_tool,robust,robust_bias_aware,rwr",
                        help="Comma-separated list of algorithms to analyze. Defaults to 'diamond,domino,firstneighbor,no_tool,robust,robust_bias_aware,rwr'.")
    parser.add_argument("--prioritization_algorithms", type=str, default="trustrank,closeness,degree",
                        help="Comma-separated list of prioritization algorithms to analyze. Defaults to 'trustrank,closeness,degree'.")
    parser.add_argument("--significance_level", type=float, default=DEFAULT_SIGNIFICANCE_LEVEL,
                        help=f"Significance level for p-value analysis (default: {DEFAULT_SIGNIFICANCE_LEVEL}).")
    parser.add_argument("--skip_regression", action='store_true', default=False, help='Skip Logistic Regression')
    parser.add_argument("--dist_columns", type=str,
                        default="seed_entropy,seed_specificity_adjusted_mean,seed_specificity_mean,jaccard_max,jaccard_mean,num_parents",
                        help="Comma-separated columns to plot distributions for (KDE and ECDF).", )
    parser.add_argument("--skip_dist_plots", action='store_true', default=False,
                        help="Skip the KDE/ECDF distribution grids.", )
    args = parser.parse_args()
    if args.no_show:
        global SHOW_PLOTS
        SHOW_PLOTS = False

    return args


def process_significance_level(results_df, significance_level):
    # If significant column is not present, create it based on p_value_dcg
    if 'significant' not in results_df.columns:
        results_df['significant'] = results_df['p_value_DCG'] < significance_level
        logger.info("Created 'significant' column based on p_value_dcg and significance level %s", significance_level)
    else:
        logger.info("'significant' column already present in data; no changes made.")


def read_input_tsv(path):
    """
    Read a TSV or gzipped TSV and return a pandas DataFrame.
    Logs file detection, row/column counts, and errors.
    """
    if not os.path.exists(path):
        logger.error("Input file not found: %s", path)
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        if str(path).lower().endswith('.gz'):
            logger.info("Detected gzip-compressed TSV, reading with gzip: %s", path)
            df = pd.read_csv(path, sep="\t", compression='gzip')
        else:
            logger.info("Detected plain TSV, reading: %s", path)
            df = pd.read_csv(path, sep="\t")
        logger.info("Loaded data from %s: %d rows x %d columns", path, df.shape[0], df.shape[1])
        return df
    except pd.errors.EmptyDataError:
        logger.error("No data: input TSV appears empty: %s", path)
        raise
    except Exception:
        logger.exception("Failed to read input TSV: %s", path)
        raise


def main():
    args = parse_arguments()
    logger.info("Starting analysis with arguments: %s", args)
    # Create plots directory if it doesn't exist

    plots_dir = args.plots_dir
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logger.info("Created plots directory: %s", plots_dir)
    results_df = read_input_tsv(args.input_tsv)

    results_df = remove_irrelevant_rows(results_df, args.networks, args.module_algorithms,
                                        args.prioritization_algorithms)

    # One 'seeds' value per disease (sample), avoiding repeats across methods/networks
    seeds_per_disease = (
        results_df[['sample', 'seeds']].dropna(subset=['seeds']).drop_duplicates(subset=['sample']).set_index('sample')[
            'seeds'].astype(int))

    # List of per-disease seed counts (one entry per disease)
    seed_values = seeds_per_disease.tolist()
    logger.info("Per-disease seed counts (n=%d): %s", len(seed_values), seed_values)

    # Quartiles over per-disease seed counts
    s = pd.Series(seed_values)
    logger.info("Seed counts per disease: min=%d, 25%%=%.1f, median=%.1f, 75%%=%.1f, max=%d", int(s.min()),
                float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75)), int(s.max()))
    process_significance_level(results_df, args.significance_level)

    # Minimal: percent with <=5 seeds among samples with seed_gene_score_mean >0.5 and <=0.5
    ps = results_df.drop_duplicates('sample')[['seeds', 'seed_gene_score_mean']].copy()
    ps['seeds'] = ps['seeds'].astype(int)

    score_boundary = 0.5
    gt = ps[ps['seed_gene_score_mean'] > score_boundary]
    le = ps[ps['seed_gene_score_mean'] <= score_boundary]

    seed_boundary = 5
    pct_gt = (100.0 * int((gt['seeds'] <= seed_boundary).sum()) / len(gt))
    pct_le = (100.0 * int((le['seeds'] <= seed_boundary).sum()) / len(le))

    logger.info(f"seed_gene_score_mean>{score_boundary}: %d/%d (%.2f%%) have <={seed_boundary} seeds",
                int((gt['seeds'] <= seed_boundary).sum()), len(gt),
                pct_gt)
    logger.info(f"seed_gene_score_mean<={score_boundary}: %d/%d (%.2f%%) have <={seed_boundary} seeds",
                int((le['seeds'] <= seed_boundary).sum()), len(le),
                pct_le)

    plot_significance_percentage_heatmap(results_df, args.significance_level, plots_dir)
    plot_conditional_significance_percentage_heatmap(results_df, plots_dir)
    plot_returned_results_percentage_heatmap(results_df, plots_dir)
    plot_significant_given_returned_percentage_heatmap(results_df, plots_dir)
    plot_winrate_min_pvalue_heatmap(results_df, plots_dir)
    plot_global_winrate_heatmap(results_df, plots_dir)
    plot_median_best_ratio_heatmap(results_df, plots_dir)
    plot_median_global_best_ratio_heatmap(results_df, plots_dir)
    plot_rowwise_median_r_heatmap(results_df, plots_dir)
    plot_rowwise_network_winrate_heatmap(results_df, plots_dir)
    plot_seed_stratified_winrate_by_network(results_df, plots_dir)
    plot_seed_stratified_winrate_overall(results_df, plots_dir, show_error='sd_net')
    plot_seed_stratified_percent_significant_overall(results_df, plots_dir, show_error='sd_net')
    plot_minimum_p_value_distribution(results_df, plots_dir)

    # KDE + ECDF grids for per-disease distributions
    if not args.skip_dist_plots:
        # seed_entropy,seed_specificity_adjusted_mean,seed_specificity_mean,jaccard_max,jaccard_mean,num_parents
        dist_cols = ['seed_entropy', 'seed_specificity_adjusted_mean', 'seed_specificity_mean',
                     'jaccard_max', 'jaccard_mean', 'seed_gene_score_mean']
        plot_kde_and_ecdf_grids(
            results_df,
            plots_dir,
            columns=dist_cols,
            unique_on='sample'
        )

        new_cols = ['nodes', 'edges', 'diameter', 'candidate_count', 'num_children', 'components']
        # Only log-scale the ECDF subplot for 'edges' (base-10), zeros remain at 0 via symlog.
        plot_kde_and_ecdf_grids(
            results_df,
            plots_dir,
            columns=new_cols,
            unique_on=['sample', 'ppi', 'algorithm'],
            log_ecdf_cols=['edges'],  # <— only this column gets a log x-axis on the ECDF plot
            log_base=10,  # log10 for readability
            log_zero_lin_thresh=1.0  # keep a linear region around 0 so 0 is shown
        )

    inspect_infectious_diseases(results_df)

    plot_hexbin_regression_grid(
        results_df.query("p_value_DCG < 1 and p_value_without_ranks < 1"),
        plots_dir,
        x_col='p_value_DCG',
        y_col='p_value_without_ranks',
        stratify_cols=('prioritization_algorithm',),
        gridsize=50
    )

    compare_prioritizer_rank_performance(results_df)
    # Compare seed specificity vs size-adjusted specificity with deciles
    quality_measures = ["seed_specificity_mean", "seed_specificity_adjusted_mean", "seed_entropy",
                        "seed_gene_score_mean", "jaccard_mean", "jaccard_max"]
    plot_and_log_percent_sig_vs_quantiles(
        results_df,
        stratify_cols=quality_measures,
        significance_col="significant",
        significance_level=args.significance_level,
        sample_col="sample",
        quantiles=10,
        unit="per_sample",  # collapse to one record per disease
        metric_agg="median",  # aggregate metric across runs for each disease
        sig_rule="any",  # disease is significant if any row is significant
        plot_type="line",  # "line" or "bar"
        show_overall=True,
        plots_dir=plots_dir,
        out_prefix="decile_sig__per_sample_any__spec_vs_spec_adj"
    )

    plot_and_log_percent_sig_vs_quantiles(
        results_df,
        stratify_cols=quality_measures,
        significance_col="significant",
        significance_level=args.significance_level,
        sample_col="sample",
        quantiles=10,
        unit="per_row",  # do not aggregate per disease
        plot_type="line",  # try grouped bars for contrast
        show_overall=True,
        plots_dir=plots_dir,
        out_prefix="decile_sig__per_row__spec_vs_spec_adj"
    )

    # Stratify by nodes
    plot_and_log_percent_sig_vs_quantiles(
        results_df,
        stratify_cols=['nodes', 'seeds', 'edges'],
        significance_col="significant",
        significance_level=args.significance_level,
        sample_col="sample",
        quantiles=10,
        unit="per_row",  # do not aggregate per disease
        plot_type="line",  # try grouped bars for contrast
        show_overall=True,
        plots_dir=plots_dir,
        out_prefix="decile_sig__per_row__nodes"
    )

    if not args.skip_regression:
        significance_col = 'significant'
        sample_col = 'sample'
        network_col = 'ppi'
        algorithm_col = 'algorithm'
        prio_col = 'prioritization_algorithm'

        # With Module infos

        formula_reduced = (f"{significance_col} ~ C({network_col}, Sum)")
        formula_full = (
            f"{significance_col} ~  C({network_col}, Sum) + C(_combo, Sum) + seeds + num_children + seed_gene_score_mean + jaccard_mean + seed_specificity_adjusted_mean + seed_entropy + approved_drugs_with_targets + nodes + edges  + diameter + components + largest_component + isolated_nodes + seeds_not_in_network")
        fit_logistic_models_generic(results_df, plots_dir, y_col=significance_col, sample_col=sample_col,
                                    network_col=network_col, algorithm_col=algorithm_col, prio_col=prio_col,
                                    build_combo_if_needed=True,
                                    formula_reduced=formula_reduced, formula_full=formula_full,
                                    factors_for_implied=['_combo', network_col],
                                    id_tag=f"all_module_columns_no_sample",
                                    add_roc_pr=True, )

        formula_reduced = (f"{significance_col} ~ C({sample_col}, Sum) + C({network_col}, Sum)")
        formula_full = (
            f"{significance_col} ~  C({sample_col}, Sum) + C({network_col}, Sum) + C(_combo, Sum) + seeds + num_children + seed_gene_score_mean + jaccard_mean + seed_specificity_adjusted_mean + seed_entropy + approved_drugs_with_targets + nodes + edges + diameter + components + largest_component + isolated_nodes + seeds_not_in_network")
        fit_logistic_models_generic(results_df, plots_dir, y_col=significance_col, sample_col=sample_col,
                                    network_col=network_col, algorithm_col=algorithm_col, prio_col=prio_col,
                                    build_combo_if_needed=True,
                                    formula_reduced=formula_reduced, formula_full=formula_full,
                                    factors_for_implied=['_combo', network_col],
                                    drop_term_prefixes=(f"C({sample_col}",),
                                    id_tag=f"all_module_columns_with_sample",
                                    add_roc_pr=True, )

        fit_logistic_success_models_methods(results_df, plots_dir)

        formula_reduced = (f"{significance_col} ~ C({network_col}, Sum)")
        formula_full = (
            f"{significance_col} ~  C({network_col}, Sum) + C(_combo, Sum) + seeds + num_children + seed_gene_score_mean + jaccard_mean + seed_specificity_adjusted_mean + seed_entropy + approved_drugs_with_targets")
        fit_logistic_models_generic(results_df, plots_dir, y_col=significance_col, sample_col=sample_col,
                                    network_col=network_col, algorithm_col=algorithm_col, prio_col=prio_col,
                                    build_combo_if_needed=True,
                                    formula_reduced=formula_reduced, formula_full=formula_full,
                                    factors_for_implied=['_combo', network_col],
                                    id_tag=f"pre_module_columns_no_sample",
                                    add_roc_pr=True, )

        formula_reduced = (f"{significance_col} ~ C({sample_col}, Sum) + C({network_col}, Sum)")
        formula_full = (
            f"{significance_col} ~ C({sample_col}, Sum) + C({network_col}, Sum) + C(_combo, Sum) + seeds + num_children + seed_gene_score_mean + jaccard_mean + seed_specificity_adjusted_mean + seed_entropy + approved_drugs_with_targets")
        fit_logistic_models_generic(results_df, plots_dir, y_col=significance_col, sample_col=sample_col,
                                    network_col=network_col, algorithm_col=algorithm_col, prio_col=prio_col,
                                    build_combo_if_needed=True,
                                    formula_reduced=formula_reduced, formula_full=formula_full,
                                    factors_for_implied=['_combo', network_col],
                                    drop_term_prefixes=(f"C({sample_col}",), id_tag=f"pre_module_columns_with_sample",
                                    add_roc_pr=True, )


# ---------- Refactored helper: multi-metric quantile stratification + plot ----------
def plot_and_log_percent_sig_vs_quantiles(
        results_df: pd.DataFrame,
        stratify_cols: list[str],
        *,
        significance_col: str = "significant",
        significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
        sample_col: str = "sample",
        quantiles: int = 10,
        unit: str = "per_sample",  # "per_sample" or "per_row"
        metric_agg: str = "median",  # if unit == "per_sample": "median" | "mean" | "min" | "max"
        sig_rule: str = "any",  # if unit == "per_sample": "any" | "min_p"
        plot_type: str = "line",  # "line" | "bar"
        show_overall: bool = True,  # draw overall % significant as a horizontal line
        plots_dir: str | None = None,
        out_prefix: str | None = None,
):
    """
    Compare multiple metrics by stratifying each into empirical quantile bins and
    computing % significant per bin. Plots one series per metric.

    unit:
      - "per_sample": collapse to one record per disease. Metric is aggregated by `metric_agg`.
                      Significance by `sig_rule`:
                        - "any": True if the disease is significant at least once across rows
                        - "min_p": True if min p_value_DCG < alpha
      - "per_row": no collapsing. Each row contributes once.

    Returns dict with:
      - 'summary'    long DataFrame: metric, bin_idx, pct range, value range, n, n_sig, pct_sig
      - 'cutpoints'  dict[str -> DataFrame] of percentile cutpoints per metric
      - 'overall'    dict with overall n, n_sig, pct_sig (computed at the chosen unit)
      - 'figure_path', 'csv_summary_path'
    """
    df = results_df.copy()

    # Ensure significance indicator
    if significance_col not in df.columns:
        if 'p_value_DCG' not in df.columns:
            raise KeyError(f"Need '{significance_col}' or 'p_value_DCG' to derive significance.")
        df[significance_col] = pd.to_numeric(df['p_value_DCG'], errors='coerce') < float(significance_level)
    else:
        df[significance_col] = pd.to_numeric(df[significance_col], errors='coerce').astype(bool)

    # Validate requested stratify columns exist
    missing_cols = [c for c in stratify_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing stratify columns: {missing_cols}")

    # Build analysis frame based on unit
    if unit == "per_sample":
        # Metric aggregation across rows per sample
        agg_funcs = {
            "median": lambda s: float(np.nanmedian(pd.to_numeric(s, errors='coerce'))),
            "mean": lambda s: float(np.nanmean(pd.to_numeric(s, errors='coerce'))),
            "min": lambda s: float(np.nanmin(pd.to_numeric(s, errors='coerce'))),
            "max": lambda s: float(np.nanmax(pd.to_numeric(s, errors='coerce'))),
        }
        if metric_agg not in agg_funcs:
            raise ValueError("metric_agg must be one of {'median','mean','min','max'}")

        # Aggregate each metric
        metric_aggs = {m: agg_funcs[metric_agg] for m in stratify_cols}

        # Aggregate significance per disease
        if sig_rule == "any":
            sig_agg = 'any'
        elif sig_rule == "min_p":
            if 'p_value_DCG' not in df.columns:
                raise KeyError("sig_rule=='min_p' requires 'p_value_DCG' column.")
            # compute at disease-level: min p < alpha
            df['_p_lt_alpha'] = pd.to_numeric(df['p_value_DCG'], errors='coerce') < float(significance_level)
            sig_agg = '_p_lt_alpha'
        else:
            raise ValueError("sig_rule must be 'any' or 'min_p'")

        # Build aggregation dict
        agg_dict = {**metric_aggs, significance_col: ('any' if sig_rule == 'any' else (sig_agg, 'any'))}
        # pandas agg signature differences handled explicitly
        grp = df.groupby(sample_col, as_index=False)
        # Aggregate metrics
        metrics_df = grp[stratify_cols].agg({m: agg_funcs[metric_agg] for m in stratify_cols})
        # Aggregate significance
        if sig_rule == 'any':
            sig_df = grp[significance_col].any().rename(columns={significance_col: significance_col})
        else:
            sig_df = grp['_p_lt_alpha'].any().rename(columns={'_p_lt_alpha': significance_col})

        ana = metrics_df.merge(sig_df, on=sample_col, how='inner')

    elif unit == "per_row":
        # Use rows as is
        # Cast metrics to numeric
        for m in stratify_cols:
            df[m] = pd.to_numeric(df[m], errors='coerce')
        ana = df[[sample_col, significance_col] + stratify_cols].copy()
    else:
        raise ValueError("unit must be 'per_sample' or 'per_row'")

    # Overall summary at chosen unit
    ana_non_na_sig = ana.dropna(subset=[significance_col])
    n_all = int(len(ana_non_na_sig))
    n_sig_all = int(ana_non_na_sig[significance_col].sum())
    pct_all = float(100.0 * n_sig_all / n_all) if n_all else float('nan')
    overall = {'n_total': n_all, 'n_significant': n_sig_all, 'pct_significant': pct_all}

    # Quantile cutpoints and per-bin summaries for each metric
    qs = np.linspace(0.0, 1.0, quantiles + 1)
    cutpoints = {}
    rows = []

    for metric in stratify_cols:
        s = pd.to_numeric(ana[metric], errors='coerce')
        valid = ana.loc[s.notna(), [metric, significance_col]].copy()
        if valid.empty:
            logger.warning("No valid data for metric %r. Skipping.", metric)
            continue

        x = valid[metric].astype(float).values

        # Exact percentile cutpoints 0..100
        qvals = np.quantile(x, qs)
        cut_df = pd.DataFrame({'percentile': (qs * 100.0).round(1), metric: qvals})
        cutpoints[metric] = cut_df

        # Bin by empirical quantiles. duplicates='drop' ensures fewer bins in flat regions
        bins = pd.qcut(x, q=qs, duplicates='drop')
        valid = valid.assign(_bin=bins)

        # Keep ordered categories
        cats = list(valid['_bin'].cat.categories)

        # Helper for percentile rank of thresholds
        def _pct_leq(t):
            return float(100.0 * np.mean(x <= t))

        for i, cat in enumerate(cats, start=1):
            left = float(cat.left)
            right = float(cat.right)
            p_lo = _pct_leq(left)
            p_hi = _pct_leq(right)

            sub = valid[valid['_bin'] == cat]
            n = int(len(sub))
            n_sig = int(sub[significance_col].sum())
            pct_sig = (100.0 * n_sig / n) if n else np.nan

            rows.append({
                'metric': metric,
                'bin_idx': i,
                'percentile_range': f"{p_lo:.1f}% - {p_hi:.1f}%",
                'value_range': f"({left:.6g}, {right:.6g}]",
                'n': n,
                'n_sig': n_sig,
                'pct_sig': pct_sig
            })

        # Log cutpoints for this metric
        logger.info("Percentile cutpoints for %r (percentile -> value):\n%s",
                    metric, cut_df.to_string(index=False))

    summary = pd.DataFrame(rows)
    if summary.empty:
        logger.warning("No summary rows produced. Check inputs.")
        return {'summary': summary, 'cutpoints': cutpoints, 'overall': overall}

    # Log the table per metric
    for m in summary['metric'].unique():
        tab = summary[summary['metric'] == m].copy()
        _fmt = {'pct_sig': lambda v: f"{v:.2f}%"} if 'pct_sig' in tab.columns else None
        logger.info("Percent significant by quantile for %r (%s, %s):\n%s",
                    m, unit, sig_rule, tab.to_string(index=False))

    logger.info("Overall significant at unit=%s, sig_rule=%s: %d / %d = %.2f%%",
                unit, sig_rule, overall['n_significant'], overall['n_total'], overall['pct_significant'])

    # Plot
    n_metrics = len(summary['metric'].unique())
    fig_w = max(8.0, 1.0 * quantiles + 5.0)
    fig_h = 4.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # x positions 1..B
    # Build a consistent bin index sequence per metric
    bin_order = sorted(summary['bin_idx'].unique())

    if plot_type == "line":
        for m in summary['metric'].unique():
            y = []
            for b in bin_order:
                row = summary[(summary['metric'] == m) & (summary['bin_idx'] == b)]
                y.append(float(row['pct_sig'].iloc[0]) if not row.empty else np.nan)
            ax.plot(bin_order, y, marker='o', linewidth=1.8, label=m)
    elif plot_type == "bar":
        # grouped bars with small offsets
        width = 0.8 / max(1, n_metrics)
        offsets = np.linspace(-0.4 + width / 2.0, 0.4 - width / 2.0, n_metrics)
        for j, m in enumerate(summary['metric'].unique()):
            y = []
            for b in bin_order:
                row = summary[(summary['metric'] == m) & (summary['bin_idx'] == b)]
                y.append(float(row['pct_sig'].iloc[0]) if not row.empty else np.nan)
            x = [b + offsets[j] for b in bin_order]
            ax.bar(x, y, width=width, align='center', label=m)
    else:
        raise ValueError("plot_type must be 'line' or 'bar'")

    # Axes and cosmetics
    ax.set_xlim(0.5, max(bin_order) + 0.5)
    ax.set_xticks(bin_order)
    ax.set_xticklabels([f"{int((b - 1) * (100 / quantiles))}-{int(b * (100 / quantiles))}%" for b in bin_order])
    ax.set_xlabel(f"Percentile bins ({quantiles} bins)")
    ax.set_ylabel("% significant")
    ax.set_ylim(0.0, min(100.0,
                         np.nanmax(summary['pct_sig']) * 1.15 if np.isfinite(np.nanmax(summary['pct_sig'])) else 50.0))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.20, color='0.5')

    if show_overall and np.isfinite(pct_all):
        ax.axhline(pct_all, linestyle='--', linewidth=1.2)
        ax.text(0.5, pct_all, f" overall = {pct_all:.2f}%", va='bottom', ha='left', fontsize=8)

    title_core = f"% significant vs quantile bins [{unit}, sig_rule={sig_rule}, agg={metric_agg if unit == 'per_sample' else 'n/a'}]"
    ax.set_title(title_core)
    ax.legend(title="metric", frameon=False, ncols=min(len(stratify_cols), 3))

    fig.tight_layout()
    fig_path = None
    csv_path = None
    if plots_dir:
        slug = _safe_slug(out_prefix or f"quantile_sig__{unit}__{sig_rule}__{metric_agg}")
        fig_path = os.path.join(plots_dir, f"{slug}.pdf")
        csv_path = os.path.join(plots_dir, f"{slug}__summary.csv")
        summary.to_csv(csv_path, index=False)
        plt.savefig(fig_path)
        logger.info("Saved summary CSV to %s", csv_path)
        logger.info("Saved figure to %s", fig_path)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return {
        'summary': summary,
        'cutpoints': cutpoints,
        'overall': overall,
        'figure_path': fig_path,
        'csv_summary_path': csv_path,
    }


# ---------- end helper ----------


def compare_prioritizer_rank_performance(results_df):
    EPS = 1e-12  # tolerance for equality
    _cols = ['prioritization_algorithm', 'p_value_DCG', 'p_value_without_ranks']
    _df = results_df[_cols].dropna().copy()
    _df['A'] = _df['p_value_without_ranks'].astype(float)
    _df['B'] = _df['p_value_DCG'].astype(float)
    diff = _df['A'] - _df['B']
    _df['relation'] = np.where(diff > EPS, 'A>B', np.where(diff < -EPS, 'A<B', 'A=B'))
    # Per-prioritizer counts and percents
    _counts = (
        _df.groupby(['prioritization_algorithm', 'relation'])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all three relation columns exist
    for rel in ['A>B', 'A=B', 'A<B']:
        if rel not in _counts.columns:
            _counts[rel] = 0
    _counts = _counts[['A>B', 'A=B', 'A<B']]
    _counts['total'] = _counts.sum(axis=1)
    _pcts = _counts[['A>B', 'A=B', 'A<B']].div(_counts['total'].replace(0, np.nan), axis=0) * 100.0
    _out = _counts.join(_pcts.add_suffix('_pct')).reset_index()
    # Overall counts and percents
    _overall_counts = _df['relation'].value_counts().reindex(['A>B', 'A=B', 'A<B'], fill_value=0)
    _overall_total = int(_overall_counts.sum())
    _overall = pd.DataFrame({
        'relation': ['A>B', 'A=B', 'A<B'],
        'n': [_overall_counts['A>B'], _overall_counts['A=B'], _overall_counts['A<B']],
        'pct': [(100.0 * _overall_counts['A>B'] / _overall_total) if _overall_total else np.nan,
                (100.0 * _overall_counts['A=B'] / _overall_total) if _overall_total else np.nan,
                (100.0 * _overall_counts['A<B'] / _overall_total) if _overall_total else np.nan],
    })
    # Log tables
    _fmt = {c: (lambda v: f"{v:.2f}%") for c in ['A>B_pct', 'A=B_pct', 'A<B_pct']}
    logger.info("A vs B by prioritizer:\n%s", _out.to_string(index=False, formatters=_fmt))
    logger.info("Overall A vs B:\n%s", _overall.to_string(index=False, formatters={'pct': lambda v: f"{v:.2f}%"}))


def inspect_infectious_diseases(results_df):
    # Infectious disease subset analysis:

    # Make one subset with only infectious diseases and one without,  where infectious_disease_parent is in parents (list)
    infectious_disease_subset = results_df[results_df["parents"].apply(lambda p: INFECTIOUS_DISEASE_PARENT in p)]

    # Subset where mondo.0005550 is NOT in parents
    non_infectious_disease_subset = results_df[
        results_df["parents"].apply(lambda p: INFECTIOUS_DISEASE_PARENT not in p)]
    # Proportion of significant for infectious diseases, and non-infectious diseases, log and results and also with result size
    inf_sig_pct = (infectious_disease_subset[
                       'significant'].mean() * 100.0) if not infectious_disease_subset.empty else float('nan')
    non_inf_sig_pct = (non_infectious_disease_subset[
                           'significant'].mean() * 100.0) if not non_infectious_disease_subset.empty else float('nan')
    logger.info("Infectious disease subset: %d rows, %d unique diseases, %0.2f%% significant",
                len(infectious_disease_subset), infectious_disease_subset['sample'].nunique(), inf_sig_pct)
    logger.info("Non-infectious disease subset: %d rows, %d unique diseases, %0.2f%% significant",
                len(non_infectious_disease_subset), non_infectious_disease_subset['sample'].nunique(), non_inf_sig_pct)
    # Proportion below p=1  for both
    inf_p1_pct = (100.0 * (infectious_disease_subset[
                               'p_value_DCG'] < 1.0).mean()) if not infectious_disease_subset.empty else float('nan')
    non_inf_p1_pct = (100.0 * (non_infectious_disease_subset[
                                   'p_value_DCG'] < 1.0).mean()) if not non_infectious_disease_subset.empty else float(
        'nan')
    logger.info("Infectious disease subset: %0.2f%% with p_value_DCG < 1.0", inf_p1_pct)
    logger.info("Non-infectious disease subset: %0.2f%% with p_value_DCG < 1.0", non_inf_p1_pct)


def annotate_heatmap(ax, im, values, fmt="{:.2f}", special_labels=None, fontsize=10, textcolors=("black", "white"),
                     min_contrast=None, use_outline=False, outline_width=1.2, bbox=False, bbox_kwargs=None, ):
    """
    Annotate a heatmap with per-cell labels using contrast-aware text colors.

    ax, im, values: as before (values must match the imshow array order)
    fmt: numeric format when special_labels not provided
    special_labels: optional dict {(row, col): "text"}
    fontsize: label font size
    textcolors: (dark, light) colors to choose from (defaults to black/white)
    min_contrast: optional float (e.g., 3.0 or 4.5). If provided, skip labeling
                  cells where *neither* black nor white achieves this WCAG
                  contrast ratio vs. the background.
    use_outline: if True, draw a subtle outline around text (defaults False)
    outline_width: width for the optional outline
    bbox: if True, draw a semi-transparent box behind text for extra legibility
    bbox_kwargs: dict to customize the bbox (facecolor, alpha, boxstyle, etc.)
    """
    import math

    def _srgb_to_linear(u):
        # WCAG piecewise definition (sRGB 0..1). 0.04045 is the corrected threshold per IEC.
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

    def _relative_luminance(rgb):
        r, g, b = rgb[:3]
        rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
        return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl

    def _contrast_ratio(L1, L2):
        lo, hi = (L1, L2) if L1 <= L2 else (L2, L1)
        return (hi + 0.05) / (lo + 0.05)

    if bbox and bbox_kwargs is None:
        bbox_kwargs = dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.6, edgecolor="none")

    nrows, ncols = values.shape
    for i in range(nrows):
        for j in range(ncols):
            # Pick text
            if special_labels and (i, j) in special_labels:
                text = special_labels[(i, j)]
                v_for_color = np.nan if math.isnan(values[i, j]) else values[i, j]
            else:
                v = values[i, j]
                if np.isnan(v):
                    continue
                text = fmt.format(v)
                v_for_color = v

            # Background color from heatmap
            if np.isfinite(v_for_color):
                rgba = im.cmap(im.norm(v_for_color))
            else:
                rgba = (1, 1, 1, 1)

            L = _relative_luminance(rgba)
            contrast_dark = _contrast_ratio(L, 0.0)  # vs. black
            contrast_light = _contrast_ratio(L, 1.0)  # vs. white

            # Choose foreground that yields higher contrast
            if contrast_dark >= contrast_light:
                fg = textcolors[0]
                outline_color = textcolors[1]
                chosen_contrast = contrast_dark
            else:
                fg = textcolors[1]
                outline_color = textcolors[0]
                chosen_contrast = contrast_light

            # Optionally skip if contrast too low for both
            if min_contrast is not None and chosen_contrast < float(min_contrast):
                continue

            pe = None
            if use_outline:
                pe = [patheffects.withStroke(linewidth=outline_width, foreground=outline_color), patheffects.Normal(), ]

            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=fg, path_effects=pe,
                    bbox=(bbox_kwargs if bbox else None), )


# --- New function: plot_significant_given_returned_percentage_heatmap ---
def plot_significant_given_returned_percentage_heatmap(results_df, plots_dir, network_col='ppi'):
    """
    For each method combo × network, compute the percent of diseases where that combo achieved
    significance among the diseases for which that combo returned any result on that network.
    Denominator: # unique diseases per (combo, network) where the combo appears at least once.
    Numerator  : among those, # diseases where the combo is significant at least once.
    """
    base_cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'significant']
    df0 = results_df[base_cols].dropna().copy()
    df0['_combo'] = df0['algorithm'].astype(str) + ' | ' + df0['prioritization_algorithm'].astype(str)

    # Collapse to disease-level within (combo, network): did this combo ever achieve significance?
    per = (df0.groupby(['_combo', network_col, 'sample'])['significant'].any().reset_index(name='sig_for_combo'))

    # Denominator: number of diseases that this combo returned on this network
    den = (per.groupby(['_combo', network_col])['sample'].nunique().reset_index(name='n_returned'))

    # Numerator: number of those diseases where combo was significant at least once
    num = (per.groupby(['_combo', network_col])['sig_for_combo'].sum().reset_index(name='n_sig'))

    cov = den.merge(num, on=['_combo', network_col], how='left')
    cov['n_sig'] = cov['n_sig'].fillna(0.0)
    cov['pct'] = np.where(cov['n_returned'] > 0, 100.0 * cov['n_sig'] / cov['n_returned'], np.nan)

    # Pivot to heatmap form
    heat = (cov.pivot(index='_combo', columns=network_col, values='pct').sort_index())

    # Ensure consistent combo ordering (all combos present in data)
    all_combos = sorted(df0['_combo'].unique())
    heat = heat.reindex(index=all_combos)

    # Plot heatmap
    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)

    # Marginal means bars
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))

    # Cell labels
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')

    fig.suptitle('Percent significant among diseases where the combo returned results')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% significant (given returned)')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_path = os.path.join(plots_dir, 'significant_given_returned_pct_heatmap.pdf')
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved significant-given-returned heatmap to %s", out_path)


# at top of file:
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_marginal_means_bars(ax, data_2d, *, top_size="12%", right_size="12%", pad=0.15, fmt="{:.2f}", force_xlim=None,
                            force_ylim=None):
    """
    Append tiny bar plots for column means (top) and row means (right) to a heatmap axis.

    ax        : the Axes with your imshow heatmap
    data_2d   : 2D numpy array (same order as shown by imshow)
    top_size  : height of the top bars (e.g., '12%' or in inches like 0.5)
    right_size: width of the right bars
    pad       : gap between heatmap and bars (in inches if float; % if str)
    fmt       : label format for bar-end annotations
    force_xlim/ylim: set to (0,100) for percent heatmaps, otherwise None for auto
    """
    # Compute means ignoring NaNs (matches how you handle NaNs in the heatmap)
    col_means = np.nanmean(data_2d, axis=0)
    row_means = np.nanmean(data_2d, axis=1)

    # Create appended axes that share the heatmap’s x / y to align perfectly
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=top_size, pad=pad, sharex=ax)
    ax_right = divider.append_axes("right", size=right_size, pad=pad, sharey=ax)

    # --- Top bars: column means across x ---
    idx_x = np.arange(data_2d.shape[1])
    ax_top.bar(idx_x, col_means, align="center")
    if force_ylim is not None:
        ax_top.set_ylim(force_ylim)
    ax_top.xaxis.set_visible(False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.set_ylabel("col mean", fontsize=8)
    # Optional tiny number at each bar top
    for j, v in enumerate(col_means):
        ax_top.text(j, v, fmt.format(v), ha="center", va="bottom", fontsize=7)

    # --- Right bars: row means along y ---
    idx_y = np.arange(data_2d.shape[0])
    ax_right.barh(idx_y, row_means, align="center")
    if force_xlim is not None:
        ax_right.set_xlim(force_xlim)
    ax_right.yaxis.set_visible(False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.set_xlabel("row mean", fontsize=8)
    for i, v in enumerate(row_means):
        ax_right.text(v, i, fmt.format(v), ha="left", va="center", fontsize=7)

    # Keep the marginal axes from drawing spines on the shared edges
    for spine in ["bottom", "left"]:
        ax_top.spines[spine].set_visible(False)
    for spine in ["top", "right"]:
        ax_right.spines[spine].set_visible(False)


def plot_minimum_p_value_distribution(results_df, plots_dir):
    # KDE of minimum p_value_DCG per sample
    mins = (results_df[['sample', 'p_value_DCG']].dropna().groupby('sample')['p_value_DCG'].min().clip(lower=0.0,
                                                                                                       upper=1.0))
    x_grid = np.linspace(0.0, 1.0, 512)
    kde = gaussian_kde(mins.values)  # default bandwidth
    y = kde(x_grid)
    fig_kde, ax_kde = plt.subplots(figsize=(8, 4))
    ax_kde.plot(x_grid, y)
    ax_kde.set_xlim(0.0, 1.0)
    ax_kde.set_xlabel('Minimum p_value_DCG per sample')
    ax_kde.set_ylabel('Density')
    ax_kde.set_title('KDE of min p_value_DCG per sample')
    plt.tight_layout()
    kde_path_pdf = os.path.join(plots_dir, 'kde_min_p_value_per_sample.pdf')
    plt.savefig(kde_path_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_kde)


# ---- Distribution grid plots: KDE and ECDF ----
import math


def _auto_grid(n):
    """Return (nrows, ncols) for a near-square grid. Examples: 16->(4,4), 4->(2,2)."""
    if n <= 0:
        return (1, 1)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / float(ncols)))
    return nrows, ncols


def _prepare_unique(df, columns, unique_on='sample', sample_col='sample'):
    """
    Drop duplicates so each unique combination of unique_on columns contributes once.
    Return (df_unique, valid_cols, missing_cols).
    """
    if unique_on is None or (isinstance(unique_on, str) and unique_on.lower() == 'none'):
        df_u = df.copy()
    elif isinstance(unique_on, str):
        if unique_on not in df.columns:
            raise KeyError(f"unique_on column '{unique_on}' not found in results_df")
        df_u = df.drop_duplicates(subset=[unique_on]).copy()
    elif isinstance(unique_on, (list, tuple)):
        for col in unique_on:
            if col not in df.columns:
                raise KeyError(f"unique_on column '{col}' not found in results_df")
        df_u = df.drop_duplicates(subset=list(unique_on)).copy()
    else:
        raise ValueError(f"Invalid unique_on: {unique_on!r}")
    # Keep only requested columns that exist
    valid_cols = [c for c in columns if c in df_u.columns]
    missing = sorted(set(columns) - set(valid_cols))
    if missing:
        logger.warning("Columns missing for distribution plots and will be skipped: %s", missing)
    return df_u, valid_cols, missing


def plot_kde_and_ecdf_grids(
        results_df,
        plots_dir,
        columns=None,
        unique_on='sample',
        sample_col='sample',
        *,
        # per-column log scaling for ECDF only (cosmetic; data untransformed)
        log_ecdf_cols=None,
        log_base=10,
        log_zero_lin_thresh=1.0
):
    """
    Plot two separate grids: KDE and ECDF for each requested numeric column.
    - Uses one row per unique value of `unique_on` (default: sample) to avoid skew.
    - Grid shape is chosen automatically to be near-square.
    - Saves two PDFs in `plots_dir`.
    - Optional: log-scale the ECDF x-axis for selected columns without transforming data.
      Zeros remain visible via a small linear region around 0 (symlog).
    """
    if columns is None or len(columns) == 0:
        columns = ['seed_entropy', 'seed_specificity_adjusted_mean', 'seed_specificity_mean',
                   'jaccard_max', 'jaccard_mean']

    # Normalize log columns set
    if log_ecdf_cols is None:
        log_ecdf_cols = set()
    else:
        log_ecdf_cols = set(log_ecdf_cols)

    # Unique per unique_on and validate columns
    df_u, cols, missing = _prepare_unique(results_df, columns, unique_on=unique_on, sample_col=sample_col)
    if not cols:
        logger.warning('No valid columns found for KDE/ECDF plots. Skipping.')
        return

    # Make a descriptor for suptitles
    if unique_on is None or (isinstance(unique_on, str) and unique_on.lower() == 'none'):
        uniq_desc = 'all rows'
    elif isinstance(unique_on, str):
        uniq_desc = {'sample': 'unique per sample', 'sample_combo': 'unique per sample×combo'}.get(
            unique_on.lower(), f"unique per {unique_on}"
        )
    elif isinstance(unique_on, (list, tuple)):
        uniq_desc = 'unique per ' + ','.join(map(str, unique_on))
    else:
        uniq_desc = 'unique'

    # Figure geometry
    n = len(cols)
    nrows, ncols = _auto_grid(n)
    fig_w = max(8.0, ncols * 3.2)
    fig_h = max(6.0, nrows * 2.6)

    # --- KDE grid ---
    fig_kde, axes_kde = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_kde = axes_kde.ravel()

    # --- ECDF grid ---
    fig_ecdf, axes_ecdf = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_ecdf = axes_ecdf.ravel()

    for i, col in enumerate(cols):
        ax_k = axes_kde[i]
        ax_e = axes_ecdf[i]

        s = pd.to_numeric(df_u[col], errors='coerce').dropna().astype(float)
        if s.empty:
            ax_k.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=9)
            ax_e.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=9)
            continue

        # -----------------------
        # KDE (always linear axis)
        # -----------------------
        x_min = float(np.min(s))
        x_max = float(np.max(s))
        rng = x_max - x_min
        pad = 0.05 * rng if rng > 0 else 1.0

        # Force KDE domain to start at 0
        x_lo = 0.0
        x_hi = x_max + pad
        x_grid = np.linspace(x_lo, x_hi, 512)

        y_kde = None
        try:
            if np.std(s) > 0 and s.nunique() > 1:
                kde = gaussian_kde(s.values)
                y_kde = kde(x_grid)
        except Exception as e:
            logger.warning("KDE failed for %s: %s", col, e)

        if y_kde is not None:
            ax_k.plot(x_grid, y_kde)
            ax_k.set_ylabel('Density')
        else:
            ax_k.axvline(max(x_min, 0.0), linestyle='-', linewidth=1.0)
            ax_k.set_ylabel('Degenerate')

        ax_k.set_xlim(left=0.0)
        ax_k.grid(True, which='both', alpha=0.3)
        ax_k.set_xlabel(col)
        ax_k.grid(alpha=0.2)

        # -----------------------
        # ECDF (optionally symlog axis; data untouched)
        # -----------------------
        v = np.sort(s.values)
        n_obs = v.size
        y = np.arange(1, n_obs + 1, dtype=float) / float(n_obs)
        ax_e.step(v, y, where='post')
        ax_e.set_ylim(0.0, 1.0)
        ax_e.set_ylabel('ECDF')
        ax_e.set_xlabel(col)
        ax_e.set_axisbelow(True)

        # Y ticks (linear) for readability
        ax_e.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_e.yaxis.set_minor_locator(MultipleLocator(0.1))

        if col in log_ecdf_cols:
            # Cosmetic log axis with a linear region near 0 so zero is shown
            ax_e.set_xscale('symlog', base=log_base, linthresh=log_zero_lin_thresh, linscale=1.0)
            ax_e.set_xlim(left=0.0)
            ax_e.margins(x=0.02)

            # --- Compact decade-only ticks + "0" ---
            # Determine positive range for decades
            max_pos = float(np.max(v)) if np.size(v) else 0.0
            if max_pos <= 0:
                decade_ticks = [0.0]
            else:
                e_max = int(np.floor(np.log10(max_pos)))  # highest decade
                e_min = 0  # start from 10^0 since left=0 and we keep a 0 tick
                # Keep labels from overlapping: target <= 8 tick labels beyond 0
                max_labels = 8
                step = max(1, int(np.ceil((e_max - e_min + 1) / max_labels)))
                decade_ticks = [0.0] + [float(10 ** k) for k in range(e_min, e_max + 1, step)]

            # Disable minor ticks completely to avoid clutter like 0.1, 0.2, ...
            ax_e.xaxis.set_minor_locator(NullLocator())

            # Apply our ticks explicitly (decades + 0)
            ax_e.set_xticks(decade_ticks)

            # Format as 10^x for positive decades; keep "0" at zero
            def _pow10_fmt(x, pos):
                if x == 0:
                    return "0"
                # Expect exact decades only; be robust to roundoff
                exp = int(round(np.log10(x))) if x > 0 else 0
                if np.isclose(x, 10 ** exp, rtol=1e-6, atol=1e-12):
                    return rf"$10^{{{exp}}}$"
                return ""  # no label for non-decades (shouldn't happen with our xticks)

            ax_e.xaxis.set_major_formatter(FuncFormatter(_pow10_fmt))

            # Gridlines: majors only on x (decades)
            ax_e.grid(which='major', axis='x', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
            # Keep y-grid as before
            ax_e.grid(which='major', axis='y', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
            ax_e.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.20, color='0.5')

        else:
            # Linear x-axis ECDF ticks/grids
            ax_e.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_e.grid(which='major', axis='both', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
            ax_e.grid(which='minor', axis='both', linestyle='--', linewidth=0.5, alpha=0.20, color='0.5')

    # Hide any unused axes
    for j in range(len(cols), len(axes_kde)):
        axes_kde[j].axis('off')
        axes_ecdf[j].axis('off')

    fig_kde.suptitle(f'KDE distributions ({uniq_desc})')
    fig_ecdf.suptitle(f'Empirical CDFs ({uniq_desc})')

    fig_kde.tight_layout(rect=HEATMAP_TIGHT_RECT, w_pad=1.6, h_pad=0.6)
    fig_ecdf.tight_layout(rect=HEATMAP_TIGHT_RECT, w_pad=1.2, h_pad=0.6)

    var_slug = re.sub(r'[^A-Za-z0-9._-]+', '_', "_".join(cols)).strip('_') or "metrics"

    out_kde = os.path.join(plots_dir, f'kde_grid_distributions__{var_slug}.pdf')
    out_ecdf = os.path.join(plots_dir, f'ecdf_grid_distributions__{var_slug}.pdf')

    plt.figure(fig_kde.number)
    plt.savefig(out_kde)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_kde)
    logger.info("Saved KDE grid to %s", out_kde)

    plt.figure(fig_ecdf.number)
    plt.savefig(out_ecdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_ecdf)
    logger.info("Saved ECDF grid to %s", out_ecdf)


def plot_significance_percentage_heatmap(results_df, significance_level, plots_dir, network_col='ppi'):
    """
    Unconditional % significant per (combo × network).
    Numerator  : count diseases where combo is significant at least once on that network.
    Denominator: total unique diseases for that network across all methods.
    Non-returns are treated as non-significant.
    """
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'significant']
    df = results_df[cols].dropna().copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # disease-level any() for significance
    per = (df.groupby(['_combo', network_col, 'sample'])['significant'].any().reset_index(name='sig_for_combo'))
    num = (per.groupby(['_combo', network_col])['sig_for_combo'].sum().reset_index(name='n_sig'))

    # total diseases per network
    den = (df[[network_col, 'sample']].drop_duplicates().groupby(network_col)['sample'].nunique().rename(
        'n_diseases').reset_index())
    den_map = {r[network_col]: float(r['n_diseases']) for _, r in den.iterrows()}

    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(den[network_col].unique())
    heat = pd.DataFrame(0.0, index=all_combos, columns=all_networks)

    for _, r in num.iterrows():
        net = r[network_col];
        combo = r['_combo'];
        d = den_map.get(net, 0.0)
        heat.at[combo, net] = 100.0 * float(r['n_sig']) / d if d > 0 else np.nan

    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")
    ax.set_xticks(range(heat.shape[1]));
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]));
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network');
    ax.set_ylabel('Module | Prioritization')
    fig.suptitle(
        f"Percentage of significant results (p < {significance_level})\nDenominator = total unique diseases per network")
    cbar = fig.colorbar(im, ax=ax);
    cbar.set_label('% significant (unconditional)')
    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_path = os.path.join(plots_dir, 'significant_pct_heatmap.pdf')
    plt.savefig(out_path);
    plt.show() if SHOW_PLOTS else plt.close(fig)
    logger.info("Saved heatmap to %s", out_path)


def plot_conditional_significance_percentage_heatmap(results_df, plots_dir, network_col='ppi'):
    """
    Conditional significance heatmap.
    For each network (column), restrict the denominator to diseases (samples) where at least
    one method combo achieved a significant result on that network. Each cell shows the percent
    of those eligible diseases for which the given method combo achieved significance at least once
    (across seeds/runs) on that network.
    """
    # Keep only the necessary columns and build a readable combo label
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'significant']
    df = results_df[cols].dropna().copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # Eligible diseases per network: at least one method significant on that network
    elig = (df.groupby([network_col, 'sample'])['significant'].any().reset_index(name='any_sig'))
    elig = elig[elig['any_sig']]

    # Filter to eligible (network, sample) pairs
    df_elig = df.merge(elig[[network_col, 'sample']], on=[network_col, 'sample'], how='inner')

    # For each (combo, network, sample), did that combo achieve significance at least once?
    per = (df_elig.groupby(['_combo', network_col, 'sample'])['significant'].any().reset_index(name='sig_for_combo'))

    # Numerator: count eligible diseases where combo was significant
    num = (per.groupby(['_combo', network_col])['sig_for_combo'].sum().reset_index(name='n_sig'))

    # Denominator: number of eligible diseases per network (any method significant)
    den = (elig.groupby(network_col)['sample'].nunique().rename('n_eligible').reset_index())

    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(den[network_col].unique())

    # Initialize matrix of zeros and then convert to column-wise percentages
    heat = pd.DataFrame(0.0, index=all_combos, columns=all_networks)

    # Fill numerators
    for _, r in num.iterrows():
        heat.at[r['_combo'], r[network_col]] = float(r['n_sig'])

    # Convert to percentages using denominators per network
    den_map = {row[network_col]: float(row['n_eligible']) for _, row in den.iterrows()}
    for net in all_networks:
        d = den_map.get(net, 0.0)
        if d > 0.0:
            heat[net] = (heat[net] / d) * 100.0
        else:
            heat[net] = np.nan  # no eligible diseases for this network

    # Plot heatmap (style consistent with other heatmaps)
    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)

    # Marginal mean bars using finite values
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))

    # Cell labels
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')

    fig.suptitle('Conditional % significant given any method significant (per network)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% significant (conditional)')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_path = os.path.join(plots_dir, 'conditional_significant_pct_heatmap.pdf')
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved conditional significance heatmap to %s", out_path)


def plot_returned_results_percentage_heatmap(results_df, plots_dir, network_col='ppi'):
    """Coverage heatmap: percent of diseases with a returned result by (method combo × network).
    Numerator: # unique diseases (samples) where this method combo appears at least once on the network.
    Denominator: # unique diseases for that network overall (in the data).
    """
    # Deduplicate to disease granularity (ignore seeds/runs)
    base_cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample']
    df0 = results_df[base_cols].dropna().drop_duplicates()
    df0['_combo'] = df0['algorithm'].astype(str) + ' | ' + df0['prioritization_algorithm'].astype(str)

    # Numerator: unique diseases per (combo, network)
    num = (df0.groupby(['_combo', network_col])['sample'].nunique().reset_index(name='n_returned'))

    # Denominator: unique diseases per network
    den = (df0.groupby([network_col])['sample'].nunique().rename('n_diseases').reset_index())

    # Join and compute percentage
    cov = num.merge(den, on=network_col, how='left')
    cov['pct'] = np.where(cov['n_diseases'] > 0, 100.0 * cov['n_returned'] / cov['n_diseases'], np.nan)

    # Pivot; fill missing with 0
    heat = (cov.pivot(index='_combo', columns=network_col, values='pct').sort_index().fillna(0.0))

    # Plot (same style as your other heatmaps)
    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)

    # Marginal mean bars
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))

    # Labels
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')

    fig.suptitle('Coverage: percent of diseases with a returned result by method × network\n'
                 'Denominator = unique diseases per network; Numerator = diseases where the method returned any result')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% returned results')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_path = os.path.join(plots_dir, 'returned_results_pct_heatmap.pdf')
    plt.savefig(out_path)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved returned-results coverage heatmap to %s", out_path)


def plot_winrate_min_pvalue_heatmap(results_df, plots_dir, network_col='ppi'):
    """Heatmap of win rate: for each network column, percentage of samples where a method combo
    achieved the strictly lowest p_value_DCG. Samples with p_value_DCG == 1.0 are excluded,
    and ties for the minimum are excluded. Each column sums to 100%."""

    # 1) Build a readable method-combo label and keep only needed columns
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # 2) Exclude ineligible rows (p == 1)
    df = df[df['p_value_DCG'] < 1.0]

    # 3) Determine a UNIQUE winner per (network, sample) explicitly and readably
    #    For each network and each sample, sort by p_value_DCG; if the minimum is unique,
    #    record that combo as the winner for that network+sample.
    winner_rows = []
    for net, df_net in df.groupby(network_col):
        for sample, df_s in df_net.groupby('sample'):
            df_s_sorted = df_s.sort_values('p_value_DCG', kind='mergesort')
            min_val = df_s_sorted['p_value_DCG'].iloc[0]
            # Skip if there is a tie for the minimum
            if (df_s_sorted['p_value_DCG'] == min_val).sum() != 1:
                continue
            winner_rows.append({'_combo': df_s_sorted['_combo'].iloc[0], network_col: net, 'sample': sample})

    winners = pd.DataFrame(winner_rows)

    # 4) Count wins per (combo, network)
    win_counts = (winners.groupby(['_combo', network_col]).size().reset_index(name='wins'))

    # 5) Denominator per network: how many samples produced a unique winner in that network
    eligible_samples = winners.groupby(network_col)['sample'].nunique()

    # 6) Build a complete matrix of combos x networks (fill missing with 0 wins)
    all_combos = sorted(df['_combo'].unique())
    all_networks = list(eligible_samples.index)
    heat = pd.DataFrame(0.0, index=all_combos, columns=all_networks)
    for _, row in win_counts.iterrows():
        heat.at[row['_combo'], row[network_col]] = float(row['wins'])

    # 7) Convert raw win counts to column-wise percentages so each column sums to 100
    for net in all_networks:
        denom = float(eligible_samples.loc[net])
        if denom > 0:
            heat[net] = (heat[net] / denom) * 100.0

    # 8) Plot heatmap
    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))

    # 9) Numeric labels (2 decimals)
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")
    # 10) Axes and title
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')
    fig.suptitle('Win rate: % lowest p_value_DCG (excluding p=1 and ties)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% wins')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'winrate_min_pvalue_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved win-rate heatmap to %s", out_pdf)


def plot_global_winrate_heatmap(results_df, plots_dir, network_col='ppi'):
    """
    Global win rate heatmap.

    A "win" is the strictly lowest p_value_DCG for a given (network, sample) among all method combos,
    with p_value_DCG < 1.0 required and ties excluded. Each cell shows the percentage of ALL such wins
    captured by (method combo × network). The sum of all cells in the matrix is 100%.

    Denominator: total number of unique (network, sample) pairs that yield a unique winner across the dataset.
    Numerator  : number of those wins attributed to the given (combo, network).
    """
    # 1) Keep only the needed columns and build a readable combo label
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # 2) Exclude ineligible rows (p == 1)
    df = df[df['p_value_DCG'] < 1.0]

    # 3) Determine unique winners per (network, sample)
    winner_rows = []
    for net, df_net in df.groupby(network_col):
        for sample, df_s in df_net.groupby('sample'):
            df_s_sorted = df_s.sort_values('p_value_DCG', kind='mergesort')
            min_val = df_s_sorted['p_value_DCG'].iloc[0]
            if (df_s_sorted['p_value_DCG'] == min_val).sum() != 1:
                continue  # drop ties
            winner_rows.append({'_combo': df_s_sorted['_combo'].iloc[0], network_col: net, 'sample': sample})

    winners = pd.DataFrame(winner_rows)

    # If no unique winners, exit gracefully
    if winners.empty:
        logger.warning("No unique winners found for global win rate heatmap (after excluding p=1 and ties).")
        return

    # 4) Counts per (combo, network) and the global denominator
    win_counts = winners.groupby(['_combo', network_col]).size().reset_index(name='wins')
    global_total = float(len(winners))  # total number of unique winners across all networks and samples

    # 5) Build complete matrix of combos × networks and convert to percentages of the GLOBAL total
    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(results_df[network_col].dropna().astype(str).unique())
    heat = pd.DataFrame(0.0, index=all_combos, columns=all_networks)

    for _, row in win_counts.iterrows():
        combo = row['_combo']
        net = str(row[network_col])
        heat.at[combo, net] = (100.0 * float(row['wins']) / global_total) if global_total > 0 else np.nan

    # 6) Plot heatmap
    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = float(np.nanmax(heat.values)) if np.isfinite(np.nanmax(heat.values)) else 100.0
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=vmax)

    # Marginal bars (means of cell shares; matrix itself sums to 100% globally)
    add_marginal_means_bars(ax, heat.values, fmt="{:.2f}", force_xlim=(0, vmax), force_ylim=(0, vmax))

    # Cell labels
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')

    fig.suptitle('Global win rate: % of all unique wins (p<1, ties excluded)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% of all unique wins')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'winrate_global_min_pvalue_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved global win-rate heatmap to %s", out_pdf)


def compute_best_ratio_distribution(results_df, network_col='ppi'):
    """Compute ratios per (combo, network, sample): best_score / combo_score,
    where score = -log10(p_value_DCG). Skip (network, sample) where all p==1.
    Keep p==1 cases for individual combos (ratio becomes 0 when their score==0)."""
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    rows = []
    for net, df_net in df.groupby(network_col):
        for sample, df_s in df_net.groupby('sample'):
            pvals = df_s['p_value_DCG'].astype(float).values
            # Skip uninformative cases where all methods have p==1 (scores==0)
            if np.all(pvals == 1.0):
                continue
            scores = -np.log10(pvals)
            best_score = np.max(scores)
            # Build rows explicitly for readability
            for rec, score in zip(df_s.itertuples(index=False), scores):
                ratio = score / best_score
                rows.append({'_combo': getattr(rec, 'algorithm') + ' | ' + getattr(rec, 'prioritization_algorithm'),
                             network_col: getattr(rec, network_col), 'sample': getattr(rec, 'sample'),
                             'ratio': float(ratio)})
    return pd.DataFrame(rows)


def plot_median_best_ratio_heatmap(results_df, plots_dir, network_col='ppi'):
    """Heatmap of the mean(best_score/score) per (combo x network).
    best_score is the maximum -log10(p) within each (network, sample).
    Ignores (network, sample) where all p==1; includes p==1 for specific combos (ratio=0)."""
    ratios = compute_best_ratio_distribution(results_df, network_col=network_col)

    mean_df = (ratios.groupby(['_combo', network_col])['ratio'].mean().reset_index(name='mean_ratio'))

    heat = (mean_df.pivot(index='_combo', columns=network_col, values='mean_ratio').sort_index())

    # Use finite values for color mapping
    heat_num = heat.replace(np.inf, np.nan)

    fig_width = max(8, heat_num.shape[1] * 1.2)
    fig_height = max(8, heat_num.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat_num.values, aspect='auto')
    add_marginal_means_bars(ax, heat_num.values, fmt="{:.2f}", force_xlim=(0, 1), force_ylim=(0, 1))

    # Numeric labels (2 decimals)
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')
    fig.suptitle('Mean score/best score ratio (−log10 p)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean ratio (finite only)')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'mean_best_ratio_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved mean best-ratio heatmap to %s", out_pdf)


def compute_rowwise_r_distribution(results_df, network_col='ppi'):
    """For each method combo and disease, compare across networks and compute r = score/best_score
    where score = -log10(p_value_DCG). Drop combo-diseases where all networks have p==1 (no signal).
    Keep r=0 cases when a network has p==1 but at least one network succeeds."""
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    rows = []
    for combo, df_c in df.groupby('_combo'):
        for sample, df_cs in df_c.groupby('sample'):
            pvals = df_cs['p_value_DCG'].astype(float).values
            # Skip if all networks have p==1 for this combo and disease
            if np.all(pvals == 1.0):
                continue
            scores = -np.log10(pvals)
            best_score = float(np.max(scores))
            for rec, score in zip(df_cs.itertuples(index=False), scores):
                r = 0.0 if score == 0.0 else float(score / best_score)
                rows.append(
                    {'_combo': combo, network_col: getattr(rec, network_col), 'sample': getattr(rec, 'sample'), 'r': r})
    return pd.DataFrame(rows)


def compute_global_best_ratio_distribution(results_df, network_col='ppi'):
    """Compute r per (combo, network, sample): r = score / best_score_global,
    where score = -log10(p_value_DCG) and best_score_global is the maximum score
    across ALL methods and networks for that sample (disease).
    Drop diseases where best_score_global == 0 (all p==1 across everything).
    Keep r=0 for combos with p==1 when others have signal.
    """
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # Per-row score and per-disease global best score
    df['score'] = -np.log10(df['p_value_DCG'].astype(float).clip(lower=0.0, upper=1.0))
    best_global = df.groupby('sample')['score'].max().rename('best_score_global').reset_index()
    df = df.merge(best_global, on='sample', how='left')

    # Drop diseases with no signal anywhere
    df = df[df['best_score_global'] > 0.0]

    # Ratio in [0,1]; p=1 rows get 0 when others succeed
    df['ratio'] = (df['score'] / df['best_score_global']).fillna(0.0).clip(0.0, 1.0)
    return df[['_combo', network_col, 'sample', 'ratio']]


def plot_median_global_best_ratio_heatmap(results_df, plots_dir, network_col='ppi'):
    """Heatmap of mean r with r = score / best_score_global.
    best_score_global is computed across ALL methods and networks per disease.
    Drops only diseases where all p==1. Cells in [0,1]."""
    ratios = compute_global_best_ratio_distribution(results_df, network_col=network_col)

    mean_df = (ratios.groupby(['_combo', network_col])['ratio'].mean().reset_index(name='mean_r'))

    heat = (mean_df.pivot(index='_combo', columns=network_col, values='mean_r').sort_index())

    heat_num = heat.copy()  # all values in [0,1]

    fig_width = max(8, heat_num.shape[1] * 1.2)
    fig_height = max(8, heat_num.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat_num.values, aspect='auto', vmin=0.0, vmax=1.0)

    # Marginal means bars (same style you’re using elsewhere)
    add_marginal_means_bars(ax, heat_num.values, fmt="{:.2f}", force_xlim=(0, 1), force_ylim=(0, 1))

    # Value labels
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Module | Prioritization')

    fig.suptitle('Mean closeness-to-best ratio across ALL methods and networks\n'
                 '(r = score / best score per disease, score = −log10 p)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean r')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'mean_global_best_ratio_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved global-best mean ratio heatmap to %s", out_pdf)


def plot_rowwise_median_r_heatmap(results_df, plots_dir, network_col='ppi'):
    """Row-wise mean r across networks per method combo, r = score/best_score within combo.
    Higher is closer to the best network for that method. Cells in [0,1]."""
    ratios = compute_rowwise_r_distribution(results_df, network_col=network_col)

    mean_df = (ratios.groupby(['_combo', network_col])['r'].mean().reset_index(name='mean_r'))

    heat = (mean_df.pivot(index='_combo', columns=network_col, values='mean_r').sort_index())

    heat_num = heat.copy()  # all finite in [0,1]

    fig_width = max(8, heat_num.shape[1] * 1.2)
    fig_height = max(8, heat_num.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat_num.values, aspect='auto', vmin=0.0, vmax=1.0)
    add_marginal_means_bars(ax, heat_num.values, fmt="{:.2f}", force_xlim=(0, 1), force_ylim=(0, 1))

    # Labels to 2 decimals
    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")

    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Method combo')
    fig.suptitle('Row-wise mean r across networks per method (r = score/best within method)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean r')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'rowwise_mean_r_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved row-wise mean r heatmap to %s", out_pdf)


def plot_rowwise_network_winrate_heatmap(results_df, plots_dir, network_col='ppi'):
    """For each method combo, across networks, compute the fraction of diseases where a network
    achieves the strictly lowest p_value_DCG (unique minimum) - tie cases dropped, p==1 excluded.
    Rows sum to 100% per method."""
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    df = results_df[cols].copy()
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # Exclude p==1 from candidate winners to avoid random minima
    df = df[df['p_value_DCG'] < 1.0]

    winner_rows = []
    for combo, df_c in df.groupby('_combo'):
        for sample, df_cs in df_c.groupby('sample'):
            df_sorted = df_cs.sort_values('p_value_DCG', kind='mergesort')
            min_val = df_sorted['p_value_DCG'].iloc[0]
            if (df_sorted['p_value_DCG'] == min_val).sum() != 1:
                continue  # drop ties
            winner_rows.append({'_combo': combo, network_col: df_sorted[network_col].iloc[0], 'sample': sample})

    winners = pd.DataFrame(winner_rows)

    win_counts = (winners.groupby(['_combo', network_col]).size().reset_index(name='wins'))

    # Denominator per method combo
    denom = winners.groupby('_combo')['sample'].nunique()

    # Build matrix rows that sum to 100
    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(df[network_col].unique())
    heat = pd.DataFrame(0.0, index=all_combos, columns=all_networks)
    for _, row in win_counts.iterrows():
        heat.at[row['_combo'], row[network_col]] = float(row['wins'])

    for combo in all_combos:
        d = float(denom.get(combo, 0.0))
        if d > 0:
            heat.loc[combo, :] = (heat.loc[combo, :] / d) * 100.0

    fig_width = max(8, heat.shape[1] * 1.2)
    fig_height = max(8, heat.shape[0] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.values, aspect='auto', vmin=0.0, vmax=100.0)
    add_marginal_means_bars(ax, heat.values, fmt="{:.1f}", force_xlim=(0, 100), force_ylim=(0, 100))

    annotate_heatmap(ax, im, heat.values, fmt="{:.2f}")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha='right')
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(list(heat.index))
    ax.set_xlabel('Network')
    ax.set_ylabel('Method combo')
    fig.suptitle('Row-wise network win rate per method (unique min p, p=1 excluded)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('% wins')

    plt.tight_layout(rect=HEATMAP_TIGHT_RECT)
    out_pdf = os.path.join(plots_dir, 'rowwise_network_winrate_heatmap.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved row-wise network win-rate heatmap to %s", out_pdf)


# --- Helper and refactored plotting functions for seed-stratified winrate ---
def _compute_seed_unique_win_stats(results_df, network_col='ppi', seed_col='seeds', prio_filter='trustrank'):
    """Compute unique-win counts per (_combo, network, seed_bin) and denominators per (network, seed_bin).
    A unique win is the strictly lowest p_value_DCG among method combos for a given (network, sample, seed_bin),
    considering only rows with prioritization_algorithm == prio_filter and p_value_DCG < 1.0; ties are dropped.
    Seed binning: 1..25 kept, >25 collapsed to 26. Seed == 0 is ignored.
    Returns: win_counts DataFrame, denom DataFrame, all_combos list, all_networks list, seeds_order list.
    """
    cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    if seed_col not in results_df.columns:
        raise KeyError(f"Seed column '{seed_col}' not found in results_df")
    cols = cols + [seed_col]
    df = results_df[cols].copy()

    # Filter prioritization algorithm to the requested one (TrustRank by default)
    df = df[df['prioritization_algorithm'] == prio_filter]

    # Basic cleaning for seeds
    df = df[df[seed_col].notna()]
    df[seed_col] = df[seed_col].astype(int)
    df = df[df[seed_col] != 0]  # ignore seed == 0

    # Combo label
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # Seed binning
    def _bin_seed(s):
        return s if 1 <= s <= 25 else 26

    df['seed_bin'] = df[seed_col].apply(_bin_seed)

    # Candidates for winners must have p < 1
    cand = df[df['p_value_DCG'] < 1.0]

    # Determine unique winners per (network, seed_bin, sample)
    winner_rows = []
    for (net, sbin, sample), grp in cand.groupby([network_col, 'seed_bin', 'sample']):
        grp_sorted = grp.sort_values('p_value_DCG', kind='mergesort')
        min_val = grp_sorted['p_value_DCG'].iloc[0]
        if (grp_sorted['p_value_DCG'] == min_val).sum() != 1:
            continue  # drop ties
        winner_rows.append(
            {'_combo': grp_sorted['_combo'].iloc[0], network_col: net, 'seed_bin': int(sbin), 'sample': sample})
    winners = pd.DataFrame(winner_rows)

    # Denominator: # unique-winner diseases per (network, seed_bin)
    denom = winners.groupby([network_col, 'seed_bin'])['sample'].nunique().rename('denom').reset_index()

    # Win counts per (combo, network, seed_bin)
    win_counts = (winners.groupby(['_combo', network_col, 'seed_bin']).size().reset_index(name='wins'))

    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(df[network_col].unique())
    seeds_order = list(range(1, 26)) + [26]

    return win_counts, denom, all_combos, all_networks, seeds_order


def plot_seed_stratified_winrate_by_network(results_df, plots_dir, network_col='ppi', seed_col='seeds',
                                            prio_filter='trustrank'):
    """Line plots of unique-win rate vs seed count, faceted by network (TrustRank-only).
    A "win" is the strictly lowest p_value_DCG among method combos for a given (network, sample, seed_bin),
    considering only p<1 rows; ties are excluded. Per-panel (network) denominators are diseases with a unique winner.
    """
    win_counts, denom, all_combos, all_networks, seeds_order = _compute_seed_unique_win_stats(results_df,
                                                                                              network_col=network_col,
                                                                                              seed_col=seed_col,
                                                                                              prio_filter=prio_filter)

    n_nets = len(all_networks)
    if n_nets == 0:
        logger.warning('No networks available for seed-stratified win rate plot.')
        return

    # Dynamic grid sizing
    ncols = int(np.ceil(np.sqrt(n_nets)))
    ncols = max(2, min(5, ncols))
    nrows = int(np.ceil(n_nets / ncols))
    fig_w = max(12, ncols * 3.6)
    fig_h = max(6, nrows * 3.0)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    denom_map = {(r[network_col], int(r['seed_bin'])): int(r['denom']) for _, r in denom.iterrows()}

    for ax, net in zip(axes, all_networks):
        wins_net = win_counts[win_counts[network_col] == net]
        wins_map = {(r['_combo'], int(r['seed_bin'])): int(r['wins']) for _, r in wins_net.iterrows()}

        for combo in all_combos:
            y = []
            for s in seeds_order:
                d = denom_map.get((net, s), 0)
                if d > 0:
                    w = wins_map.get((combo, s), 0)
                    y.append(100.0 * w / d)
                else:
                    y.append(np.nan)
            ax.plot(seeds_order, y, marker='o', linewidth=1.0, markersize=3, label=combo)

        ax.set_title(f"Network: {net}")
        ax.set_ylim(0.0, 100.0)

    # Hide unused axes
    for k in range(n_nets, len(axes)):
        axes[k].axis('off')

    # X ticks and labels
    for ax in axes[:n_nets]:
        ax.set_xticks([1, 5, 10, 15, 20, 25, 26])
        ax.set_xticklabels(['1', '5', '10', '15', '20', '25', '>25'])

    fig.supxlabel('Seed count (1..25; >25 aggregated)')
    fig.supylabel('Unique-win rate across diseases (%)')
    fig.suptitle(
        'TrustRank-only: stratified unique-win rate vs seed count, faceted by network\nCandidates: p<1; ties excluded; per-seed denominators are # diseases with a unique winner')

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    out_pdf = os.path.join(plots_dir, 'seed_stratified_winrate_by_network.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved seed-stratified win-rate plot to %s", out_pdf)


def plot_seed_stratified_winrate_overall(results_df, plots_dir, network_col='ppi', seed_col='seeds',
                                         prio_filter='trustrank', show_error='range', jitter=True, capsize=2.0):
    """Single figure: seed-stratified unique-win rate pooled across networks (TrustRank-only).
    For each seed_bin, the pooled percentage for a method is sum_wins / sum_denom across networks.

    Additional visualization options:
      - show_error: 'range' | 'sd_net' | 'stderr' | 'none'
          'range'  -> vertical error bars spanning [min, max] of per-network percentages.
          'sd_net' -> symmetric standard deviation across networks of per-network percentages.
          'stderr' -> symmetric 1 s.e. binomial error bars around the pooled percentage.
          'none'   -> no error bars.
      - jitter: If True, apply small horizontal offsets per method to reduce overlap.
      - capsize: Error-bar capsize in points.
    """
    # Reuse helper to compute unique-win counts and denominators per network × seed
    win_counts, denom, all_combos, all_networks, seeds_order = _compute_seed_unique_win_stats(results_df,
                                                                                              network_col=network_col,
                                                                                              seed_col=seed_col,
                                                                                              prio_filter=prio_filter)

    # Aggregate across networks with denominators as weights (pooled percentage per seed_bin)
    wins_total = (win_counts.groupby(['_combo', 'seed_bin'])['wins'].sum().rename('wins').reset_index())
    denom_total = (denom.groupby(['seed_bin'])['denom'].sum().rename('denom').reset_index())
    denom_map = {int(r['seed_bin']): int(r['denom']) for _, r in denom_total.iterrows()}

    # Fast lookup for pooled wins
    wins_map = {(r['_combo'], int(r['seed_bin'])): int(r['wins']) for _, r in wins_total.iterrows()}

    # Precompute per-network percentages if needed for 'range' or 'sd_net'
    minmax_map = {}
    sd_map = {}
    if show_error in ('range', 'sd_net'):
        per_net = win_counts.merge(denom, on=[network_col, 'seed_bin'], how='left')  # has wins, denom
        per_net = per_net[per_net['denom'] > 0]
        per_net['pct'] = 100.0 * per_net['wins'] / per_net['denom']

        if show_error == 'range':
            agg = (per_net.groupby(['_combo', 'seed_bin'])['pct'].agg(['min', 'max']).reset_index())
            for _, row in agg.iterrows():
                key = (row['_combo'], int(row['seed_bin']))
                minmax_map[key] = (float(row['min']), float(row['max']))
        elif show_error == 'sd_net':
            # Use population SD (ddof=0) so a single-network seed gives 0 rather than NaN
            agg = (per_net.groupby(['_combo', 'seed_bin'])['pct'].agg(
                lambda s: float(np.std(np.asarray(list(s), dtype=float), ddof=0))).reset_index(name='sd'))
            for _, row in agg.iterrows():
                key = (row['_combo'], int(row['seed_bin']))
                sd_map[key] = float(row['sd'])

    # Prepare plotting canvas
    fig_w = max(10, 0.45 * len(seeds_order) + 8)
    fig_h = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Jitter offsets per combo (symmetrically spaced around 0)
    if jitter and len(all_combos) > 1:
        offsets = np.linspace(-0.12, 0.12, len(all_combos))
    else:
        offsets = np.zeros(len(all_combos))
    offset_map = {combo: float(off) for combo, off in zip(all_combos, offsets)}

    # Plot each combo with optional error bars
    for combo in all_combos:
        # Pooled percentage per seed
        y = []
        x = []
        for s in seeds_order:
            d = denom_map.get(s, 0)
            if d > 0:
                w = wins_map.get((combo, s), 0)
                y.append(100.0 * w / d)
            else:
                y.append(np.nan)
            x.append(s + offset_map[combo])

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if show_error == 'range':
            # Asymmetric errors from per-network min-max
            yerr_lower = []
            yerr_upper = []
            for s, yi in zip(seeds_order, y):
                if not np.isfinite(yi):
                    yerr_lower.append(np.nan)
                    yerr_upper.append(np.nan)
                    continue
                mm = minmax_map.get((combo, s))
                if mm is None:
                    yerr_lower.append(np.nan)
                    yerr_upper.append(np.nan)
                else:
                    y_min, y_max = mm
                    yerr_lower.append(max(0.0, yi - y_min))
                    yerr_upper.append(max(0.0, y_max - yi))
            yerr = np.vstack([yerr_lower, yerr_upper])
            ax.errorbar(x, y, yerr=yerr, fmt='-o', linewidth=1.2, markersize=4, elinewidth=0.8, capsize=capsize,
                        alpha=0.9, label=combo)
        elif show_error == 'sd_net':
            # Symmetric SD across networks of per-network percentages
            yerr = []
            for s, yi in zip(seeds_order, y):
                if not np.isfinite(yi):
                    yerr.append(np.nan)
                else:
                    yerr.append(sd_map.get((combo, s), np.nan))
            ax.errorbar(x, y, yerr=np.asarray(yerr, dtype=float), fmt='-o', linewidth=1.2, markersize=4, elinewidth=0.8,
                        capsize=capsize, alpha=0.9, label=combo)
        elif show_error == 'stderr':
            # Binomial standard error for pooled percentage: 100 * sqrt(p*(1-p)/N)
            yerr = []
            for s, yi in zip(seeds_order, y):
                d = denom_map.get(s, 0)
                if d > 0 and np.isfinite(yi):
                    p = yi / 100.0
                    yerr.append(100.0 * np.sqrt(p * (1.0 - p) / float(d)))
                else:
                    yerr.append(np.nan)
            ax.errorbar(x, y, yerr=np.asarray(yerr), fmt='-o', linewidth=1.2, markersize=4, elinewidth=0.8,
                        capsize=capsize, alpha=0.9, label=combo)
        else:
            ax.plot(x, y, marker='o', linewidth=1.2, markersize=4, label=combo)

    # Axes styling
    ax.set_xticks([1, 5, 10, 15, 20, 25, 26])
    ax.set_xticklabels(['1', '5', '10', '15', '20', '25', '>25'])
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel('Seed count (1..25; >25 aggregated)')
    ax.set_ylabel('Unique-win rate across diseases (%)')
    title_err = {'range': 'with per-network min-max bars', 'sd_net': 'with across-network SD bars',
                 'stderr': 'with binomial s.e. bars', 'none': 'no error bars'}.get(show_error,
                                                                                   'with per-network min-max bars')
    ax.set_title(f'TrustRank-only: pooled unique-win rate vs seed count across networks\n'
                 f'Candidates: p<1; ties excluded; denominators pooled across networks; {title_err}')

    # Legend outside plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    out_pdf = os.path.join(plots_dir, 'seed_stratified_winrate_overall.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved pooled seed-stratified win-rate plot to %s", out_pdf)


# ---- New: Seed-stratified percent-significant (not winrate) ----

def _compute_seed_sig_stats(results_df, network_col='ppi', seed_col='seeds', prio_filter='trustrank',
                            significance_col='significant', significance_level=DEFAULT_SIGNIFICANCE_LEVEL):
    """Compute counts of significant diseases per (_combo, network, seed_bin) and denominators per
    (network, seed_bin). A disease is counted as significant for a combo if that combo achieved
    significance at least once for that (network, seed_bin, disease). Denominator is the total number
    of diseases that appear for that (network, seed_bin) irrespective of method presence. Seed==0 is
    ignored; seeds >25 are binned to 26. Returns: sig_counts, denom, all_combos, all_networks, seeds_order.
    """
    # Required columns
    base_cols = ['algorithm', 'prioritization_algorithm', network_col, 'sample', 'p_value_DCG']
    if seed_col not in results_df.columns:
        raise KeyError(f"Seed column '{seed_col}' not found in results_df")

    # Build working DataFrame
    keep_cols = base_cols + [seed_col]
    if significance_col in results_df.columns:
        keep_cols.append(significance_col)
    df = results_df[keep_cols].copy()

    # Filter prioritization algorithm
    df = df[df['prioritization_algorithm'] == prio_filter]

    # Clean seeds and binning
    df = df[df[seed_col].notna()]
    df[seed_col] = df[seed_col].astype(int)
    df = df[df[seed_col] != 0]  # ignore seed == 0
    df['seed_bin'] = df[seed_col].apply(lambda s: s if 1 <= s <= 25 else 26)

    # Ensure significance column exists (fallback to p_value_DCG < significance_level)
    if significance_col not in df.columns:
        df[significance_col] = (df['p_value_DCG'].astype(float) < float(significance_level))

    # Combo label
    df['_combo'] = df['algorithm'].astype(str) + ' | ' + df['prioritization_algorithm'].astype(str)

    # For each (combo, network, seed_bin, sample): did the combo hit significance at least once?
    per = (df.groupby(['_combo', network_col, 'seed_bin', 'sample'])[significance_col].any().reset_index(
        name='sig_for_combo'))

    # Numerator: # significant diseases per (combo, network, seed_bin)
    sig_counts = (per.groupby(['_combo', network_col, 'seed_bin'])['sig_for_combo'].sum().reset_index(name='n_sig'))

    # Denominator: total diseases per (network, seed_bin) (method-agnostic)
    denom = (df.groupby([network_col, 'seed_bin'])['sample'].nunique().rename('denom').reset_index())

    all_combos = sorted(df['_combo'].unique())
    all_networks = sorted(df[network_col].unique())
    seeds_order = list(range(1, 26)) + [26]

    return sig_counts, denom, all_combos, all_networks, seeds_order


def plot_seed_stratified_percent_significant_overall(results_df, plots_dir, *, network_col='ppi', seed_col='seeds',
                                                     prio_filter='trustrank', significance_col='significant',
                                                     significance_level=DEFAULT_SIGNIFICANCE_LEVEL,
                                                     show_error='sd_net',  # 'sd_net' | 'stderr' | 'none'
                                                     jitter=True, capsize=2.0, ):
    """Variant of seed-stratified plot that shows *percent significant* (not win rate).

    For each seed_bin and network, compute for each method combo the percentage of diseases that are
    significant for that combo (significant==True at least once within that seed bin). The denominator
    per (network, seed_bin) is the *total number of diseases* that appear at that seed bin (method-agnostic).

    The figure aggregates across networks by plotting the *mean* of per-network percentages for each seed_bin
    with optional error bars showing across-network *standard deviation* ('sd_net') or pooled *binomial s.e.*
    ('stderr'). Seed==0 ignored; seeds >25 aggregated to 26.
    """
    # Compute per-network counts and denominators
    sig_counts, denom, all_combos, all_networks, seeds_order = _compute_seed_sig_stats(results_df,
                                                                                       network_col=network_col,
                                                                                       seed_col=seed_col,
                                                                                       prio_filter=prio_filter,
                                                                                       significance_col=significance_col,
                                                                                       significance_level=significance_level, )

    # Merge to form per-network percentages
    per_net = sig_counts.merge(denom, on=[network_col, 'seed_bin'], how='right')
    # Fill missing combos implicitly with 0 n_sig by expanding over combos
    # Build a complete grid of (_combo, network, seed_bin) present in denom
    grid = []
    for net, sbin in per_net[[network_col, 'seed_bin']].drop_duplicates().itertuples(index=False):
        for combo in all_combos:
            grid.append({'_combo': combo, network_col: net, 'seed_bin': int(sbin)})
    grid_df = pd.DataFrame(grid)
    per_net = grid_df.merge(denom, on=[network_col, 'seed_bin'], how='left').merge(sig_counts,
                                                                                   on=['_combo', network_col,
                                                                                       'seed_bin'], how='left')
    per_net['n_sig'] = per_net['n_sig'].fillna(0.0)
    per_net['pct'] = np.where(per_net['denom'] > 0, 100.0 * per_net['n_sig'] / per_net['denom'], np.nan)

    # Aggregates across networks
    # Mean across networks (ignore NaNs); SD across networks (population SD, ddof=0)
    mean_map = {}
    sd_map = {}
    pooled_denom_map = {}
    pooled_sig_map = {}

    for combo in all_combos:
        df_c = per_net[per_net['_combo'] == combo]
        # Per seed, compute mean and SD across networks
        grp = df_c.groupby('seed_bin')
        means = grp['pct'].mean().to_dict()
        sds = grp['pct'].agg(lambda s: float(np.std(np.asarray(list(s.dropna()), dtype=float), ddof=0)) if len(
            s.dropna()) else np.nan).to_dict()
        # Also pooled totals for 'stderr' if needed
        pooled = grp[['n_sig', 'denom']].sum()
        pooled_sig = pooled['n_sig'].to_dict()
        pooled_denom = pooled['denom'].to_dict()
        for sbin in set(list(means.keys()) + list(sds.keys()) + list(pooled_sig.keys()) + list(pooled_denom.keys())):
            mean_map[(combo, int(sbin))] = float(means.get(sbin, np.nan))
            sd_map[(combo, int(sbin))] = float(sds.get(sbin, np.nan))
            pooled_sig_map[(combo, int(sbin))] = float(pooled_sig.get(sbin, 0.0))
            pooled_denom_map[(combo, int(sbin))] = float(pooled_denom.get(sbin, 0.0))

    # Prepare plotting canvas
    fig_w = max(10, 0.45 * len(seeds_order) + 8)
    fig_h = 5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Jitter offsets per combo to reduce overlap
    if jitter and len(all_combos) > 1:
        offsets = np.linspace(-0.12, 0.12, len(all_combos))
    else:
        offsets = np.zeros(len(all_combos))
    offset_map = {combo: float(off) for combo, off in zip(all_combos, offsets)}

    # Plot each combo with optional error bars
    for combo in all_combos:
        y = []
        x = []
        yerr = []
        yerr_lower = []  # not used, kept for parity
        yerr_upper = []
        for s in seeds_order:
            mu = mean_map.get((combo, s), np.nan)
            x.append(s + offset_map[combo])
            y.append(mu)
            if show_error == 'sd_net':
                yerr.append(sd_map.get((combo, s), np.nan))
            elif show_error == 'stderr':
                N = pooled_denom_map.get((combo, s), 0.0)
                K = pooled_sig_map.get((combo, s), 0.0)
                if N > 0.0 and np.isfinite(mu):
                    p = (K / N)
                    yerr.append(100.0 * np.sqrt(p * (1.0 - p) / N))
                else:
                    yerr.append(np.nan)
            else:
                yerr.append(np.nan)

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        if show_error in ('sd_net', 'stderr'):
            ax.errorbar(x, y, yerr=np.asarray(yerr, dtype=float), fmt='-o', linewidth=1.2, markersize=4, elinewidth=0.8,
                        capsize=capsize, alpha=0.9, label=combo)
        else:
            ax.plot(x, y, marker='o', linewidth=1.2, markersize=4, label=combo)

    # Axes styling
    ax.set_xticks([1, 5, 10, 15, 20, 25, 26])
    ax.set_xticklabels(['1', '5', '10', '15', '20', '25', '>25'])
    ax.set_ylim(0.0, 50.0)
    ax.set_xlabel('Seed count (1..25; >25 aggregated)')
    ax.set_ylabel('% significant across diseases (mean over networks)')
    # Subtle gridlines: major every 10%, minor dashed every 5% (clear without clutter)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.6, alpha=0.25, color='0.5')
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.15, color='0.5')
    title_err = {'sd_net': 'with across-network SD bars', 'stderr': 'with pooled binomial s.e. bars',
                 'none': 'no error bars'}.get(show_error, 'with across-network SD bars')

    ax.set_title('TrustRank-only: percent significant vs seed count (averaged over networks)\n'
                 f'Candidates: significance by {significance_col!r}; denominators = total diseases per network×seed; {title_err}')

    # Legend outside plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    out_pdf = os.path.join(plots_dir, 'seed_stratified_percent_significant_overall.pdf')
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved seed-stratified percent-significant plot to %s", out_pdf)


def remove_irrelevant_rows(results_df, args_networks, module_algorithms, prioritization_algorithms):
    if args_networks:
        networks = args_networks.split(',')
        results_df = results_df[results_df['ppi'].isin(networks)]
        logger.info("Filtered results for networks: %s", networks)
    if module_algorithms:
        module_algorithms = module_algorithms.split(',')
        results_df = results_df[results_df['algorithm'].isin(module_algorithms)]
        logger.info("Filtered results for module algorithms: %s", module_algorithms)
    if prioritization_algorithms:
        prioritization_algorithms = prioritization_algorithms.split(',')
        results_df = results_df[results_df['prioritization_algorithm'].isin(prioritization_algorithms)]
        logger.info("Filtered results for prioritization algorithms: %s", prioritization_algorithms)
    return results_df


# --- New: generic 2D hexbin + regression grid --------------------------------


def plot_hexbin_regression_grid(
        results_df,
        plots_dir,
        *,
        x_col='p_value_DCG',
        y_col='p_value_without_ranks',
        stratify_cols=('prioritization_algorithm',),
        # hexbin settings
        gridsize=40,
        mincnt=1,  # minimum count per hex to display
        limit_to_unit_interval=True,
        # overlays
        draw_identity=True,
        min_points_for_fit=3,
        # layout
        sharex=True,
        sharey=True,
):
    """
    Hexbin grid of y_col vs x_col with per-panel linear regression.
    - stratify_cols: 1 -> single row; 2 -> row x col grid; >=3 -> near-square of all combos.
    - Shared LogNorm color scale across all panels with a single labeled colorbar on the right.
    - Panel legend shows: y = a + b x, p = <p>, R^2 = <r2>, n = <n>
    """
    # keep only needed columns and coerce numeric
    keep = list(stratify_cols) + [x_col, y_col]
    df = results_df[keep].dropna().copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        logger.warning("No data for hexbin grid after dropping NA in %r and %r.", x_col, y_col)
        return

    # unique levels per stratifier
    uniq = {c: sorted(pd.Series(df[c].astype(str)).dropna().unique()) for c in stratify_cols}

    # layout
    if len(stratify_cols) == 1:
        levels = uniq[stratify_cols[0]]
        nrows, ncols = 1, max(1, len(levels))
        panel_keys = [(lvl,) for lvl in levels]
        title_of = {(lvl,): f"{stratify_cols[0]} = {lvl}" for lvl in levels}
    elif len(stratify_cols) == 2:
        r_levels = uniq[stratify_cols[0]]
        c_levels = uniq[stratify_cols[1]]
        nrows, ncols = max(1, len(r_levels)), max(1, len(c_levels))
        panel_keys = [(r, c) for r in r_levels for c in c_levels]
        title_of = {(r, c): f"{stratify_cols[0]}={r} | {stratify_cols[1]}={c}" for r in r_levels for c in c_levels}
    else:
        from itertools import product
        combos = list(product(*[uniq[c] for c in stratify_cols]))
        panel_keys = combos
        nrows, ncols = _auto_grid(len(panel_keys))
        title_of = {key: ", ".join(f"{c}={v}" for c, v in zip(stratify_cols, key)) for key in panel_keys}

    # figure geometry
    fig_w = max(8.0, ncols * 3.2)
    fig_h = max(3.0, nrows * 3.0)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h),
        sharex=sharex, sharey=sharey, squeeze=False,
        constrained_layout=True  # avoids overlap and keeps colorbar on the right cleanly
    )
    axes = axes.ravel()

    # axis limits
    if limit_to_unit_interval:
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
    else:
        x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
        y_min, y_max = float(df[y_col].min()), float(df[y_col].max())
        pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
        pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_y, y_max + pad_y)

    # first pass: draw hexbins to capture max counts, add regression
    hb_artists = []
    global_max = 1
    for ax, key in zip(axes, panel_keys):
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(stratify_cols, key):
            mask &= (df[col].astype(str) == str(val))
        d = df.loc[mask, [x_col, y_col]].dropna()

        # hexbin counts (linear counts, no norm yet)
        hb = ax.hexbin(
            d[x_col].values, d[y_col].values,
            gridsize=gridsize, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            mincnt=mincnt, linewidths=0.0
        )
        hb_artists.append(hb)
        if hb.get_array().size:
            global_max = max(global_max, int(np.nanmax(hb.get_array())))

        # identity
        if draw_identity:
            lo = max(xlim[0], ylim[0]);
            hi = min(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1, alpha=0.6)

        # regression if enough points
        if len(d) >= min_points_for_fit:
            try:
                lr = linregress(d[x_col].values, d[y_col].values)
                slope = float(lr.slope);
                intercept = float(lr.intercept)
                r2 = float(lr.rvalue ** 2);
                pval = float(lr.pvalue)
                xs = np.linspace(xlim[0], xlim[1], 200)
                ys = intercept + slope * xs
                ax.plot(
                    xs, ys,
                    linestyle='--', alpha=0.7, linewidth=1.5,
                    label=f"y = {intercept:.3f} + {slope:.3f} x\np = {pval:.3g}, R² = {r2:.3f}, n = {len(d)}"
                )
                ax.legend(loc='lower right', frameon=False, fontsize=8)
            except Exception as e:
                logger.warning("Regression failed for panel %s: %s", key, e)

        ax.set_title(title_of[key], fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # shared log normalization across all panels
    norm = LogNorm(vmin=max(1, mincnt), vmax=global_max)
    for hb in hb_artists:
        hb.set_norm(norm)

    # single labeled colorbar on the right spanning all axes
    cbar = fig.colorbar(hb_artists[0], ax=axes, location='right')
    cbar.set_label('count per hex (log scale)')

    # global labels and save
    fig.supxlabel(x_col)
    fig.supylabel(y_col)
    fig.suptitle(f"{y_col} vs {x_col} stratified by {', '.join(stratify_cols)}")

    out_pdf = os.path.join(
        plots_dir,
        f"hexbin__{_safe_slug(y_col)}__vs__{_safe_slug(x_col)}__by__{'__'.join(_safe_slug(c) for c in stratify_cols)}.pdf"
    )
    plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    logger.info("Saved hexbin regression grid to %s", out_pdf)


# ---- Logistic regression model for significance ----

# --- Generic helpers for the logistic-regression refactor ---
import re as _re_for_slug


def _safe_slug(text):
    """Return a filesystem-safe, lowercase slug for filenames."""
    if text is None:
        return ""
    return _re_for_slug.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_").lower()


def _append_implied_for_factor_generic(res_full, df, factor, *, effect_type="factor"):
    """
    For Sum-coded factor `factor` present in `res_full` (statsmodels GLM result),
    append the implied last level coefficient and its uncertainty using the model
    covariance. Returns a list with zero or one dict rows suitable for concatenation
    to a coefficient table. Skips if the factor isn't Sum-coded or inference is
    ambiguous.
    """
    import numpy as _np
    try:
        cov = res_full.cov_params()
    except Exception:
        return []

    params = res_full.params

    prefix = f"C({factor}, Sum)[S."
    present_terms = [t for t in params.index if t.startswith(prefix)]
    if not present_terms:
        return []

    present_levels = [t[len(prefix):-1] for t in present_terms]
    all_levels = sorted(df[factor].astype(str).unique())
    missing_levels = [lvl for lvl in all_levels if lvl not in present_levels]
    if len(missing_levels) != 1:
        return []  # ambiguous, bail out

    miss = missing_levels[0]
    # implied coefficient is negative sum of present ones
    b = -float(sum(params[t] for t in present_terms))
    try:
        subcov = cov.loc[present_terms, present_terms].values
        var = float(_np.sum(subcov))
        se = float(_np.sqrt(var)) if var >= 0 else float("nan")
    except Exception:
        se = float("nan")

    try:
        from scipy.stats import norm as _norm
        z = (b / se) if se > 0 else float("nan")
        p = float(2.0 * (1.0 - _norm.cdf(abs(z)))) if _np.isfinite(z) else float("nan")
    except Exception:
        z = float("nan");
        p = float("nan")

    ci_low = b - 1.96 * se if _np.isfinite(se) else float("nan")
    ci_high = b + 1.96 * se if _np.isfinite(se) else float("nan")

    return [{'term': f"C({factor}, Sum)[S.{miss}]", 'effect_type': effect_type, 'coef_logit': b, 'std_err': se,
             'z_value': z, 'p_value': p, 'odds_ratio': float(_np.exp(b)) if _np.isfinite(b) else float('nan'),
             'ci_low_or': float(_np.exp(ci_low)) if _np.isfinite(ci_low) else float('nan'),
             'ci_high_or': float(_np.exp(ci_high)) if _np.isfinite(ci_high) else float('nan'), }]


def _binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def sdiv(a, b):
        return float(a) / float(b) if b else float('nan')

    acc = sdiv(tp + tn, tp + tn + fp + fn)
    precision = sdiv(tp, tp + fp)
    recall_pos = sdiv(tp, tp + fn)  # sensitivity / TPR
    recall_neg = sdiv(tn, tn + fp)  # specificity / TNR
    f1 = sdiv(2 * tp, 2 * tp + fp + fn)
    npv = sdiv(tn, tn + fn)
    bal_acc = sdiv(
        (recall_pos if not np.isnan(recall_pos) else 0.0) + (recall_neg if not np.isnan(recall_neg) else 0.0), 2.0)
    prevalence = sdiv(tp + fn, tp + tn + fp + fn)
    p_pred = sdiv(tp + fp, tp + tn + fp + fn)

    return {'threshold': threshold, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': acc, 'precision': precision,
            'recall_pos': recall_pos, 'recall_neg': recall_neg, 'f1': f1, 'npv': npv, 'balanced_accuracy': bal_acc,
            'prevalence': prevalence, 'pred_positive_rate': p_pred, }


def _roc_pr_from_scores(y_true, scores):
    """Return ROC and PR arrays for given scores.
    y_true: array-like of binary labels {0,1}
    scores: array-like of predicted probabilities for the positive class
    Returns: (fpr, tpr, precision, recall, roc_auc, avg_precision)
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    avg_precision = average_precision_score(y_true, scores)
    return fpr, tpr, precision, recall, roc_auc, avg_precision


def plot_roc_pr_curves(y_true, prob_dict, plots_dir, out_prefix="logistic"):
    """Plot ROC and PR curves for multiple models and write a summary CSV.
    prob_dict: mapping of {model_name: probability_array}
    Saves two PDFs and a CSV with AUCs. Filenames are prefixed by `out_prefix`.
    """
    # Ensure arrays
    y_true = np.asarray(y_true).astype(int)

    # Compute metrics for each model
    rows = []
    roc_curves = {}
    pr_curves = {}
    for name, prob in prob_dict.items():
        fpr, tpr, precision, recall, roc_auc, avg_prec = _roc_pr_from_scores(y_true, np.asarray(prob).astype(float))
        roc_curves[name] = (fpr, tpr, roc_auc)
        pr_curves[name] = (recall, precision, avg_prec)
        rows.append({'model': name, 'roc_auc': float(roc_auc), 'average_precision': float(avg_prec),
                     'prevalence': float(y_true.mean())})

    # Save summary CSV
    auc_df = pd.DataFrame(rows).sort_values('model')
    out_csv = os.path.join(plots_dir, f"{out_prefix}__roc_pr_auc.csv")
    auc_df.to_csv(out_csv, index=False)
    logger.info("Saved ROC/PR AUC summary to %s", out_csv)

    # ROC plot
    fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5))
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.05)
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('ROC curves')
    ax_roc.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    out_pdf_roc = os.path.join(plots_dir, f"{out_prefix}__roc_curves.pdf")
    plt.savefig(out_pdf_roc)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_roc)
    logger.info("Saved ROC curves to %s", out_pdf_roc)

    # Precision-Recall plot
    fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5))
    base_prec = float(y_true.mean()) if y_true.size else np.nan
    for name, (recall, precision, avg_prec) in pr_curves.items():
        ax_pr.plot(recall, precision, label=f"{name} (AP={avg_prec:.3f})")
    # Optional no-skill line at prevalence if available
    if not np.isnan(base_prec):
        ax_pr.hlines(base_prec, xmin=0.0, xmax=1.0, linestyles='--', linewidth=1)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.05)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall curves')
    ax_pr.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    out_pdf_pr = os.path.join(plots_dir, f"{out_prefix}__pr_curves.pdf")
    plt.savefig(out_pdf_pr)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_pr)
    logger.info("Saved PR curves to %s", out_pdf_pr)


def fit_logistic_models_generic(results_df, plots_dir, *, y_col='significant',
                                # Optional convenience columns used to construct _combo if referenced in formulas
                                sample_col='sample', network_col='ppi', algorithm_col='algorithm',
                                prio_col='prioritization_algorithm',
                                build_combo_if_needed=True,  # Formulas (required)
                                formula_reduced=None, formula_full=None,
                                # Factors to infer implied Sum-coded last levels for (order shown first in output)
                                factors_for_implied=None,  # e.g., ['_combo', 'ppi']
                                # Which prefixes to drop from the coefficient table (e.g., disease FE)
                                drop_term_prefixes=None,  # e.g., (f"C({sample_col}",)
                                # Output tagging
                                id_tag=None, add_roc_pr=True, ):
    """
    Fit user-specified reduced and full Binomial GLMs, run an LRT, export coefficients
    (with optional Sum-coded implied levels), and write ROC/PR curves.

    Parameters
    ----------
    results_df : pd.DataFrame
        Input data.
    plots_dir : str
        Output directory for artifacts.
    y_col : str
        Binary target column name.
    sample_col, network_col, algorithm_col, prio_col : str
        Column names. `_combo` can be auto-built if referenced in formulas and requested.
    formula_reduced, formula_full : str
        Patsy formulas including `y_col` on LHS. Use Sum coding explicitly where desired, e.g.,
        `C(sample, Sum) + C(ppi, Sum) + C(_combo, Sum)`.
    factors_for_implied : list[str] | None
        Factor names for which to reconstruct the omitted Sum-coded level.
    drop_term_prefixes : tuple[str] | None
        Term-string prefixes to exclude from the coefficient table (e.g., disease FE terms).
    id_tag : str | None
        Optional short tag used to prefix output filenames. If None, a tag is derived from formulas.
    add_roc_pr : bool
        If True, write ROC/PR plots and AUC CSV.

    Returns dict with artifact paths and LRT summary.
    """
    import numpy as _np
    import pandas as _pd
    import statsmodels.formula.api as _smf
    import statsmodels.api as _sm
    from scipy.stats import chi2 as _chi2

    if formula_reduced is None or formula_full is None:
        raise ValueError("Both formula_reduced and formula_full must be provided.")

    df = results_df.copy()

    # Build _combo if used in formulas and requested
    need_combo = ('_combo' in str(formula_reduced)) or ('_combo' in str(formula_full))
    if need_combo and build_combo_if_needed:
        df['_combo'] = df[algorithm_col].astype(str) + ' | ' + df[prio_col].astype(str)

    # Ensure binary numeric target
    df[y_col] = _pd.to_numeric(df[y_col], errors='coerce').astype(int)

    # Fit GLM Binomial models
    logger.info("Fitting reduced logistic GLM: %s", formula_reduced)
    mod_red = _smf.glm(formula=formula_reduced, data=df, family=_sm.families.Binomial())
    res_red = mod_red.fit()

    logger.info("Fitting full logistic GLM: %s", formula_full)
    mod_full = _smf.glm(formula=formula_full, data=df, family=_sm.families.Binomial())
    res_full = mod_full.fit()

    # Likelihood-ratio test for the added block(s)
    llf_full = float(res_full.llf)
    llf_red = float(res_red.llf)
    df_diff = int(res_full.df_model - res_red.df_model)
    lr_stat = 2.0 * (llf_full - llf_red)
    lr_p = float(_chi2.sf(lr_stat, df_diff)) if df_diff > 0 else float('nan')

    # Baseline intercept-only for reference
    logger.info("Fitting baseline logistic GLM: intercept only")
    mod_base = _smf.glm(formula=f"{y_col} ~ 1", data=df, family=_sm.families.Binomial())
    res_base = mod_base.fit()

    # ROC/PR curves with AUCs (baseline/reduced/full)
    y_true = df[y_col].values
    prob_base = res_base.predict(df)
    prob_red = res_red.predict(df)
    prob_full = res_full.predict(df)

    # Output prefix/tag
    if id_tag is None:
        tag_left = _safe_slug(formula_reduced)
        tag_right = _safe_slug(formula_full)
        id_tag = f"{_safe_slug(y_col)}__{tag_left}__vs__{tag_right}"[:160]
    out_prefix = f"logreg__{id_tag}"

    if add_roc_pr:
        plot_roc_pr_curves(y_true, {'baseline': prob_base, 'reduced': prob_red, 'full': prob_full, }, plots_dir,
                           out_prefix=out_prefix)

    # Extract coefficients from the FULL model, excluding requested term prefixes
    params = res_full.params
    bse = res_full.bse
    pvals = res_full.pvalues
    conf = res_full.conf_int()

    # defaults for dropping disease FE and similar
    if drop_term_prefixes is None:
        drop_term_prefixes = (f"C({sample_col}",)

    def _keep_term(t):
        if t == 'Intercept':
            return False
        return not any(t.startswith(pref) for pref in drop_term_prefixes)

    rows = []
    for term in params.index:
        if not _keep_term(term):
            continue
        coef = float(params[term])
        se = float(bse[term])
        p = float(pvals[term])
        z = coef / se if se > 0 else float('nan')
        ci_low, ci_high = conf.loc[term]
        or_val = _np.exp(coef)
        or_low = _np.exp(ci_low)
        or_high = _np.exp(ci_high)
        # Map effect type by factor name if possible
        if term.startswith(f"C({network_col}"):
            effect_type = 'network'
        elif term.startswith("C(_combo"):
            effect_type = 'method'
        else:
            effect_type = 'other'
        rows.append(
            {'term': term, 'effect_type': effect_type, 'coef_logit': coef, 'std_err': se, 'z_value': z, 'p_value': p,
             'odds_ratio': or_val, 'ci_low_or': or_low, 'ci_high_or': or_high, })

    # Append implied last levels for any requested Sum-coded factors
    if factors_for_implied:
        for fac in factors_for_implied:
            rows.extend(_append_implied_for_factor_generic(res_full, df, fac, effect_type=(
                'method' if fac == '_combo' else ('network' if fac == network_col else 'factor'))))

    coef_df = pd.DataFrame(rows)
    if not coef_df.empty:
        # Sort: requested factors first by effect_type, then term label
        order_map = {'method': 0, 'network': 1, 'factor': 2, 'other': 3}
        coef_df['sort_key'] = coef_df['effect_type'].map(order_map).fillna(9)
        coef_df = coef_df.sort_values(['sort_key', 'term']).drop(columns=['sort_key'])

    # Save outputs
    out_coef = os.path.join(plots_dir, f"{out_prefix}__coefficients.csv")
    coef_df.to_csv(out_coef, index=False)

    out_lrt = os.path.join(plots_dir, f"{out_prefix}__lrt.txt")
    with open(out_lrt, 'w') as fh:
        fh.write('Logistic GLM (Binomial) LRT for added terms\n')
        fh.write(f'Reduced: {formula_reduced}\n')
        fh.write(f'Full   : {formula_full}\n')
        fh.write(f'LL(full) = {llf_full:.3f}, LL(reduced) = {llf_red:.3f}\n')
        fh.write(f'df diff = {df_diff}\n')
        fh.write(f'LR stat = {lr_stat:.3f}\n')
        fh.write(f'p-value = {lr_p:.6g}\n')

    logger.info("Saved logistic coefficients to %s", out_coef)
    logger.info("Saved LRT summary to %s", out_lrt)

    return {'coef_path': out_coef, 'lrt_path': out_lrt, 'lrt': {'lr': lr_stat, 'df': df_diff, 'p': lr_p},
            'tag': id_tag, }


def fit_logistic_success_models_methods(results_df, plots_dir, significance_col='significant', sample_col='sample',
                                        network_col='ppi', algorithm_col='algorithm',
                                        prio_col='prioritization_algorithm'):
    """
    Backward-compatible wrapper that reproduces the previous behavior using the new
    generic driver. You can call the new `fit_logistic_models_generic` directly for
    arbitrary model comparisons.

    Model:
        reduced:   significant ~ C(sample, Sum) + C(network, Sum)
        full:      significant ~ C(sample, Sum) + C(network, Sum) + C(_combo, Sum)
    where _combo = algorithm | prioritization_algorithm
    """
    formula_reduced = (f"{significance_col} ~ C({sample_col}, Sum) + C({network_col}, Sum)")
    formula_full = (f"{significance_col} ~ C({sample_col}, Sum) + C({network_col}, Sum) + C(_combo, Sum)")
    return fit_logistic_models_generic(results_df, plots_dir, y_col=significance_col, sample_col=sample_col,
                                       network_col=network_col, algorithm_col=algorithm_col, prio_col=prio_col,
                                       build_combo_if_needed=True,
                                       formula_reduced=formula_reduced, formula_full=formula_full,
                                       factors_for_implied=['_combo', network_col],
                                       drop_term_prefixes=(f"C({sample_col}",),
                                       id_tag=f"default__{_safe_slug(network_col)}__with_combo",
                                       add_roc_pr=True, )


if __name__ == '__main__':
    main()
