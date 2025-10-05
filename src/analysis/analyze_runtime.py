#!/usr/bin/env python3
import argparse
import logging
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from analysis.analyze_cross_network_results import read_input_tsv

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


# NEW: helper to add a 'network' column from process_label
def add_network_column(df):
    pl = df['process_label'].astype(str)
    # default to None (will become missing in pandas)
    df['network'] = None
    mask_string = pl.str.contains('string.human_links_v12_0_min700', case=False, na=False)
    df.loc[mask_string, 'network'] = 'string_min700'
    mask_nedrex = pl.str.contains('nedrex.reviewed_proteins_exp', case=False, na=False)
    df.loc[mask_nedrex, 'network'] = 'NeDRex'
    return df


def parse_arguments():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --input_tsv with default
    parser.add_argument("--input_tsv", type=str, default="../../data/input/execution_trace_2025-09-19_12-00-42.txt",
                        help="Input TSV file with runtime data")
    parser.add_argument("--plots_dir", default="../../plots/runtime", type=str, help="Directory to save plots")
    parser.add_argument("--no_show", action="store_true", help="Do not show plots interactively")
    args = parser.parse_args()
    if args.no_show:
        global SHOW_PLOTS
        SHOW_PLOTS = False
    return args


def main():
    args = parse_arguments()
    logger.info("Starting analysis, with arguments: %s", args)

    plots_dir = args.plots_dir
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logger.info("Created plots directory: %s", plots_dir)

    results_df = read_input_tsv(args.input_tsv)

    # 1) split and clean names
    split_and_clean_process_columns(results_df)

    # 2) numeric conversions
    clean_numeric_columns(results_df)

    results_df = results_df[~results_df['process_name'].isin(
        ['MULTIQC', 'SAVEMODULES', 'INPUTCHECK', 'DRUGSTONEEXPORT', 'MODULEPARSER', 'NETWORKANNOTATION', 'TOPOLOGY',
         'GT_ROBUST:GRAPHTOOLPARSER', 'GT_ROBUSTBIASAWARE:GRAPHTOOLPARSER', 'GT_DIAMOND:GRAPHTOOLPARSER',
         'GT_DOMINO:GRAPHTOOLPARSER', 'GT_RWR:GRAPHTOOLPARSER', 'GRAPHTOOLPARSER', 'PREFIXLINES'])]
    PREFERRED_ORDER = ['DOWNLOAD_DRUG', 'DOMINO_SLICER', 'DIAMOND', 'DOMINO_DOMINO', 'FIRSTNEIGHBOR', 'ROBUST',
                       'ROBUSTBIASAWARE', 'RWR', 'DRUGPREDICTIONS', 'PRIORITIZATIONEVALUATION']

    # NEW in main(): after filtering the processes you want
    results_df = add_network_column(results_df)

    # 3) plots
    metrics = ['realtime', '%cpu', 'peak_rss', 'peak_vmem', 'rchar', 'wchar']
    for m in metrics:
        paths = plot_metric_box_violin(results_df, m, plots_dir, show_plots=SHOW_PLOTS, add_dots=True, dot_jitter=0.12,
                                       dot_size=10, dot_alpha=0.35)

        logger.info("Saved %s plots: %s", m, paths)


def split_and_clean_process_columns(df):
    """
    - Split 'name' into 'process_name' and 'process_label'
    - Strip NF-core prefixes from process_name
    - Replace the original 'name' column
    """
    parts = df['name'].astype(str).str.strip().str.split(' ', n=1, expand=True)
    # remove duplicates if they already exist
    for c in ('process_name', 'process_label'):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    i = df.columns.get_loc('name')
    df.insert(i, 'process_name', parts[0])
    df.insert(i + 1, 'process_label', parts[1].fillna(''))
    df.drop(columns='name', inplace=True)

    # strip both possible prefixes at start
    df['process_name'] = df['process_name'].str.replace(
        r'^NFCORE_DISEASEMODULEDISCOVERY:DISEASEMODULEDISCOVERY:(?:NETWORKEXPANSION:)?(?:GT_ROBUST(BIASAWARE)?:)?(?:GT_RWR:)?(?:GT_DOMINO:)?(?:GT_DIAMOND:)?(?:GT_FIRSTNEIGHBOR:)?',
        '', regex=True)
    return df


def _to_seconds(x):
    """Accepts numbers, '4.8s', '2m 10s', '1h 2m', or hh:mm:ss(.ms). Returns float seconds."""
    import numpy as _np
    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return _np.nan
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().lower()
    if not s:
        return _np.nan

    # hh:mm[:ss[.ms]]
    if ':' in s and all(p.isdigit() or ('.' in p and p.replace('.', '', 1).isdigit()) for p in s.split(':')):
        parts = s.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, sec = parts
        elif len(parts) == 2:
            h, m, sec = 0.0, parts[0], parts[1]
        else:
            # treat as seconds
            return float(parts[0])
        return h * 3600.0 + m * 60.0 + sec

    # token list like "1h 2m 3.5s"
    import re as _re
    total = 0.0
    for val, unit in _re.findall(r'(\d+(?:\.\d+)?)\s*(ms|[smhd])', s):
        v = float(val)
        if unit == 's':
            total += v
        elif unit == 'ms':
            total += v * 0.001
        elif unit == 'm':
            total += v * 60.0
        elif unit == 'h':
            total += v * 3600.0
        elif unit == 'd':
            total += v * 86400.0
        else:
            logger.error('Unexpected time unit: %s', unit)
    if total > 0.0:
        return total

    # bare number fallback
    try:
        return float(s)
    except Exception:
        return _np.nan


def _to_megabytes(x):
    """Parses '151.5 MB', '2 GB', '2048', '36.4 MiB' into float MB. Numeric values are treated as bytes if large."""
    import numpy as _np, re as _re
    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return _np.nan
    if isinstance(x, (int, float)):
        # Heuristic: treat big numbers as bytes, small as already MB
        return float(x) / (1024.0 * 1024.0) if float(x) >= 4096 else float(x)

    s = str(x).strip()
    m = _re.match(r'^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?i?B)?\s*$', s, flags=_re.I)
    if not m:
        # try bare float
        try:
            v = float(s)
            return v / (1024.0 * 1024.0) if v >= 4096 else v
        except Exception:
            return _np.nan

    val = float(m.group(1))
    unit = (m.group(2) or '').upper()

    # bytes to MB factors
    factors = {'B': 1.0 / (1024.0 * 1024.0), 'KB': 1024.0 / (1024.0 * 1024.0), 'KIB': 1024.0 / (1024.0 * 1024.0),
               'MB': 1.0, 'MIB': 1.0, 'GB': 1024.0, 'GIB': 1024.0, 'TB': 1024.0 * 1024.0, 'TIB': 1024.0 * 1024.0,
               '': 1.0,  # assume MB if omitted
               }
    return val * factors.get(unit, 1.0)


def _to_percent(x):
    """Parses '35.5%' or numeric into float percent."""
    import numpy as _np
    if x is None or (isinstance(x, float) and _np.isnan(x)):
        return _np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace('%', '')
    try:
        return float(s)
    except Exception:
        return _np.nan


def clean_numeric_columns(df):
    """
    Convert columns to numeric with consistent units:
      - realtime -> seconds (float)
      - %cpu -> percent (float)
      - peak_rss, peak_vmem, rchar, wchar -> MB (float)
    """
    converters = {'realtime': _to_seconds, '%cpu': _to_percent, 'peak_rss': _to_megabytes, 'peak_vmem': _to_megabytes,
                  'rchar': _to_megabytes, 'wchar': _to_megabytes, }
    for col, fn in converters.items():
        if col in df.columns:
            df[col] = df[col].map(fn)
    return df


def _metric_title_and_fname(metric):
    title_map = {'realtime': 'Realtime (s)', '%cpu': 'CPU (%)', 'peak_rss': 'Peak RSS (MB)',
                 'peak_vmem': 'Peak VMEM (MB)', 'rchar': 'Read bytes (MB)', 'wchar': 'Written bytes (MB)', }
    safe = metric.replace('%', 'pct')
    return title_map.get(metric, metric), f"runtime_{safe}"


# UPDATED: color dots by network and add legend (top-left)
def plot_metric_box_violin(df, metric, out_dir, show_plots=True, add_dots=True, dot_jitter=0.12, dot_size=10,
                           dot_alpha=0.35):
    if metric not in df.columns or 'process_name' not in df.columns:
        return []
    PREFERRED_ORDER = ['DOWNLOAD_DRUG', 'DOMINO_SLICER', 'DIAMOND', 'DOMINO_DOMINO', 'FIRSTNEIGHBOR', 'ROBUST',
                       'ROBUSTBIASAWARE', 'RWR', 'DRUGPREDICTIONS', 'PRIORITIZATIONEVALUATION']

    color_map = {'string_min700': 'blue', 'NeDRex': 'red'}

    grouped = df.groupby('process_name')[metric]
    med = grouped.median()

    labels = [p for p in PREFERRED_ORDER if p in med.index]
    others = med.drop(labels, errors='ignore').sort_values(ascending=False).index.tolist()
    labels += others

    data = [grouped.get_group(p).dropna().values for p in labels]
    if not any(len(v) for v in data):
        return []

    width = max(8, min(24, 0.6 * len(labels) + 2))
    title, base = _metric_title_and_fname(metric)
    saved = []

    # --- BOX ---
    fig1 = plt.figure(figsize=(width, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    if add_dots:
        for xi, p in enumerate(labels, start=1):
            sub = df[df['process_name'] == p][[metric, 'network']].dropna(subset=[metric])
            if sub.empty:
                continue
            x = np.random.uniform(xi - dot_jitter, xi + dot_jitter, size=len(sub))
            colors = sub['network'].map(color_map).fillna('0.5').values
            plt.scatter(x, sub[metric].values, s=dot_size, alpha=dot_alpha, rasterized=True, c=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(title)
    plt.title(f'{title} by process')

    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.20, color='0.5')

    # legend in top-left
    if 'network' in df.columns:
        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_map['string_min700'],
                                 markeredgecolor='none', label='string_min700'),
                          Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_map['NeDRex'],
                                 markeredgecolor='none', label='NeDRex'), ]
        ax.legend(handles=legend_handles, loc='upper left', frameon=False)

    plt.tight_layout()
    box_path = os.path.join(out_dir, f"{base}_box.pdf")
    plt.savefig(box_path, dpi=500)
    saved.append(box_path)
    if show_plots:
        plt.show()
    plt.close(fig1)

    # --- VIOLIN ---
    fig2 = plt.figure(figsize=(width, 6))
    plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha='right')
    if add_dots:
        for xi, p in enumerate(labels, start=1):
            sub = df[df['process_name'] == p][[metric, 'network']].dropna(subset=[metric])
            if sub.empty:
                continue
            x = np.random.uniform(xi - dot_jitter, xi + dot_jitter, size=len(sub))
            colors = sub['network'].map(color_map).fillna('0.5').values
            plt.scatter(x, sub[metric].values, s=dot_size, alpha=dot_alpha, rasterized=True, c=colors)
    plt.ylabel(title)
    plt.title(f'{title} by process')

    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.6, alpha=0.30, color='0.5')
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5, alpha=0.20, color='0.5')

    # legend in top-left
    if 'network' in df.columns:
        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_map['string_min700'],
                                 markeredgecolor='none', label='string_min700'),
                          Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=color_map['NeDRex'],
                                 markeredgecolor='none', label='NeDRex'), ]
        ax.legend(handles=legend_handles, loc='upper left', frameon=False)

    plt.tight_layout()
    violin_path = os.path.join(out_dir, f"{base}_violin.pdf")
    plt.savefig(violin_path, dpi=500)
    saved.append(violin_path)
    if show_plots:
        plt.show()
    plt.close(fig2)

    return saved


if __name__ == '__main__':
    main()
