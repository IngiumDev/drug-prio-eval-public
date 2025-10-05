#!/usr/bin/env python3
"""
Script to process MultiQC JSON data and NedRexDB CSV files, merge statistics and annotations,
and output a combined TSV for downstream analysis.
"""
import argparse
import json
import logging
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

DEFAULT_SIGNIFICANCE_LEVEL = 0.05

TRANSLATE_PPI = {  # STRING - functional links
    "string.human_links_v12_0_min700": "string_min700", "string.human_links_v12_0_min900": "string_min900",

    # STRING - physical links
    "string.human_physical_links_v12_0_min700": "string_physical_min700",
    "string.human_physical_links_v12_0_min900": "string_physical_min900",

    # BioGRID
    "biogrid.4_4_242_homo_sapiens": "biogrid", "biogrid": "biogrid",

    # HIPPIE
    "hippie.v2_3_high_confidence": "hippie_high_confidence",
    "hippie.v2_3_medium_confidence": "hippie_medium_confidence",

    # IID
    "iid.human": "iid", "iid": "iid",

    # NeDRex
    "nedrex:reviewed_proteins_exp": "NeDRex", "nedrex_reviewed_proteins_exp": "NeDRex",
    "nedrex.reviewed_proteins_exp": "NeDRex", "nedrex.reviewed_proteins_exp_high_confidence": "NeDRex_high_confidence",
    "nedrex": "NeDRex", "NeDRexDB_exp": "NeDRex", "NeDRexDB_exp_hc": "NeDRex_high_confidence", }


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_multiqc_data(json_path):
    logging.info(f"Loading MultiQC data from {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)


def process_general_stats(multiqc_data):
    input_df = pd.DataFrame.from_dict(multiqc_data['report_saved_raw_data']['multiqc_input_1'], orient='index')
    general_stats = pd.DataFrame.from_dict(
        {k: v for d in multiqc_data['report_general_stats_data'] for k, v in d.items()}, orient='index')
    if 'Not in network' in input_df.columns:
        seeds_not_in = input_df['Not in network'].rename('seeds_not_in_network')
        merged = general_stats.merge(seeds_not_in, left_on=general_stats.index.str.rsplit('.', n=1).str[0],
                                     right_index=True, how='left')
    else:
        logging.warning("'Not in network' column missing in MultiQC input; 'seeds_not_in_network' will be NaN")
        merged = general_stats.copy()
        merged['seeds_not_in_network'] = pd.NA
    merged.reset_index(inplace=True)
    # Split index into sample, ppi, namespace, and algorithm
    parts = merged['index'].str.rsplit('.', n=2, expand=True)
    parts.columns = ['prefix', 'namespace', 'algorithm']
    # Split prefix into sample (first two segments) and ppi (the rest)
    temp = parts['prefix'].str.split('.', n=2, expand=True)
    temp.columns = ['sg', 'id', 'ppi']
    temp['sample'] = temp['sg'] + '.' + temp['id']
    sample_ppi = temp[['sample', 'ppi']]
    cols = pd.concat([sample_ppi, parts[['namespace', 'algorithm']]], axis=1)
    merged = pd.concat([cols, merged.drop(columns='index')], axis=1)
    return merged


def process_prioritization(multiqc_data):
    df = pd.DataFrame.from_dict(multiqc_data['report_saved_raw_data']['multiqc_prioritizationevaluation'],
                                orient='index')
    df.reset_index(inplace=True)
    # Split index into sample, ppi, namespace, algorithm, and prioritization_algorithm
    parts = df['index'].str.rsplit('.', n=3, expand=True)
    parts.columns = ['prefix', 'namespace', 'algorithm', 'prioritization_algorithm']
    # Split prefix into sample (first two segments) and ppi (the rest)
    temp = parts['prefix'].str.split('.', n=2, expand=True)
    temp.columns = ['sg', 'id', 'ppi']
    temp['sample'] = temp['sg'] + '.' + temp['id']
    sample_ppi = temp[['sample', 'ppi']]
    cols = pd.concat([sample_ppi, parts[['namespace', 'algorithm', 'prioritization_algorithm']]], axis=1)
    df = pd.concat([cols, df.drop(columns='index')], axis=1)
    df.rename(columns={'empirical_p_value_without_considering_ranks': 'p_value_without_ranks',
                       'empirical_DCG_based_p_value': 'p_value_DCG'}, inplace=True)
    return df


def load_nedrex_db(path, files):
    logging.info(f"Loading NedRexDB CSV files from {path}")
    dfs = {}
    for name, fname in files.items():
        full = os.path.join(path, fname)
        logging.info(f"Reading {name} from {full}")
        dfs[name] = pd.read_csv(full)
    return dfs


def compute_drug_stats(combined_df, drug_has_indication, drug_df, drug_has_target):
    drug_counts = drug_has_indication.groupby('targetDomainId')['sourceDomainId'].nunique()
    approved = drug_df.loc[drug_df['drugGroups'].apply(lambda grp: 'approved' in grp), 'primaryDomainId']
    targets = drug_has_target.groupby('sourceDomainId')['targetDomainId'].nunique()
    approved_counts = \
        drug_has_indication[drug_has_indication['sourceDomainId'].isin(approved)].groupby('targetDomainId')[
            'sourceDomainId'].nunique()

    combined_df['num_drugs'] = combined_df['sample'].map(drug_counts).fillna(0).astype(int)
    combined_df['num_approved_drugs'] = combined_df['sample'].map(approved_counts).fillna(0).astype(int)
    combined_df['num_targets'] = combined_df['sample'].map(targets).fillna(0).astype(int)
    # Number of approved drugs that also have targets
    drugs_with_targets = set(drug_has_target['sourceDomainId'].unique())
    approved_with_targets = set(approved).intersection(drugs_with_targets)
    approved_with_targets_counts = (
        drug_has_indication[drug_has_indication['sourceDomainId'].isin(approved_with_targets)].groupby(
            'targetDomainId')['sourceDomainId'].nunique())
    combined_df['approved_drugs_with_targets'] = (
        combined_df['sample'].map(approved_with_targets_counts).fillna(0).astype(int))

    return combined_df


# --- Helper: Add seed genes and scores ---
def add_seed_genes_and_scores(combined: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three columns to `combined`:
      - seed_genes: list of seed gene IDs per disease
      - seed_gene_scores: list of scores aligned 1:1 with seed_genes
      - seed_gene_score_mean: mean of numeric scores per disease (NaN if none)
    """
    logging.info("Adding seed genes, seed scores, and per-disease mean score")

    if 'score' in seed_df.columns:
        tmp = seed_df[['targetDomainId', 'sourceDomainId', 'score']].copy()
        tmp['score'] = pd.to_numeric(tmp['score'], errors='coerce')
        grouped = (tmp.groupby('targetDomainId').agg({'sourceDomainId': list, 'score': list}).rename(
            columns={'sourceDomainId': 'seed_genes', 'score': 'seed_gene_scores'}))
        logging.info("Found 'score' column; building aligned gene-score lists")
    else:
        logging.warning("Column 'score' not found in gene_associated_with_disorder. Seed scores will be empty.")
        grouped = (seed_df.groupby('targetDomainId')['sourceDomainId'].apply(list).to_frame(name='seed_genes'))
        grouped['seed_gene_scores'] = [[] for _ in range(len(grouped))]

    seed_map = grouped['seed_genes'].to_dict()
    score_map = grouped['seed_gene_scores'].to_dict()

    # Map onto combined
    combined = combined.copy()
    combined['seed_genes'] = combined['sample'].map(seed_map).apply(lambda x: x if isinstance(x, list) else [])
    combined['seed_gene_scores'] = combined['sample'].map(score_map).apply(lambda x: x if isinstance(x, list) else [])

    # Mean score per disease (NaN if no numeric scores)
    def _mean_or_nan(seq):
        if not isinstance(seq, list) or len(seq) == 0:
            return np.nan
        s = pd.to_numeric(pd.Series(seq), errors='coerce')
        return float(s.mean()) if s.notna().any() else np.nan

    combined['seed_gene_score_mean'] = combined['seed_gene_scores'].apply(_mean_or_nan)

    n_with_seeds = (combined['seed_genes'].apply(len) > 0).sum()
    logging.info("Seed features added for %d/%d diseases", n_with_seeds, len(combined['sample'].unique()))
    return combined


def main():
    args = parse_arguments()

    setup_logging()

    significance_level = args.significance_level

    # Load and process MultiQC data
    mq_data = load_multiqc_data(os.path.join(args.multiqc_path, args.multiqc_file))
    gen_stats = process_general_stats(mq_data)
    prior_eval = process_prioritization(mq_data)

    logging.info("Merging MultiQC Data")
    combined = gen_stats.merge(prior_eval, on=['sample', 'ppi', 'namespace', 'algorithm'], how='inner')

    # Load NedRexDB data
    files = {'disorder': args.disorder, 'drug': args.drug, 'drug_has_indication': args.drug_has_indication,
             'drug_has_target': args.drug_has_target, 'gene': args.gene,
             'gene_associated_with_disorder': args.gene_associated_with_disorder}
    db = load_nedrex_db(args.nedrex_db_path, files)

    # Load parent/children maps
    logging.info("Loading parent/children maps from %s", args.input_path)
    parents_df = pd.read_csv(os.path.join(args.input_path, args.parents_map))
    children_df = pd.read_csv(os.path.join(args.input_path, args.children_map))
    # Normalize header if leading colon
    parents_df.columns = [col.lstrip(':') for col in parents_df.columns]
    children_df.columns = [col.lstrip(':') for col in children_df.columns]
    # Rename for merge
    parents_df.rename(columns={'disorder': 'sample', 'parents': 'parents'}, inplace=True)
    children_df.rename(columns={'disorder': 'sample', 'children': 'children'}, inplace=True)
    # Merge into combined data
    combined = combined.merge(parents_df, on='sample', how='left')
    combined = combined.merge(children_df, on='sample', how='left')
    # Parse JSON lists or fill empty
    combined['parents'] = combined['parents'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    combined['children'] = combined['children'].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    # Add number of parents and children
    combined['num_parents'] = combined['parents'].apply(len)
    combined['num_children'] = combined['children'].apply(len)

    # Add seed genes and scores
    combined = add_seed_genes_and_scores(combined, db['gene_associated_with_disorder'])

    # 1. Extract exactly one seed‐gene set per disease
    sample_to_seed = (combined.drop_duplicates(subset='sample').set_index('sample')['seed_genes'].apply(set).to_dict())

    metrics_df = calculate_seed_jaccard_index(sample_to_seed)

    metrics_df = calculate_seed_specifity(metrics_df, sample_to_seed)

    # 4. Merge back onto every row of the original `combined` DataFrame
    combined = combined.merge(metrics_df, on='sample', how='left')

    # Compute drug statistics
    combined = compute_drug_stats(combined, db['drug_has_indication'], db['drug'], db['drug_has_target'])

    # Correction
    perform_multiple_testing_correction(combined, args.correction, significance_level)

    output_file = os.path.join(args.output_path, args.output)
    logging.info("Preparing to write combined data to %s (compression=%s)", output_file, args.compression)

    # Remap PPI labels before saving
    combined = remap_ppi_column(combined)

    # Save using helper that respects compression choice and creates directories
    written_path = save_combined_df(combined, output_file, compression=args.compression)
    logging.info("Done. Wrote output to %s", written_path)


def remap_ppi_column(df: pd.DataFrame, column: str = "ppi") -> pd.DataFrame:
    """
    Rename PPI network labels in df[column] using TRANSLATE_PPI.
    Logs an error listing any labels that were not found in the map.
    Returns the modified DataFrame.
    """
    if column not in df.columns:
        logging.error("Column '%s' not found in DataFrame", column)
        return df

    # Find unknowns before replacement
    values = df[column].astype(str)
    unknown = sorted(v for v in values.unique() if v not in TRANSLATE_PPI)

    # Apply mapping (unknowns are left as-is)
    df[column] = values.replace(TRANSLATE_PPI)

    if unknown:
        logging.error("Unmapped PPI labels: %s", ", ".join(unknown))
    else:
        logging.info("All PPI labels successfully remapped")

    return df


def perform_multiple_testing_correction(combined, correction, significance_level):
    if correction == "bh":
        logging.info("Applying Benjamini-Hochberg correction")
        pvals = combined['p_value_DCG'].values

        # multipletests returns:
        # reject: boolean array (True = significant under FDR)
        # pvals_corrected: the BH-adjusted p-values
        reject, pvals_bh, _, _ = multipletests(pvals, alpha=significance_level, method='fdr_bh')

        combined['p_value_bh'] = pvals_bh
        combined['significant'] = reject
        if reject.any():
            highest_rejected = combined.loc[combined['significant'], 'p_value_DCG'].max()
            logging.info(f"Highest p-value rejected under BH: {highest_rejected:.4g}")
            # numrejected
            num_rejected = reject.sum()
            logging.info(f"Number of p-values rejected under BH: {num_rejected}")
        else:
            logging.info("No p-values were rejected under BH.")
    elif correction == "bonferroni":
        logging.info("Applying Bonferroni correction")
        significance_level_adjusted = significance_level / len(combined)
        logging.info(f"Adjusted significance level for Bonferroni correction: {significance_level_adjusted}")
        # Add column significant
        combined['significant'] = combined['p_value_DCG'] <= significance_level_adjusted
        # log amount rejected
        num_rejected = combined['significant'].sum()
        logging.info(f"Number of p-values rejected under Bonferroni: {num_rejected}")
    else:
        logging.warning("No correction method specified; skipping multiple testing correction")


def calculate_seed_specifity(metrics_df, sample_to_seed):
    logging.info("Calculating seed specificity metrics")
    N = len(sample_to_seed)
    # 2. Compute df(g): in how many diseases each gene appears
    #    Flatten all seed lists into one big list of (gene, disease) pairs:
    all_pairs = []
    for sample, genes in sample_to_seed.items():
        all_pairs.extend((g, sample) for g in genes)
    #    Count unique disease occurrences per gene
    df_counts = Counter()
    for gene, sample in set(all_pairs):
        df_counts[gene] += 1
    #    Turn into a pandas Series for easy lookup
    df_series = pd.Series(df_counts, name='df').sort_index()
    # 3. Compute IDF(g) = log((N+1)/(df(g)+1))
    idf_series = np.log((N + 1) / (df_series + 1))
    idf_series.name = 'idf'
    # 4. For each disease, get its seed genes’ IDF values, then mean/median
    spec_metrics = []
    for sample, genes in sample_to_seed.items():
        # number of seeds for this disease
        sd_size = len(genes)

        # look up IDF for each seed gene (missing genes get NaN → dropped)
        idf_vals = idf_series.reindex(list(genes)).dropna()

        if idf_vals.empty:
            mean_idf = np.nan
            median_idf = np.nan
            adjusted_mean_idf = np.nan
            adjusted_median_idf = np.nan
        else:
            mean_idf = idf_vals.mean()
            median_idf = idf_vals.median()
            # adjust by ln(1 + |S_d|)
            factor = np.log(1 + sd_size)
            adjusted_mean_idf = mean_idf * factor
            adjusted_median_idf = median_idf * factor

        spec_metrics.append((sample, mean_idf, median_idf, adjusted_mean_idf, adjusted_median_idf))
    spec_df = pd.DataFrame(spec_metrics, columns=['sample', 'seed_specificity_mean', 'seed_specificity_median',
                                                  'seed_specificity_adjusted_mean', 'seed_specificity_adjusted_median'])
    # 5. Merge back into your metrics_df
    metrics_df = metrics_df.merge(spec_df, on='sample', how='left')
    entropy_list = []
    for sample, genes in sample_to_seed.items():
        # get df(g) for each seed gene (missing → 0)
        df_vals = df_series.reindex(list(genes)).fillna(0)
        total = df_vals.sum()

        if total <= 0:
            H = np.nan
        else:
            p = df_vals / total  # p_g = df(g) / sum_{h in S_d} df(h)
            # avoid 0*log(0) by masking
            mask = p > 0
            H = - (p[mask] * np.log(p[mask])).sum()  # Shannon entropy

        entropy_list.append((sample, H))
    entropy_df = (pd.DataFrame(entropy_list, columns=['sample', 'seed_entropy']))
    # --- Merge into your metrics_df ---
    metrics_df = metrics_df.merge(entropy_df, on='sample', how='left')
    return metrics_df


def calculate_seed_jaccard_index(sample_to_seed):
    logging.info("Calculating Jaccard index metrics for seed genes")
    # 2. Precompute all pairwise Jaccard similarities between diseases
    #    We'll store them in a dict of dicts: sims[a][b] = J(a,b)
    sims = {s: {} for s in sample_to_seed}
    samples = list(sample_to_seed)
    for s1, s2 in combinations(samples, 2):
        a, b = sample_to_seed[s1], sample_to_seed[s2]
        union = a | b
        j = 1.0 if not union else len(a & b) / len(union)
        sims[s1][s2] = j
        sims[s2][s1] = j
    # 3. For each disease, gather its similarities vs all *other* diseases
    metrics = []
    for s in samples:
        values = list(sims[s].values())
        # if there's only one disease in total, you could choose to set NaNs:
        if not values:
            jmax = jmean = jmed = np.nan
        else:
            arr = np.array(values)
            jmax, jmean, jmed = arr.max(), arr.mean(), np.median(arr)
        metrics.append((s, jmax, jmean, jmed))
    metrics_df = pd.DataFrame(metrics, columns=['sample', 'jaccard_max', 'jaccard_mean', 'jaccard_median'])
    return metrics_df


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare combined TSV from MultiQC JSON and NedRexDB CSVs")
    parser.add_argument("--multiqc_path", default="../../data/multiqc_data", help="Path to MultiQC files")
    parser.add_argument("--multiqc_file", default="multiqc_data.json", help="MultiQC JSON filename")
    parser.add_argument("--nedrex_db_path", default="../../data/nedrexDB", help="Path to NedRexDB CSVs")
    parser.add_argument("--disorder", default="disorder.csv", help="Disorder CSV filename")
    parser.add_argument("--drug", default="drug.csv", help="Drug CSV filename")
    parser.add_argument("--drug_has_indication", default="drug_has_indication.csv", help="Drug-indication CSV filename")
    parser.add_argument("--drug_has_target", default="drug_has_target.csv", help="Drug-target CSV filename")
    parser.add_argument("--gene", default="gene.csv", help="Gene CSV filename")
    parser.add_argument("--gene_associated_with_disorder", default="gene_associated_with_disorder.csv",
                        help="Gene-disorder CSV filename")
    parser.add_argument("--output_path", default="../../data/input", help="Output TSV filename")
    parser.add_argument("--output", default="combined_data.tsv", help="Output combined TSV filename")
    parser.add_argument("--input_path", default="../../data/input", help="Path to input parent/child maps")
    parser.add_argument("--parents_map", default="parents_map.csv", help="Parents map CSV filename")
    parser.add_argument("--children_map", default="children_map.csv", help="Children map CSV filename")
    parser.add_argument("--significance_level", type=float, default=DEFAULT_SIGNIFICANCE_LEVEL,
                        help=f"Significance level for p-value analysis (default: {DEFAULT_SIGNIFICANCE_LEVEL}).")
    parser.add_argument("--correction", choices=["bh", "bonferroni"], default="bh",
                        help="Multiple testing correction method (bh or bonferroni)")
    parser.add_argument(
        "--compression",
        choices=["gzip", "none"],
        default="gzip",
        help="Output compression: 'gzip' to write a gzipped TSV (appends .gz if missing), 'none' for plain TSV.",
    )

    args = parser.parse_args()
    return args


def save_combined_df(df: pd.DataFrame, output_path: str, compression: str = "none") -> str:
    """
    Save DataFrame `df` to `output_path` as a TSV.
    If compression == "gzip", ensure filename ends with .gz and use pandas compression='gzip'.
    Returns the actual path written.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if compression == "gzip":
        # Ensure .gz extension
        if not output_path.endswith(".gz"):
            output_path = output_path + ".gz"
        logging.info("Saving gzipped TSV to %s", output_path)
        # pandas will handle gzip compression
        df.to_csv(output_path, sep="\t", index=False, compression="gzip")
    else:
        logging.info("Saving plain TSV to %s", output_path)
        df.to_csv(output_path, sep="\t", index=False)

    return output_path


if __name__ == '__main__':
    main()
