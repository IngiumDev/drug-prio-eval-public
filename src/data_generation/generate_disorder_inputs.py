#!/usr/bin/env python3
import argparse
import csv
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from upsetplot import UpSet, from_contents
from venn import venn
import matplotlib.cm as cm

SEED_GENES_BY_DISEASE = 'seed_genes_by_disease'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(handler)

_BASE_MD = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
# Keys are case-sensitive; use exactly these names. :contentReference[oaicite:2]{index=2}

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

# --- KDE plotting helpers ---
def _choose_minor_step(xmax: float) -> int:
    """Pick a minor tick step so we get ~10-interval spacing, defaulting to 10 when range is large."""
    if xmax <= 0:
        return 1
    if xmax >= 20:
        return 10
    # Aim for about 10 intervals across the range for small domains
    step = max(1, int(round(xmax / 10)))
    return step

def _style_kde_axes(ax, title: str, xlabel: str, xmax: float):
    ax.set_xlim(0, xmax)
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    # Minor gridlines on x at about every 10 units
    step = _choose_minor_step(xmax)
    ax.xaxis.set_minor_locator(MultipleLocator(step))
    ax.grid(axis='x', which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
    # Subtle major grid on x and y for readability
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.5)
    plt.tight_layout()

def _plot_kde(counts, title: str, xlabel: str, filename: str, plots_path: str, label: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if label is not None:
        counts.plot.kde(ax=ax, label=label)
        ax.legend()
    else:
        counts.plot.kde(ax=ax)
    xmax = max(1, float(counts.max()))
    _style_kde_axes(ax, title, xlabel, xmax)
    plt.savefig(os.path.join(plots_path, filename), bbox_inches='tight')
    plt.show()

def _plot_kde_multi(series_label_list, title: str, xlabel: str, filename: str, plots_path: str):
    """Plot several KDEs on one axis. series_label_list is a list of (counts_series, label)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    xmax = 0.0
    for counts, label in series_label_list:
        if counts is None or getattr(counts, 'empty', False):
            continue
        counts.plot.kde(ax=ax, label=label)
        try:
            xmax = max(xmax, float(counts.max()))
        except Exception:
            pass
    ax.legend()
    _style_kde_axes(ax, title, xlabel, max(1.0, xmax))
    plt.savefig(os.path.join(plots_path, filename), bbox_inches='tight')
    plt.show()
def build_drug_sets(drug_df, drug_has_indication_df, drug_has_target_df, plots_path, unique_proteins):
    """
    Build and return a dict of drug sets and create the Venn plot.
    """
    sets = {'All Drugs': set(drug_df['primaryDomainId']),
            'Approved Drugs': set(
                drug_df.loc[drug_df['drugGroups'].apply(lambda groups: 'approved' in groups), 'primaryDomainId']),
            'Drugs with Indications': set(drug_has_indication_df['sourceDomainId']),
           }
    # TODO durgs with protein targets in PPI
    # Quick set of drugs whose targets occur in the PPI protein set
    ppi_target_mask = drug_has_target_df['targetDomainId'].isin(unique_proteins)
    sets['Drugs with Targets'] = set(drug_has_target_df.loc[ppi_target_mask, 'sourceDomainId'])
    sets['Drugs with Targets and Indications'] = (
        sets['Drugs with Targets'].intersection(sets['Drugs with Indications']))

    sets['Approved Drugs with Targets'] = (sets['Drugs with Targets'].intersection(sets['Approved Drugs']))
    sets['Approved Drugs with Indications'] = (sets['Drugs with Indications'].intersection(sets['Approved Drugs']))
    sets['Approved Drugs with Targets and Indications'] = (
        sets['Approved Drugs'].intersection(sets['Drugs with Targets and Indications']))
    # Plot drug Venn
    plt.figure(figsize=(8, 8))
    venn({'All Drugs': sets['All Drugs'], 'Approved Drugs': sets['Approved Drugs'],
          'Drugs with Targets': sets['Drugs with Targets'], 'Drugs with Indications': sets['Drugs with Indications']})
    plt.title("Overlap of Drug Subsets")
    # Save Venn plot for drug subsets
    plt.savefig(os.path.join(plots_path, "venn_drug_subsets.pdf"), bbox_inches='tight')
    plt.show()

    # Plot UpSet for drug subsets
    upset_data = from_contents({
        'All Drugs': sets['All Drugs'],
        'Approved Drugs': sets['Approved Drugs'],
        'Drugs with Targets': sets['Drugs with Targets'],
        'Drugs with Indications': sets['Drugs with Indications']
    })
    # Scientific color palette: blue bars, gray shading
    face_color = "#0072B2"
    shading_color = "#E5E5E5"
    other_color = "#999999"
    upset = UpSet(
        upset_data,
        show_counts=True,
        sort_by="cardinality",
        facecolor=face_color,
        shading_color=shading_color,
        other_dots_color=other_color
    )
    upset.plot()
    plt.suptitle("UpSet Plot of Drug Subsets")
    # Save colored UpSet plot
    plt.savefig(os.path.join(plots_path, "upset_drug_subsets_colored.pdf"), bbox_inches='tight')
    plt.show()

    return sets


def build_disorder_sets(disorder_df, drug_has_indication_df, gene_associated_with_disorder_df, approved_drugs_set,
                        drugs_with_targets_set, approved_drugs_with_target_set, plots_path):
    """
    Build and return a dict of disorder sets and create Venn and UpSet plots.
    """
    sets = {'All Disorders': set(disorder_df['primaryDomainId']),
            'Disorders with Indications': set(drug_has_indication_df['targetDomainId']),
            'Disorders with Gene Seeds': set(gene_associated_with_disorder_df['targetDomainId']),
            'Disorders with Approved Drugs': set(drug_has_indication_df.loc[
                                                     drug_has_indication_df['sourceDomainId'].isin(
                                                         approved_drugs_set), 'targetDomainId']),
            'Disorders with Drugs with Targets': set(drug_has_indication_df.loc[
                                                         drug_has_indication_df['sourceDomainId'].isin(
                                                             drugs_with_targets_set), 'targetDomainId']),
            'Disorders with Approved Drugs with Targets': set(drug_has_indication_df.loc[
                drug_has_indication_df['sourceDomainId'].isin(approved_drugs_with_target_set), 'targetDomainId']), }
    sets['Disorders with Seeds and Approved Target Drugs'] = (
        sets['Disorders with Gene Seeds'].intersection(sets['Disorders with Approved Drugs with Targets']))
    # Print counts
    for name, s in sets.items():
        print(f"{name}: {len(s)}")
    # Plot disorder Venn
    plt.figure(figsize=(8, 8))
    venn({'All Disorders': sets['All Disorders'], 'Disorders with Indications': sets['Disorders with Indications'],
          'Disorders with Gene Seeds': sets['Disorders with Gene Seeds'],
          'Disorders with Approved Drugs': sets['Disorders with Approved Drugs'],
          'Disorders with Drugs with Targets': sets['Disorders with Drugs with Targets'],
         'Disorders with Approved Drugs with Targets': sets['Disorders with Approved Drugs with Targets']})
    plt.title("Overlap of Disorder Subsets")
    # Save Venn plot for disorder subsets
    plt.savefig(os.path.join(plots_path, "venn_disorder_subsets.pdf"), bbox_inches='tight')
    plt.show()
    # Plot UpSet
    upset_data = from_contents(
        {'All Disorders': sets['All Disorders'], 'Disorders with Indications': sets['Disorders with Indications'],
         'Disorders with Gene Seeds': sets['Disorders with Gene Seeds'],
         'Disorders with Approved Drugs': sets['Disorders with Approved Drugs'],
         'Disorders with Drugs with Targets': sets['Disorders with Drugs with Targets'],
         'Disorders with Approved Drugs with Targets': sets['Disorders with Approved Drugs with Targets']})
    # Scientific color palette: blue bars, gray shading
    face_color = "#0072B2"
    shading_color = "#E5E5E5"
    other_color = "#999999"
    upset = UpSet(
        upset_data,
        show_counts=True,
        sort_by="cardinality",
        facecolor=face_color,
        shading_color=shading_color,
        other_dots_color=other_color
    )
    upset.plot()
    plt.suptitle("UpSet Plot of Disorder Subsets")
    # Save colored UpSet plot
    plt.savefig(os.path.join(plots_path, "upset_disorder_subsets_colored.pdf"), bbox_inches='tight')
    plt.show()
    return sets


def plot_indication_distribution(drug_has_indication_df, plots_path):
    """
    Plot a styled KDE of the number of disorders each drug is indicated for, with minor x-gridlines.
    """
    counts = drug_has_indication_df.groupby('sourceDomainId')['targetDomainId'].nunique()
    _plot_kde(
        counts,
        title='KDE of Number of Disorders per Drug',
        xlabel='Number of Disorders per Drug',
        filename='kde_disorders_per_drug.pdf',
        plots_path=plots_path,
    )
    return counts


def plot_targets_per_drug(drug_has_target_df, plots_path, unique_proteins=None):
    """
    Plot KDE distributions for the number of targets per drug on a linear x-scale, with minor x-gridlines.
    Produces one plot for all targets and, if unique_proteins is provided, a second plot restricted to targets
    present in the provided protein set (e.g., PPI proteins).
    """
    # All targets per drug
    counts_all = drug_has_target_df.groupby('sourceDomainId')['targetDomainId'].nunique()
    counts_all = counts_all[counts_all > 0]
    _plot_kde(
        counts_all,
        title='KDE of Targets per Drug',
        xlabel='Number of Targets per Drug',
        filename='kde_targets_per_drug.pdf',
        plots_path=plots_path,
    )

    # Targets per drug restricted to provided protein set (e.g., PPI)
    if unique_proteins is not None:
        mask = drug_has_target_df['targetDomainId'].isin(unique_proteins)
        counts_ppi = drug_has_target_df[mask].groupby('sourceDomainId')['targetDomainId'].nunique()
        counts_ppi = counts_ppi[counts_ppi > 0]
        if not counts_ppi.empty:
            _plot_kde(
                counts_ppi,
                title='KDE of Targets per Drug (restricted to PPI proteins)',
                xlabel='Number of Targets per Drug (PPI)',
                filename='kde_targets_in_ppi_per_drug.pdf',
                plots_path=plots_path,
            )


def plot_drugs_per_target(drug_has_target_df, plots_path):
    """
    Plot a KDE of the number of drugs per protein target (reverse view) with minor x-gridlines.
    """
    counts = drug_has_target_df.groupby('targetDomainId')['sourceDomainId'].nunique()
    counts = counts[counts > 0]
    _plot_kde(
        counts,
        title='KDE of Drugs per Protein Target',
        xlabel='Number of Drugs per Target',
        filename='kde_drugs_per_target.pdf',
        plots_path=plots_path,
    )
    return counts


def plot_disease_indication_distribution(drug_has_indication_df, drug_sets, plots_path):
    """
    Plot KDE distributions of the number of drug indications per disease for different drug filters,
    with minor x-gridlines.
    """
    # Base count: all indicated drugs per disease
    counts_all = drug_has_indication_df.groupby('targetDomainId')['sourceDomainId'].nunique()
    # Approved drugs only
    approved_set = set(drug_sets['Approved Drugs'])
    counts_approved = (
        drug_has_indication_df[drug_has_indication_df['sourceDomainId'].isin(approved_set)]
        .groupby('targetDomainId')['sourceDomainId']
        .nunique()
    )
    # Drugs with targets only
    targets_set = set(drug_sets['Drugs with Targets'])
    counts_targets = (
        drug_has_indication_df[drug_has_indication_df['sourceDomainId'].isin(targets_set)]
        .groupby('targetDomainId')['sourceDomainId']
        .nunique()
    )
    # Drugs both approved and with targets
    approved_targets_set = approved_set.intersection(targets_set)
    counts_approved_targets = (
        drug_has_indication_df[drug_has_indication_df['sourceDomainId'].isin(approved_targets_set)]
        .groupby('targetDomainId')['sourceDomainId']
        .nunique()
    )

    # Exclude diseases with no indicated drugs
    counts_all = counts_all[counts_all > 0]
    counts_approved = counts_approved.reindex(counts_all.index, fill_value=0)
    counts_targets = counts_targets.reindex(counts_all.index, fill_value=0)
    counts_approved_targets = counts_approved_targets.reindex(counts_all.index, fill_value=0)

    _plot_kde_multi(
        [
            (counts_all, 'All Indications'),
            (counts_approved, 'Approved Indications'),
            (counts_targets, 'Indications with Targets'),
            (counts_approved_targets, 'Approved & Targets'),
        ],
        title='KDE of Drug Indications per Disease',
        xlabel='Number of Drug Indications',
        filename='kde_indications_per_disease.pdf',
        plots_path=plots_path,
    )


def plot_seed_genes_per_disease(gene_associated_with_disorder_df, plots_path):
    """
    Plot a styled KDE of the number of seed genes per disease, with minor x-gridlines.
    """
    counts = gene_associated_with_disorder_df.groupby('targetDomainId')['sourceDomainId'].nunique()
    counts = counts[counts > 0]
    _plot_kde(
        counts,
        title='KDE of Seed Genes per Disease',
        xlabel='Number of Seed Genes per Disease',
        filename='kde_seed_genes_per_disease.pdf',
        plots_path=plots_path,
    )
    logger.info(f"Number of disorders with more than 25 seed genes: {len(counts[counts > 25])}")


def plot_diseases_per_seed_gene(gene_associated_with_disorder_df, plots_path):
    """
    Plot a styled KDE of the number of diseases per seed gene, with minor x-gridlines.
    """
    counts = gene_associated_with_disorder_df.groupby('sourceDomainId')['targetDomainId'].nunique()
    counts = counts[counts > 0]
    _plot_kde(
        counts,
        title='KDE of Diseases per Seed Gene',
        xlabel='Number of Diseases per Seed Gene',
        filename='kde_diseases_per_seed_gene.pdf',
        plots_path=plots_path,
    )
    logger.info(f"Number of seed genes associated with more than 25 diseases: {len(counts[counts > 25])}")


def main():
    args = parse_arguments()

    try:
        os.makedirs(args.output_path, exist_ok=True)
        logger.info(f"Output directory '{args.output_path}' is ready.")
    except Exception as e:
        logger.error(f"Could not create output directory '{args.output_path}': {e}")
        return

    disorder_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.disorder_file))
    drug_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.drug_file))
    drug_has_indication_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.drug_indication_file))
    # Plot distribution of indications per drug
    plot_indication_distribution(drug_has_indication_df, args.plots_path)
    drug_has_target_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.drug_target_file))
    unique_proteins = set()
    logger.info(f"Loading precomputed protein list from '{args.protein_list_file}'")
    try:
        with open(os.path.join(args.nedrex_db_path,args.protein_list_file)) as f:
            for line in f:
                s = line.strip()
                if s:
                    unique_proteins.add(s)
        logger.info("Loaded %d unique proteins from precomputed list", len(unique_proteins))
    except Exception as e:
        logger.error(f"Failed to load precomputed protein list '{args.protein_list_file}': {e}")
        return
    # Plot distribution of targets per drug (all targets)
    plot_targets_per_drug(drug_has_target_df, args.plots_path)
    # Plot distribution of drugs per protein target (reverse)
    plot_drugs_per_target(drug_has_target_df, args.plots_path)
    gene_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.gene_file))
    gene_associated_with_disorder_df = pd.read_csv(os.path.join(args.nedrex_db_path, args.gene_association_file))
    # Plot seed gene distributions
    plot_seed_genes_per_disease(gene_associated_with_disorder_df, args.plots_path)
    plot_diseases_per_seed_gene(gene_associated_with_disorder_df, args.plots_path)

    # Build drug sets and Venn plot
    drug_sets = build_drug_sets(drug_df, drug_has_indication_df, drug_has_target_df, args.plots_path,unique_proteins)

    # Build disorder sets, Venn and UpSet plots
    disorder_sets = build_disorder_sets(disorder_df, drug_has_indication_df, gene_associated_with_disorder_df,
                                        drug_sets['Approved Drugs'], drug_sets['Drugs with Targets'],
                                        drug_sets['Approved Drugs with Targets'], args.plots_path)
    # Plot distribution of drug indications per disease
    plot_disease_indication_distribution(drug_has_indication_df, drug_sets, args.plots_path)

    # Save the dataframes to CSV files
    if not args.skip_disorder_sets:
        # Save disorder subsets to CSV files in the output folder
        output_file = 'all_disorders.csv'
        output_path = args.output_path
        save_set(disorder_sets['All Disorders'], output_file, output_path)
        save_set(disorder_sets['Disorders with Indications'], 'disorders_with_indications.csv', args.output_path)
        save_set(disorder_sets['Disorders with Gene Seeds'], 'disorders_with_gene_seeds.csv', args.output_path)
        save_set(disorder_sets['Disorders with Approved Drugs'], 'disorders_with_approved_drugs.csv', args.output_path)
        save_set(disorder_sets['Disorders with Drugs with Targets'], 'disorders_with_drugs_with_targets.csv',
                 args.output_path)
        save_set(disorder_sets['Disorders with Seeds and Approved Target Drugs'],
                 'disorders_with_seeds_and_approved_target_drugs.csv', args.output_path)

    # Create the seed genes by disease input files
    if not args.skip_seed_genes:
        create_seeds_by_disease_input(disorder_sets['Disorders with Gene Seeds'], gene_associated_with_disorder_df,
                                      args.output_path)
    # Create true drug lists by disease for various filters
    if not args.skip_true_drugs_by_disease:
        create_true_drugs_by_disease_input(disorder_sets['Disorders with Indications'], drug_has_indication_df,
                                           drug_sets['Drugs with Indications'], args.output_path,
                                           'true_drugs_by_disease')
        create_true_drugs_by_disease_input(disorder_sets['Disorders with Approved Drugs'], drug_has_indication_df,
                                           drug_sets['Approved Drugs'], args.output_path,
                                           'true_approved_drugs_by_disease')
        create_true_drugs_by_disease_input(disorder_sets['Disorders with Seeds and Approved Target Drugs'],
                                           drug_has_indication_df, drug_sets['Approved Drugs with Targets'],
                                           args.output_path, 'true_approved_drugs_with_targets_by_disease')


def create_seeds_by_disease_input(disorders_with_gene_seeds_set, gene_associated_with_disorder_df, output_path):
    seed_dir = os.path.join(output_path, SEED_GENES_BY_DISEASE)
    try:
        os.makedirs(seed_dir, exist_ok=True)
        logger.info(f"Seed directory '{seed_dir}' is ready.")
    except Exception as e:
        logger.error(f"Could not create seed directory '{seed_dir}': {e}")
    for disorder in tqdm(sorted(disorders_with_gene_seeds_set), desc="Saving seed genes"):
        # extract seed gene IDs without prefix (everything before the first dot and the dot itself)
        seeds = gene_associated_with_disorder_df.loc[
            gene_associated_with_disorder_df['targetDomainId'] == disorder, 'sourceDomainId'].str.split('.').str[1]
        file_path = os.path.join(seed_dir, f"{disorder}.csv")
        seeds.to_csv(file_path, header=False, index=False)


def create_true_drugs_by_disease_input(disorder_set, drug_has_indication_df, filter_set, output_path, dir_name):
    """
    Create per-disease CSV files of true drugs.
    Only writes files for diseases where at least one drug in filter_set has an indication.
    """
    dir_path = os.path.join(output_path, dir_name)
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory '{dir_path}' is ready.")
    except Exception as e:
        logger.error(f"Could not create directory '{dir_path}': {e}")
        return
    for disorder in tqdm(sorted(disorder_set), desc=f"Saving true drugs in {dir_name}"):
        # collect drugs with indication for this disorder
        drugs = drug_has_indication_df.loc[drug_has_indication_df['targetDomainId'] == disorder, 'sourceDomainId']
        # filter to the requested set
        true_drugs = [d for d in drugs if d in filter_set]
        if not true_drugs:
            logger.debug(f"No true drugs found for disorder {disorder} in {dir_name}, skipping.")
            continue
        file_path = os.path.join(dir_path, f"{disorder}.csv")
        pd.Series(true_drugs).to_csv(file_path, header=['trueDrugs'], index=False)


def save_set(all_disorders_set, output_file, output_path):
    pd.Series(sorted(all_disorders_set)).to_csv(os.path.join(output_path, output_file), header=False, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate disorder input files for drug prioritization")
    parser.add_argument("--plots_path", default="../../plots/nedrexDB/", help="Base path for plots output")
    parser.add_argument("--nedrex_db_path", default="../../data/nedrexDB",
                        help="Base path for the NEDREX database files")
    parser.add_argument("--output_path", default="../../data/input/",
                        help="Directory where input files will be written")
    parser.add_argument("--disorder_file", default="disorder.csv", help="Filename for the disorder CSV")
    parser.add_argument("--drug_file", default="drug.csv", help="Filename for the drug CSV")
    parser.add_argument("--drug_indication_file", default="drug_has_indication.csv",
                        help="Filename for the drug_has_indication CSV")
    parser.add_argument("--drug_target_file", default="drug_has_target.csv",
                        help="Filename for the drug_has_target CSV")
    parser.add_argument("--gene_file", default="gene.csv", help="Filename for the gene CSV")
    parser.add_argument("--gene_association_file", default="gene_associated_with_disorder.csv",
                        help="Filename for the gene_associated_with_disorder CSV")
    parser.add_argument("--skip_seed_genes", action='store_true',
                        help="Skip creation of seed genes by disease input files")
    parser.add_argument("--skip_disorder_sets", action='store_true', help="Skip saving disorder sets to CSV files")
    parser.add_argument("--skip_true_drugs_by_disease", action='store_true',
        help="Skip creating true drugs by disease input files")
    parser.add_argument(
        "--protein_list_file",
        default="protein_interacts_with_protein.unique_proteins.txt",
        help="Optional path to a precomputed newline-delimited list of unique protein IDs "
             "(from precompute_unique_proteins.py). If provided, the PPI CSV will not be scanned."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
