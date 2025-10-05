import argparse
import logging
import os
import sys

from validation.extract_true_drugs import load_csv

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(handler)


def resolve_primary_id_from_alt(disorder_df, alt_id):
    # domainIds is assumed to be a comma-separated list of alternative IDs
    matches = disorder_df[
        disorder_df['domainIds'].fillna("")
        .apply(lambda cells: alt_id in [i.strip() for i in cells.split(',')])
    ]
    if len(matches) == 0:
        logger.error(f"No disorder found containing alternative ID '{alt_id}'")
        sys.exit(1)
    if len(matches) > 1:
        logger.error(f"Multiple disorders found containing alternative ID '{alt_id}':\n"
                     + matches['primaryDomainId'].tolist().__str__())
        sys.exit(1)
    return matches.iloc[0]['primaryDomainId']


def resolve_primary_id_from_name(disorder_df, name):
    name_lower = name.lower()
    # match displayName exactly
    mask_name = disorder_df['displayName'].str.lower() == name_lower
    # synonyms assumed comma-separated list
    mask_syn = disorder_df['synonyms'].fillna("") \
        .apply(lambda cells: name_lower in [s.strip().lower() for s in cells.split(',')])
    matches = disorder_df[mask_name | mask_syn]
    if len(matches) == 0:
        logger.error(f"No disorder found matching name '{name}'")
        sys.exit(1)
    if len(matches) > 1:
        logger.error(f"Multiple disorders match name '{name}':\n"
                     + matches[['primaryDomainId','displayName']].to_string(index=False))
        sys.exit(1)
    return matches.iloc[0]['primaryDomainId']


def main():
    parser = argparse.ArgumentParser(
        description="Extract associated genes for a given disease"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p", "--primary-id",
        help="Primary disease ID (e.g., 'mondo.0001234')"
    )
    group.add_argument(
        "-a", "--alt-id",
        help="Alternative disease ID (e.g., 'umls.C0007102')"
    )
    group.add_argument(
        "-n", "--name",
        help="Disease name to look up in displayName or synonyms"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Export associated genes for all disorders in the disorder file"
    )
    parser.add_argument(
        "-c", "--disorder-file",
        help="CSV file with disorder collection (required for alt-id or name)"
    )
    parser.add_argument(
        "-g", "--gene-file",
        required=True,
        help="Path to the gene associated with disorder CSV file"
    )
    parser.add_argument(
        "-o", "--output-folder",
        required=True,
        help="Folder to save the output CSV"
    )
    parser.add_argument(
        "-f", "--output-file",
        default="associated_genes.csv",
        help="Name of the output CSV file"
    )
    args = parser.parse_args()

    # Ensure output folder exists
    try:
        os.makedirs(args.output_folder, exist_ok=True)
        logger.info(f"Output folder '{args.output_folder}' is ready.")
    except OSError as e:
        logger.error(f"Failed to create output folder '{args.output_folder}': {e}")
        sys.exit(1)

    # Load gene associations early for grouping by disorder when --all
    associated_genes_df = load_csv(args.gene_file)

    # Determine mode: single disorder or all disorders
    if args.all:
        # Derive all disorders directly from association data
        primary_ids = associated_genes_df['targetDomainId'].dropna().unique().tolist()
        logger.info(f"Exporting associated genes for all {len(primary_ids)} disorders found in gene associations.")
    else:
        if args.primary_id:
            primary_id = args.primary_id
        else:
            if not args.disorder_file:
                logger.error("--disorder-file is required when using --alt-id or --name")
                sys.exit(1)
            disorder_df = load_csv(args.disorder_file)
            if args.alt_id:
                primary_id = resolve_primary_id_from_alt(disorder_df, args.alt_id)
            else:
                primary_id = resolve_primary_id_from_name(disorder_df, args.name)
        logger.info(f"Using primaryDomainId: {primary_id}")
        primary_ids = [primary_id]

    # Process output for single or all mode
    if args.all:
        for pid in primary_ids:
            genes = associated_genes_df.loc[
                associated_genes_df['targetDomainId'] == pid,
                'sourceDomainId'
            ].tolist()
            genes = [g.split('.')[1] if '.' in g else g for g in genes]
            output_path = os.path.join(args.output_folder, f"{pid}.csv")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for gene in genes:
                        f.write(f"{gene}\n")
                logger.info(f"Associated genes for '{pid}' saved to '{output_path}'")
            except IOError as e:
                logger.error(f"Failed to write to '{output_path}': {e}")
                sys.exit(1)
    else:
        output_path = os.path.join(args.output_folder, args.output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for pid in primary_ids:
                    genes = associated_genes_df.loc[
                        associated_genes_df['targetDomainId'] == pid,
                        'sourceDomainId'
                    ].tolist()
                    genes = [g.split('.')[1] if '.' in g else g for g in genes]
                    for gene in genes:
                        f.write(f"{gene}\n")
            logger.info(f"Associated genes saved to '{output_path}'")
        except IOError as e:
            logger.error(f"Failed to write to '{output_path}': {e}")
            sys.exit(1)


if __name__ == '__main__':
    #sys.argv = ['extract_associated_genes.py', '--gene-file', '../../data/nedrexDB/gene_associated_with_disorder.csv', '--all','--output-folder', './data']
    main()
