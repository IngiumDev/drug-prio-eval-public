#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import random
import sys

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(handler)


def parse_size_arg(size_str, total):
    if size_str is None:
        return total
    if size_str.endswith('%'):
        pct = float(size_str.rstrip('%')) / 100.0
        return max(1, int(total * pct))
    count = int(size_str)
    return min(count, total)


def parse_arguments():
    current_user = os.environ.get('USER', '')
    parser = argparse.ArgumentParser(description="Sample diseases and generate Nextflow params YAML")
    parser.add_argument("--disorders-file", default="disorders_with_seeds_and_approved_target_drugs.csv",
                        help="Path to file with one disease ID per line")
    parser.add_argument("--seed-dir",
                        default=f"/nfs/data3/{current_user}/drug-prio-eval/data/input/seed_genes_by_disease",
                        help="Directory containing per-disease seed CSVs")
    parser.add_argument("--drug-dir",
                        default=f"/nfs/data3/{current_user}/drug-prio-eval/data/input/true_approved_drugs_with_targets_by_disease",
                        help="Directory containing per-disease true_drugs CSVs")
    parser.add_argument("--size", default=None,
                        help="Number or percentage of diseases to sample (e.g. '50' or '25%'); default is all")
    parser.add_argument("--output", default="params.yaml", help="Output params filename (YAML format)")
    parser.add_argument("--children-map-csv", default=None,
                        help="Path to precomputed parent→children map CSV (columns: disorder,children as JSON list)")
    parser.add_argument("--exclude-parent-disorders", nargs="*", default=[],
                        help="List of parent disorder IDs to exclude (plus their descendants)")
    return parser.parse_args()


def load_disorder_ids(path):
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        sys.exit(f"ERROR: disorders file not found: {path}")


def load_children_map(csv_path):
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            cmap = {}
            for row in reader:
                children = json.loads(row["children"])
                cmap[row["disorder"]] = set(children)
            return cmap
    except Exception as e:
        sys.exit(f"ERROR loading children map from {csv_path}: {e}")


def collect_descendants(children_map, pid):
    seen = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in children_map.get(current, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def apply_exclusions(all_ids, children_map, exclude_parents):
    if not exclude_parents:
        return all_ids
    if not children_map:
        sys.exit("ERROR: --children-map-csv is required when using --exclude-parent-disorders")
    exclude_set = set()
    for parent in exclude_parents:
        desc = collect_descendants(children_map, parent)
        if not desc:
            logger.warning(f"No descendants found for parent disorder {parent}")
        exclude_set.add(parent)
        exclude_set.update(desc)
    original = len(all_ids)
    filtered = [d for d in all_ids if d not in exclude_set]
    removed = original - len(filtered)
    logger.info(f"Identified {len(exclude_set)} to exclude; actually removed {removed} disorders")
    return filtered


def sample_ids(all_ids, size):
    n = parse_size_arg(size, len(all_ids))
    return random.sample(all_ids, n)


def build_paths(sampled, seed_dir, drug_dir):
    seeds = ",".join(os.path.join(seed_dir, f"{d}.csv") for d in sampled)
    drugs = ",".join(os.path.join(drug_dir, f"{d}.csv") for d in sampled)
    return seeds, drugs


def write_yaml(params, output):
    with open(output, "w") as f:
        yaml.dump(params, f, default_flow_style=False)


def main():
    args = parse_arguments()

    # if exclusion requested, preload the parent→children map
    children_map = {}
    if args.exclude_parent_disorders:
        children_map = load_children_map(args.children_map_csv)

    all_ids = load_disorder_ids(args.disorders_file)
    filtered = apply_exclusions(all_ids, children_map, args.exclude_parent_disorders)
    sampled = sample_ids(filtered, args.size)
    seeds, drugs = build_paths(sampled, args.seed_dir, args.drug_dir)

    write_yaml({"seeds": seeds, "true_drugs": drugs}, args.output)
    print(f"Wrote {len(sampled)} entries to {args.output}")


if __name__ == "__main__":
    main()
