#!/usr/bin/env python3
import argparse
import csv
import json
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build parent→children and child→parents maps from a Mondo subtype CSV"
    )
    parser.add_argument(
        "--subtype-file",
        default="../../data/nedrexDB/disorder_is_subtype_of_disorder.csv",
        help="Input CSV: sourceDomainId,targetDomainId,…"
    )
    parser.add_argument(
        "--output-children-csv",
        default="children_map.csv",
        help="Output CSV with columns: disorder,children (JSON list)"
    )
    parser.add_argument(
        "--output-parents-csv",
        default="parents_map.csv",
        help="Output CSV with columns: disorder,parents (JSON list)"
    )
    return parser.parse_args()

def load_subtype(subtype_file):
    children_map = {}
    try:
        with open(subtype_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                child = row["sourceDomainId"]
                parent = row["targetDomainId"]
                children_map.setdefault(parent, []).append(child)
    except FileNotFoundError:
        sys.exit(f"ERROR: subtype file not found: {subtype_file}")
    return children_map

def invert_map(children_map):
    parents_map = {}
    for parent, children in children_map.items():
        for child in children:
            parents_map.setdefault(child, []).append(parent)
    return parents_map

def dump_map(map_data, output_path, header_name):
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["disorder", header_name])
        for disorder, rels in sorted(map_data.items()):
            writer.writerow([disorder, json.dumps(rels)])

def main():
    args = parse_arguments()
    children_map = load_subtype(args.subtype_file)
    parents_map = invert_map(children_map)

    dump_map(children_map, args.output_children_csv, "children")
    dump_map(parents_map, args.output_parents_csv, "parents")

    print(
        f"Wrote {len(children_map)} parent→children entries to {args.output_children_csv}\n"
        f"Wrote {len(parents_map)} child→parents entries to {args.output_parents_csv}"
    )

if __name__ == "__main__":
    main()