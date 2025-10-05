#!/usr/bin/env bash
set -euo pipefail

# Resolve path relative to this script’s location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
INPUT_DIR="$SCRIPT_DIR/../../data/nedrexDB"
INPUT_CSV="$INPUT_DIR/protein_interacts_with_protein.csv"
OUTPUT_TXT="$INPUT_DIR/protein_interacts_with_protein.unique_proteins.txt"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input not found: $INPUT_CSV" >&2
  exit 1
fi

echo "Reading:  $INPUT_CSV"
echo "Writing:  $OUTPUT_TXT"

# Stream:
# - strip CRs for Git Bash safety
# - skip header (NR>1)
# - print column 1 and 2 if non-empty
# - sort unique under C locale for speed and portability
tr -d '\r' < "$INPUT_CSV" \
| awk -F',' 'NR>1 { if ($1!="") print $1; if ($2!="") print $2 }' \
| LC_ALL=C sort -u > "$OUTPUT_TXT"

# Report count
COUNT="$(wc -l < "$OUTPUT_TXT" | tr -d ' ')"
echo "Done. Unique proteins: $COUNT"