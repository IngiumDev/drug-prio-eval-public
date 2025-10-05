import argparse
import csv
import json
import logging
import os
import sys
from typing import List, Dict

import requests
from dotenv import load_dotenv, find_dotenv

from validation.extract_true_drugs import load_drugs_list
from exploratory.nedrex_validation_wrapper import run_drug_validation

# load .env
load_dotenv(find_dotenv())

# reuse the module‐level logger

# base URL for all NeDRex licensed API calls
_BASE_URL = "https://api.nedrex.net/licensed"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)


def map_ids(collection: str, external_ids: List[str]) -> Dict[str, List[str]]:
    """
    Call the NeDRex ID‐Map endpoint to turn user IDs into primaryDomainIds.
    Returns a dict {external_id: [primary_id, …], …}.

    Args:
        collection:  The collection to map IDs from (e.g.  "disorder", "drug",etc).
        external_ids: A list of external IDs to be mapped.

    Returns:
        Dict[str, List[str]]: A dictionary mapping external IDs to primary domain IDs.
    """
    endpoint = f"{_BASE_URL}/id_map/{collection}"
    api_key = os.getenv("NEDREX_LICENSED_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = requests.get(endpoint, params={"q": external_ids}, headers=headers, timeout=5.0)
        resp.raise_for_status()
        return resp.json() or {}
    except requests.RequestException as e:
        logger.error("ID‐map request failed for %s: %s", collection, e)
        return {}


def fetch_indicated_drugs_for_disease(external_disease_id: str) -> List[str]:
    """
    1. Map the external disease ID (e.g. 'omim.130020') to primaryDomainId(s).
    2. Call 'get_drugs_indicated_for_disorders' with those primary IDs.
    3. If *every* primary ID returns an empty list, log an error and exit.
    4. Otherwise, return the unique set of all indicated DrugBank IDs.

    Args:
        external_disease_id: The external disease ID to fetch indicated drugs for.

    Returns:
        List[str]: A list of unique DrugBank IDs indicated for the disease.
    """
    logger.info("Fetching primary node IDs for disease %s …", external_disease_id)
    # Choose the right collection: e.g. "omim", "mondo", or "disorder"
    mapping: Dict[str, List[str]] = map_ids("disorder", [external_disease_id])
    primary_ids = mapping.get(external_disease_id, [])
    if not primary_ids:
        logger.error("No primaryDomainId found for %s; cannot proceed.", external_disease_id)
        sys.exit(1)

    logger.info("Calling relations/get_drugs_indicated_for_disorders with %s", primary_ids)
    rel_endpoint = f"{_BASE_URL}/relations/get_drugs_indicated_for_disorders"
    headers = {"Content-Type": "application/json",
               **({"x-api-key": os.getenv("NEDREX_LICENSED_KEY")} if os.getenv("NEDREX_LICENSED_KEY") else {})}

    try:
        resp = requests.post(rel_endpoint, json={"nodes": primary_ids}, headers=headers, timeout=5.0)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Relations request failed: %s", e)
        sys.exit(1)

    try:
        data = resp.json()
    except ValueError as e:
        logger.error("Failed to parse JSON from relations response: %s", e)
        sys.exit(1)

    # Expecting a dict { primary_id: [drugId, …], … }
    if not isinstance(data, dict):
        logger.error("Unexpected response format (expected dict): %r", data)
        sys.exit(1)

    # Check for empty lists across all primary IDs
    all_empty = all(not isinstance(drug_list, list) or len(drug_list) == 0 for drug_list in data.values())
    if all_empty:
        logger.error("No indicated drugs returned for disease %s (primary IDs: %s); cannot continue.",
                     external_disease_id, primary_ids)
        sys.exit(1)

    # Flatten into list
    indicated = list(
        {f"drugbank.{drug}" for drug_list in data.values() if isinstance(drug_list, list) for drug in drug_list})

    logger.debug("Indicated drugs for %s → %s", external_disease_id, indicated)
    return indicated


def load_indicated_drugs(indicated_drugs: str) -> List[str]:
    """
    Reads a file containing a list of indicated drugs in the first column with a header,
    and returns the drugs as a Python list of strings.

    Parameters:
        indicated_drugs (str): Path to the file containing indicated drugs.

    Returns:
        List[str]: A list of indicated drugs from the file.
    """
    drugs: List[str] = []
    with open(indicated_drugs, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header row
        for row in reader:
            if row:
                drugs.append(row[0])
    return drugs

def main():
    parser = argparse.ArgumentParser(description="Prototype tool to reproduce web-based drug-disease analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--candidate-drugs', required=True, help="Candidate drug results from drug prioritization tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--indicated-drugs', help="New line separated list of indicated drugs to validate, including header row"),
    group.add_argument('--disease', help="Disease name to fetch indicated drugs for")

    args = parser.parse_args()
    logger.debug(f"Parsed arguments: {args}")

    candidates = load_drugs_list(args.candidate_drugs)

    if args.indicated_drugs:
        indicated = load_indicated_drugs(args.indicated_drugs)
    else:
        indicated = fetch_indicated_drugs_for_disease(args.disease)
    result = run_drug_validation(_BASE_URL, os.getenv("NEDREX_LICENSED_KEY"), candidates, indicated, 1000, False)
    print("Validation completed:", json.dumps(result, indent=2))


if __name__ == '__main__':
    sys.argv = ['nedrex_validation_runner.py', '--candidate-drugs',
                '../../data/PipelineTestOut/drug_prioritization/drugstone/entrez_seeds_1.entrez_ppi.diamond.trustrank.csv',
                #'--disease', 'mondo.0004975'
                # '--disease', 'omim.130020'
                '--indicated-drugs', '../../data/drug_validation_test_cases/AlzheimersDrugs.csv'
                ]
    main()
