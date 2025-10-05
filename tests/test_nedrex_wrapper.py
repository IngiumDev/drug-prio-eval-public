import os
from dotenv import load_dotenv, find_dotenv
import glob
import json
import logging
import unittest
from src.exploratory.nedrex_validation_wrapper import run_drug_validation
from src.exploratory.generate_NEDREX_API_key import get_api_key

logging.basicConfig(level=logging.INFO)

class APIWrapperTestCase(unittest.TestCase):
    load_dotenv(find_dotenv())
    def test_all_cases(self):
        base_url = os.getenv("NEDREX_BASE_URL", "https://api.nedrex.net/licensed")
        api_key = get_api_key(base_url)
        test_files = glob.glob("../data/drug_validation_test_cases/NEDREX_API_wrapper/*.json")
        # Log The test files found
        logging.info(f"Test files found: {test_files}")
        for tf in test_files:
            with self.subTest(test_case=tf):
                logging.info(f"Testing {tf}")
                data = json.load(open(tf))
                result = run_drug_validation(
                    base_url,
                    api_key,
                    data["test_drugs"],
                    data["true_drugs"],
                    data["permutations"],
                    data["only_approved_drugs"]
                )
                # Compare p-values:
                self.assertAlmostEqual(
                    result["empirical DCG-based p-value"],
                    data["empirical DCG-based p-value"],
                    places=6,
                    msg=f"File {tf}: empirical DCG-based p-value mismatch (almost equal). Expected {data['empirical DCG-based p-value']}, got {result['empirical DCG-based p-value']}."
                )
                # Check if they are equal
                self.assertEqual(
                    result["empirical DCG-based p-value"],
                    data["empirical DCG-based p-value"],
                    msg=f"File {tf}: empirical DCG-based p-value mismatch (exact equality). Expected {data['empirical DCG-based p-value']}, got {result['empirical DCG-based p-value']}."
                )
                self.assertAlmostEqual(
                    result["empirical p-value without considering ranks"],
                    data["empirical p-value without considering ranks"],
                    places=6,
                    msg=f"File {tf}: empirical p-value without considering ranks mismatch (almost equal). Expected {data['empirical p-value without considering ranks']}, got {result['empirical p-value without considering ranks']}."
                )
                # Check if they are equal
                self.assertEqual(
                    result["empirical p-value without considering ranks"],
                    data["empirical p-value without considering ranks"],
                    msg=f"File {tf}: empirical p-value without considering ranks mismatch (exact equality). Expected {data['empirical p-value without considering ranks']}, got {result['empirical p-value without considering ranks']}."
                )
                # Log the found vs expected p-values
                logging.debug(f"Found p-values: {result['empirical DCG-based p-value']}, {result['empirical p-value without considering ranks']}")
                logging.debug(f"Expected p-values: {data['empirical DCG-based p-value']}, {data['empirical p-value without considering ranks']}")

