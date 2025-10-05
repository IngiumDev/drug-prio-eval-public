import logging
import os
import glob
import json
import unittest

from validation.drug_validation import drug_list_validation
from validation.extract_true_drugs import load_csv

logging.basicConfig(level=logging.INFO)

# Directory containing your JSON test cases (set via env var or default)
JSON_TEST_DIR = '../data/drug_validation_test_cases/NEDREX_API_wrapper'

# Path to your complete drug list CSV
DRUG_CSV_PATH = '../data/nedrexDB/drug.csv'


class DrugValidationTest(unittest.TestCase):
    def test_validation_json_files(self):
        """For each .json in JSON_TEST_DIR, run drug_list_validation and compare p-values."""
        pattern = os.path.join(JSON_TEST_DIR, '*.json')
        json_files = glob.glob(pattern)
        # fail early if no tests found
        self.assertTrue(json_files, f'No JSON files found in {JSON_TEST_DIR!r}')

        # Load the drugs_df once
        drugs_df = load_csv(DRUG_CSV_PATH)

        for fn in json_files:
            logging.info(f"Testing {fn}")
            with self.subTest(json_file=fn):
                with open(fn, 'r') as f:
                    data = json.load(f)

                candidates = data['test_drugs']
                true_drugs = data['true_drugs']
                permutations = data.get('permutations')
                only_approved = data.get('only_approved_drugs', False)
                expected_dcg_p = data['empirical DCG-based p-value']
                expected_ov_p = data['empirical p-value without considering ranks']

                # first try a single run
                result = drug_list_validation(
                    drugs_df,
                    true_drugs,
                    candidates,
                    permutation_count=permutations,
                    only_approved=only_approved
                )

                dcg_p = result['empirical DCG-based p-value']
                ov_p = result['empirical p-value without considering ranks']

                try:
                    # if it already matches to 6 decimals, great
                    self.assertAlmostEqual(dcg_p, expected_dcg_p, places=6,
                                           msg=f"DCG p-value mismatch in {fn}")
                    self.assertAlmostEqual(ov_p, expected_ov_p, places=6,
                                           msg=f"Overlap p-value mismatch in {fn}")
                except AssertionError:
                    # otherwise, average over 10 runs
                    dcg_accum = 0.0
                    ov_accum = 0.0
                    runs = 10
                    for _ in range(runs):
                        r = drug_list_validation(
                            drugs_df,
                            true_drugs,
                            candidates,
                            permutation_count=permutations,
                            only_approved=only_approved
                        )
                        dcg_accum += r['empirical DCG-based p-value']
                        ov_accum += r['empirical p-value without considering ranks']
                    avg_dcg = dcg_accum / runs
                    avg_ov = ov_accum / runs

                    # now assert on the averages
                    self.assertAlmostEqual(avg_dcg, expected_dcg_p, places=6,
                                           msg=(f"Avg DCG p-value {avg_dcg:.6f} vs "
                                                f"expected {expected_dcg_p:.6f} in {fn}"))
                    self.assertAlmostEqual(avg_ov, expected_ov_p, places=6,
                                           msg=(f"Avg overlap p-value {avg_ov:.6f} vs "
                                                f"expected {expected_ov_p:.6f} in {fn}"))


if __name__ == '__main__':
    unittest.main()