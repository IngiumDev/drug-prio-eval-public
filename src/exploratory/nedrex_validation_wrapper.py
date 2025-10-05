import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv, find_dotenv

# Constants
_BASE_URL = "https://api.nedrex.net/licensed"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)


def poll_job_status(uid: str, status_endpoint: str, api_key: str, timeout: float = 5.0, interval: float = 5.0,
                    max_attempts: int = 60) -> Dict[str, Any]:
    """
    Polls status_endpoint until status == 'completed', an unauthorized error occurs,
    or max_attempts is reached. Returns the final JSON dict (or {'error':401} on 401,
    or {} on other failures/timeouts).
    """
    headers = {"x-api-key": api_key}
    for attempt in range(1, max_attempts + 1):
        logger.debug("Polling attempt %d/%d for uid %s", attempt, max_attempts, uid)
        try:
            resp = requests.get(status_endpoint, params={"uid": uid}, headers=headers, timeout=timeout)
            if resp.status_code == 401:
                logger.error("Unauthorized request for uid %s", uid)
                return {"error": 401}
            resp.raise_for_status()
            status_data = resp.json()
            logger.debug("Received status data: %s", status_data)
            if status_data.get("status") == "completed":
                return status_data
            elif status_data.get("status") == "failed":
                logger.error("Job failed for uid %s: %s", uid, status_data.get("error", "Unknown error"))
                return {}
        except requests.RequestException as e:
            logger.error("Status request failed: %s", e)
            return {}
        except ValueError as e:
            logger.error("Failed to parse JSON: %s", e)
            return {}
        time.sleep(interval)

    logger.error("Polling timed out after %d attempts for uid %s", max_attempts, uid)
    return {}


def run_drug_validation(base_url: str, api_key: str, test_drugs: list[Any], true_drugs: list[str], permutations: int,
                        only_approved_drugs: bool, timeout: Optional[float] = 5.0) -> Dict[str, Any]:
    """
    Runs validation by submitting a job to the NeDRex API and polling for its status.
    Args:
        base_url: Base URL for the NeDRex API.
        api_key: API key for authentication.
        test_drugs: List of test drugs, each represented as a list containing a drug ID and its rank.
        true_drugs: List of true drug IDs.
        permutations:  Number of permutations to use in the validation.
        only_approved_drugs:   Whether to consider only approved drugs.
        timeout: Timeout for the HTTP requests.

    Returns:  JSON response from the validation job.

    """
    validate_endpoint = f"{base_url}/validation/drug"
    status_endpoint = f"{base_url}/validation/status"
    payload = {"test_drugs": test_drugs, "true_drugs": true_drugs, "permutations": permutations,
               "only_approved_drugs": only_approved_drugs, }
    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    if not test_drugs:
        logger.error("test_drugs list is empty; skipping validation and returning p-values of 1")
        return {**payload, "empirical DCG-based p-value": 1, "empirical p-value without considering ranks": 1, }

    try:
        logger.debug("Submitting validation job to %s", validate_endpoint)
        resp = requests.post(validate_endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        uid = resp.json()  # assuming the API returns just the UID
    except requests.RequestException as e:
        logger.error("HTTP request failed: %s", e)
        return {}
    except ValueError as e:
        logger.error("Failed to parse JSON response: %s", e)
        return {}

    return poll_job_status(uid, status_endpoint, api_key, timeout=timeout)


# --- New functions for joint and module validation ---
def run_joint_validation(base_url: str, api_key: str, module_members: List[str], module_member_type: str,
                         test_drugs: List[str], true_drugs: List[str], permutations: int, only_approved_drugs: bool,
                         timeout: Optional[float] = 5.0) -> Dict[str, Any]:
    """
    Runs joint validation by submitting a job to the NeDRex API and polling for its status.

    Args:
        base_url: Base URL for the NeDRex API.
        api_key: API key for authentication.
        module_members: List of module members (proteins/genes) in the disease module.
        module_member_type: Type of module members ('gene' or 'protein').
        test_drugs: List of test drug IDs to validate.
        true_drugs: List of true drug IDs indicated to treat the disease.
        permutations: Number of permutations to use in the validation.
        only_approved_drugs: Whether to consider only approved drugs.
        timeout: Timeout for the HTTP requests.

    Returns:
        JSON response from the joint validation job.
    """
    validate_endpoint = f"{base_url}/validation/joint"
    status_endpoint = f"{base_url}/validation/status"
    payload = {"module_members": module_members, "module_member_type": module_member_type, "test_drugs": test_drugs,
               "true_drugs": true_drugs, "permutations": permutations, "only_approved_drugs": only_approved_drugs, }
    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    if not module_members or not test_drugs:
        logger.error("module_members or test_drugs list is empty; skipping joint validation"
                     " and returning p-values of 1")
        return {**payload, "empirical DCG-based p-value": 1, "empirical p-value without considering ranks": 1}

    try:
        logger.debug("Submitting joint validation job to %s", validate_endpoint)
        resp = requests.post(validate_endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        uid = resp.json()
    except requests.RequestException as e:
        logger.error("HTTP request failed: %s", e)
        return {}
    except ValueError as e:
        logger.error("Failed to parse JSON response: %s", e)
        return {}

    return poll_job_status(uid, status_endpoint, api_key, timeout=timeout)


def run_module_validation(base_url: str, api_key: str, module_members: List[str], module_member_type: str,
                          true_drugs: List[str], permutations: int, only_approved_drugs: bool,
                          timeout: Optional[float] = 5.0) -> Dict[str, Any]:
    """
    Runs module validation by submitting a job to the NeDRex API and polling for its status.

    Args:
        base_url: Base URL for the NeDRex API.
        api_key: API key for authentication.
        module_members: List of module members (proteins/genes) in the disease module.
        module_member_type: Type of module members ('gene' or 'protein').
        true_drugs: List of true drug IDs indicated to treat the disease.
        permutations: Number of permutations to use in the validation.
        only_approved_drugs: Whether to consider only approved drugs.
        timeout: Timeout for the HTTP requests.

    Returns:
        JSON response from the module validation job.
    """
    validate_endpoint = f"{base_url}/validation/module"
    status_endpoint = f"{base_url}/validation/status"
    payload = {"module_members": module_members, "module_member_type": module_member_type, "true_drugs": true_drugs,
               "permutations": permutations, "only_approved_drugs": only_approved_drugs, }
    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    if not module_members:
        logger.error("module_members list is empty; skipping module validation"
                     " and returning p-values of 1")
        return {**payload, "empirical DCG-based p-value": 1, "empirical p-value without considering ranks": 1}

    try:
        logger.debug("Submitting module validation job to %s", validate_endpoint)
        resp = requests.post(validate_endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        uid = resp.json()
    except requests.RequestException as e:
        logger.error("HTTP request failed: %s", e)
        return {}
    except ValueError as e:
        logger.error("Failed to parse JSON response: %s", e)
        return {}

    return poll_job_status(uid, status_endpoint, api_key, timeout=timeout)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    base_url = "https://api.nedrex.net/licensed"
    api_key = os.getenv("NEDREX_LICENSED_KEY")

    # Test Drug Validation
    test_drugs = [["drugbank.DB12010", 1], ["drugbank.DB04216", 2], ["drugbank.DB07159", 3], ["drugbank.DB12500", 4],
                  ["drugbank.DB01268", 5], ["drugbank.DB00398", 6], ["drugbank.DB06595", 6], ["drugbank.DB09079", 7],
                  ["drugbank.DB01254", 8], ["drugbank.DB06616", 9], ["drugbank.DB00530", 10], ["drugbank.DB05294", 11],
                  ["drugbank.DB00619", 12], ["drugbank.DB12141", 13], ["drugbank.DB06589", 14],
                  ["drugbank.DB08865", 14], ["drugbank.DB11829", 15], ["drugbank.DB06626", 16],
                  ["drugbank.DB09063", 16], ["drugbank.DB11828", 17], ["drugbank.DB00317", 18],
                  ["drugbank.DB08877", 19], ["drugbank.DB00675", 20], ["drugbank.DB08901", 21],
                  ["drugbank.DB00945", 22], ["drugbank.DB00171", 23], ["drugbank.DB00756", 23],
                  ["drugbank.DB06803", 23], ["drugbank.DB08875", 24], ["drugbank.DB08912", 25],
                  ["drugbank.DB11638", 26], ["drugbank.DB04868", 27], ["drugbank.DB08916", 27],
                  ["drugbank.DB00257", 28], ["drugbank.DB04813", 29], ["drugbank.DB09053", 29],
                  ["drugbank.DB02709", 30], ["drugbank.DB12695", 31], ["drugbank.DB01204", 32],
                  ["drugbank.DB01259", 33], ["drugbank.DB09078", 34], ["drugbank.DB00637", 35],
                  ["drugbank.DB04395", 35], ["drugbank.DB08896", 36], ["drugbank.DB12001", 36],
                  ["drugbank.DB00882", 37], ["drugbank.DB11886", 37], ["drugbank.DB00342", 38],
                  ["drugbank.DB00841", 38], ["drugbank.DB01110", 38], ["drugbank.DB01248", 38],
                  ["drugbank.DB00477", 39], ["drugbank.DB01127", 39], ["drugbank.DB00623", 40],
                  ["drugbank.DB09330", 40], ["drugbank.DB12874", 40], ["drugbank.DB00515", 41],
                  ["drugbank.DB00755", 41], ["drugbank.DB00988", 42], ["drugbank.DB02010", 42],
                  ["drugbank.DB02546", 42], ["drugbank.DB04315", 42], ["drugbank.DB08895", 42],
                  ["drugbank.DB13050", 42], ["drugbank.DB00157", 43], ["drugbank.DB11697", 43],
                  ["drugbank.DB11800", 43], ["drugbank.DB00481", 44], ["drugbank.DB00640", 44],
                  ["drugbank.DB00783", 44], ["drugbank.DB09115", 44], ["drugbank.DB10772", 44],
                  ["drugbank.DB11963", 44], ["drugbank.DB12130", 44], ["drugbank.DB12267", 44]]
    true_drugs = ["drugbank.DB00197", "drugbank.DB00305", "drugbank.DB00309", "drugbank.DB00313", "drugbank.DB00317",
                  "drugbank.DB00361", "drugbank.DB00441", "drugbank.DB00445", "drugbank.DB00515", "drugbank.DB00530",
                  "drugbank.DB00544", "drugbank.DB00563", "drugbank.DB00570", "drugbank.DB00586", "drugbank.DB00642",
                  "drugbank.DB00707", "drugbank.DB00773", "drugbank.DB00822", "drugbank.DB00945", "drugbank.DB00958",
                  "drugbank.DB00982", "drugbank.DB01030", "drugbank.DB01041", "drugbank.DB01143", "drugbank.DB01169",
                  "drugbank.DB01181", "drugbank.DB01229", "drugbank.DB01248", "drugbank.DB01262", "drugbank.DB01396",
                  "drugbank.DB04743", "drugbank.DB05578", "drugbank.DB08865", "drugbank.DB08916", "drugbank.DB09035",
                  "drugbank.DB09037", "drugbank.DB09063", "drugbank.DB09079", "drugbank.DB09559", "drugbank.DB11363",
                  "drugbank.DB11672", "drugbank.DB11714", "drugbank.DB11737", "drugbank.DB12130", "drugbank.DB12156",
                  "drugbank.DB12267", "drugbank.DB12459", "drugbank.DB13145", "drugbank.DB13164", "drugbank.DB15133",
                  "drugbank.DB15568", "drugbank.DB15569", "drugbank.DB15685", "drugbank.DB15822", "drugbank.DB16183"]
    permutations = 1000
    only_approved_drugs = True
    result = run_drug_validation(base_url, api_key, test_drugs, true_drugs, permutations, only_approved_drugs)
    print("Drug Validation completed:", json.dumps(result, indent=2))

    # Test Module Validation
    true_drugs = ["drugbank.DB00014", "drugbank.DB00115", "drugbank.DB00158", "drugbank.DB00176", "drugbank.DB00206",
        "drugbank.DB00323", "drugbank.DB00328", "drugbank.DB00382", "drugbank.DB00502", "drugbank.DB00656",
        "drugbank.DB00674", "drugbank.DB00679", "drugbank.DB00681", "drugbank.DB00683", "drugbank.DB00715",
        "drugbank.DB00746", "drugbank.DB00834", "drugbank.DB00850", "drugbank.DB00934", "drugbank.DB00960",
        "drugbank.DB00975", "drugbank.DB00981", "drugbank.DB01017", "drugbank.DB01043", "drugbank.DB01050",
        "drugbank.DB01065", "drugbank.DB01212", "drugbank.DB01356", "drugbank.DB01618", "drugbank.DB03128",
        "drugbank.DB04115", "drugbank.DB04815", "drugbank.DB04864", "drugbank.DB07352", "drugbank.DB08846",
        "drugbank.DB09061", "drugbank.DB11094", "drugbank.DB11473", "drugbank.DB11672", "drugbank.DB12052",
        "drugbank.DB12110", "drugbank.DB12216"]
    permutations = 1000
    only_approved_drugs = True
    validation_type = "module"
    module_member_type = "gene"
    module_members = ["entrez.100293534", "entrez.102", "entrez.10213", "entrez.10347", "entrez.10452", "entrez.10857",
        "entrez.10858", "entrez.1139", "entrez.1141", "entrez.1191", "entrez.1195", "entrez.1200", "entrez.124152",
        "entrez.1378", "entrez.1380", "entrez.1392", "entrez.1471", "entrez.1528", "entrez.1559", "entrez.1565",
        "entrez.1604", "entrez.1636", "entrez.1718", "entrez.1803", "entrez.1808", "entrez.1965", "entrez.2",
        "entrez.2023", "entrez.2041", "entrez.2099", "entrez.2147", "entrez.2191", "entrez.2322", "entrez.23237",
        "entrez.23385", "entrez.23607", "entrez.23621", "entrez.2526", "entrez.25836", "entrez.26330", "entrez.267",
        "entrez.27328", "entrez.274", "entrez.2932", "entrez.29761", "entrez.29978", "entrez.29979", "entrez.3077",
        "entrez.3127", "entrez.3162", "entrez.341", "entrez.3416", "entrez.3479", "entrez.348", "entrez.3480",
        "entrez.3481", "entrez.3482", "entrez.351", "entrez.3553", "entrez.3630", "entrez.3635", "entrez.3643",
        "entrez.3952", "entrez.4129", "entrez.4137", "entrez.4225", "entrez.43", "entrez.4353", "entrez.4524",
        "entrez.4736", "entrez.479", "entrez.4846", "entrez.4852", "entrez.4886", "entrez.4887", "entrez.4889",
        "entrez.4893", "entrez.498", "entrez.51225", "entrez.51338", "entrez.51374", "entrez.51738", "entrez.51741",
        "entrez.5328", "entrez.53339", "entrez.5336", "entrez.54209", "entrez.5447", "entrez.5468", "entrez.5499",
        "entrez.55676", "entrez.5621", "entrez.5649", "entrez.5663", "entrez.5664", "entrez.5697", "entrez.57091",
        "entrez.581", "entrez.5819", "entrez.590", "entrez.596", "entrez.627", "entrez.63929", "entrez.6449",
        "entrez.6517", "entrez.6648", "entrez.6653", "entrez.7018", "entrez.7019", "entrez.712", "entrez.7124",
        "entrez.7167", "entrez.718", "entrez.720", "entrez.7376", "entrez.7422", "entrez.7447", "entrez.7706",
        "entrez.7782", "entrez.7917", "entrez.79890", "entrez.801", "entrez.8202", "entrez.8301", "entrez.8350",
        "entrez.836", "entrez.84675", "entrez.84676", "entrez.84898", "entrez.8633", "entrez.945", "entrez.9510"]
    result =  run_module_validation(base_url,api_key,module_members,module_member_type,true_drugs,permutations,only_approved_drugs)
    print("Module Validation completed:", json.dumps(result, indent=2))

    # Test Joint Validation
    result = run_joint_validation(base_url,api_key,module_members,module_member_type,true_drugs,true_drugs,permutations,only_approved_drugs)
    print("Joint Validation completed:", json.dumps(result, indent=2))
