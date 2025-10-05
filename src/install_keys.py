#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from validation.generate_NEDREX_API_key import get_api_key

# allow importing your key-gen module
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))



def fetch_key(base_url, label):
    """
    Call get_api_key(...) and extract the 'api_key' field.
    Raises if generation fails or field is missing.
    """
    resp = get_api_key(base_url)
    if not resp:
        raise RuntimeError(f"Failed to generate API key for {label}: {resp}")
    return resp

def update_env(env_path: Path, updates: dict):
    """
    Read (or create) .env at env_path, then set or overwrite each key=value.
    """
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()
    new_lines = []
    seen = set()

    for line in lines:
        if not line.strip() or line.strip().startswith('#') or '=' not in line:
            new_lines.append(line)
            continue
        key, _, _ = line.partition('=')
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    # append missing keys
    for key, val in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(new_lines) + "\n")

def main():
    # locate .env (one level up from src/)
    dotenv_path = find_dotenv()
    if not dotenv_path:
        # fallback: assume .env is one directory above this script
        dotenv_path = str(SCRIPT_DIR.parent / '.env')
    load_dotenv(dotenv_path, override=False)

    # pull base URLs from .env (uppercase)
    exbio_base = os.getenv('EXBIO_BASE_URL')
    nedrex_base = os.getenv('NEDREX_BASE_URL')

    # fetch API keys (get_api_key(None) uses default EXBIO URL)
    exbio_key  = fetch_key(exbio_base,  "EXBIO")
    nedrex_key = fetch_key(nedrex_base, "NEDREX")

    # write back the two LICENSED_KEY vars
    env_file = Path(dotenv_path)
    update_env(env_file, {
        "EXBIO_LICENSED_KEY":  exbio_key,
        "NEDREX_LICENSED_KEY": nedrex_key
    })

    print("Updated .env with EXBIO_LICENSED_KEY and NEDREX_LICENSED_KEY")

if __name__ == "__main__":
    main()