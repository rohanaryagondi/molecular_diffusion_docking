"""
Download the ZINC250K Dataset
==============================
ZINC250K is a subset of ~250,000 drug-like molecules from the ZINC database.
It's the standard benchmark for molecular generation papers.

Each molecule is represented as a SMILES string -- a line notation that
encodes molecular structure as text. For example:
    CC(=O)O       -> acetic acid (vinegar)
    c1ccccc1      -> benzene (aromatic ring)
    CC(=O)Oc1ccccc1C(=O)O  -> aspirin

Usage:
    python scripts/download_data.py
"""

import os
import sys
import urllib.request
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ZINC250K is commonly distributed as a text file with one SMILES per line.
# This URL points to the dataset used in the Junction Tree VAE paper (ICML 2018).
ZINC250K_URL = (
    "https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt"
)

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zinc250k.csv")


def download_zinc250k():
    """Download ZINC250K and save as CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Dataset already exists at {OUTPUT_FILE}")
        return

    print(f"Downloading ZINC250K from {ZINC250K_URL} ...")
    tmp_path = OUTPUT_FILE + ".tmp"
    urllib.request.urlretrieve(ZINC250K_URL, tmp_path)

    # Convert from plain text (one SMILES per line) to CSV
    with open(tmp_path, "r") as f_in, open(OUTPUT_FILE, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["smiles"])  # header
        count = 0
        for line in f_in:
            smiles = line.strip()
            if smiles:
                writer.writerow([smiles])
                count += 1

    os.remove(tmp_path)
    print(f"Saved {count} molecules to {OUTPUT_FILE}")


if __name__ == "__main__":
    download_zinc250k()
