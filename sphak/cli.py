import argparse
import sys
from sphak.main import analyze_sequence
import importlib.resources as pkg_resources
import sphak.data  # 👈 This is important!

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to FASTA file')
    parser.add_argument('--host_type', required=True, choices=['animal', 'plant'], help='Host type')
    args = parser.parse_args()

    db_filename = f"{args.host_type}_reference_database.pkl"

    # Correct way: use sphak.data as the package
    try:
        with pkg_resources.path(sphak.data, db_filename) as db_path:
            analyze_sequence(args.input, str(db_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference database not found for host type '{args.host_type}' "
            f"in package at sphak/data/{db_filename}"
        )
