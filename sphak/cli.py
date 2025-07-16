import argparse
import os
import sys
from sphak.main import analyze_sequence
import importlib.resources as pkg_resources
import sphak

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to FASTA file')
    parser.add_argument('--host_type', required=True, choices=['animal', 'plant'], help='Host type')
    args = parser.parse_args()

    # Correct path to reference DB inside the installed package
    try:
        with pkg_resources.path(sphak, f"data/{args.host_type}_reference_database.pkl") as db_path:
            analyze_sequence(args.input, str(db_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference database not found for host type '{args.host_type}' in SPHAK's data folder")
