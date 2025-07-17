import argparse
import sys
from sphak.main import analyze_sequence
from importlib.resources import files
import sphak.data
import os
import pickle
from pathlib import Path

def show_install_message_once():
    # Store flag in user home directory
    flag_path = Path.home() / ".sphak_first_run"
    if not flag_path.exists():
        print("✅ SPHAK installed and ready to use! Use --help to see options.\n")
        flag_path.write_text("shown")

def main():
    show_install_message_once()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to FASTA file')
    parser.add_argument('--host_type', required=True, choices=['animal', 'plant'], help='Host type')
    args = parser.parse_args()

    db_filename = f"{args.host_type}_reference_database.pkl"

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"FASTA input file not found: {args.input}")

    try:
        db_path = files(sphak.data).joinpath(db_filename)
        if not db_path.exists():
            raise FileNotFoundError()

        with db_path.open('rb') as f:
            reference_data = pickle.load(f)

        analyze_sequence(args.input, reference_data)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference database not found for host type '{args.host_type}' in sphak/data/{db_filename}"
        )
