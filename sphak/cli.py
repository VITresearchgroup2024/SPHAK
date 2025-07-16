import argparse
from sphak.main import analyze_sequence
from importlib.resources import files
import sphak.data  # this must point to the `data/` submodule (which has `__init__.py`)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to FASTA file')
    parser.add_argument('--host_type', required=True, choices=['animal', 'plant'], help='Host type')
    args = parser.parse_args()

    db_filename = f"{args.host_type}_reference_database.pkl"

    try:
        db_path = files(sphak.data).joinpath(db_filename)
        analyze_sequence(args.input, str(db_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference database not found for host type '{args.host_type}' "
            f"in package at sphak/data/{db_filename}"
        )
