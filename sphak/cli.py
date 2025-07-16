import argparse
import os
import sys
from sphak.main import analyze_sequence

def main():
    parser = argparse.ArgumentParser(description="Run SPHAK analysis")
    parser.add_argument("--input", required=True, help="Path to input FASTA file")
    parser.add_argument("--host_type", choices=["animal", "plant"], required=True, help="Type of host: 'animal' or 'plant'")
    args = parser.parse_args()

    # Point to local data directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    db_path = os.path.join(base_dir, f"{args.host_type}_reference_database.pkl")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Reference database not found for host type '{args.host_type}' at {db_path}")

    analyze_sequence(args.input, db_path)

if __name__ == "__main__":
    main()
