import os
import argparse
from sphak.main import analyze_sequence  

def main():
    parser = argparse.ArgumentParser(description="SPHAK: Predict host from query sequence")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--host_type", required=True, choices=["plant", "animal"], help="Type of host")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, "data", f"{args.host_type}_reference_database.pkl")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Reference database not found for host type '{args.host_type}' at {db_path}")

    analyze_sequence(fasta_file=args.fasta, reference_db=db_path)
