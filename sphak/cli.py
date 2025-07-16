import os
from pathlib import Path
import sphak

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input FASTA file")
    parser.add_argument("--host_type", choices=["animal", "plant"], required=True, help="Host type")
    args = parser.parse_args()

    # Fix: Get the correct absolute path to the .pkl file inside the installed sphak package
    package_dir = Path(sphak.__file__).parent
    db_path = package_dir / "data" / f"{args.host_type}_reference_database.pkl"

    if not db_path.exists():
        raise FileNotFoundError(f"Reference database not found for host type '{args.host_type}' at {db_path}")

    # Now pass `db_path` to your main logic
    from sphak.main import analyze_sequence
    analyze_sequence(args.input, db_path)
