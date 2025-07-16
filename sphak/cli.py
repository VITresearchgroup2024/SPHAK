from importlib.resources import files
import sphak.data
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to FASTA file')
    parser.add_argument('--host_type', required=True, choices=['animal', 'plant'], help='Host type')
    args = parser.parse_args()

    db_filename = f"{args.host_type}_reference_database.pkl"

    # ✅ check if input file exists FIRST
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"FASTA input file not found at {args.input}")

    try:
        db_path = files(sphak.data).joinpath(db_filename)
        analyze_sequence(args.input, str(db_path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reference database not found for host type '{args.host_type}' "
            f"in package at sphak/data/{db_filename}"
        )
