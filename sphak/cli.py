import argparse
from sphak.main import analyze_sequence

def main():
    parser = argparse.ArgumentParser(description="SPHAK CLI Tool")
    parser.add_argument('--input', required=True, help='Path to input FASTA file')
    args = parser.parse_args()

    results = analyze_sequence(args.input)

    print("Sequence_ID\tBest_Family\tPrediction\tPosterior\tCoverage")
    for r in results:
        print(f"{r['sequence_id']}\t{r['best_family']}\t{r['prediction']}\t{r['posterior']}\t{r['coverage']}")

