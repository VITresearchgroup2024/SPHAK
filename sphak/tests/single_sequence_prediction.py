import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math
from pybloom_live import BloomFilter
import pickle
from Bio import SeqIO

# === Function Definitions ===
def generate_kmers(sequence, k):
    """Generate k-mers from a given sequence."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def filter_kmers(kmers):
    """Filter k-mers based on required criteria (e.g., remove ambiguous characters)."""
    return [kmer for kmer in kmers if 'X' not in kmer]

# === Load the Pickled File ===
with open('reference_database.pkl', 'rb') as f:      # give correct path and filename
    data = pickle.load(f)

family_kmers = data['family_kmers']
total_kmers = data['total_kmers']

# === Create k-mer Sets for Each Family ===
family_kmer_sets = {}
family_valid_ks = {}

for family, kmers_obj in family_kmers.items():
    # Store all k-mers as a set for quick intersection calculations
    family_kmer_sets[family] = set(kmers_obj['homo'].keys()).union(set(kmers_obj['non_homo'].keys()))

    # Extract valid k-mer lengths
    valid_kmers = set(len(k) for k in kmers_obj['homo'].keys()).union(set(len(k) for k in kmers_obj['non_homo'].keys()))
    family_valid_ks[family] = valid_kmers

# === Initialize Bloom Filters ===
family_bloom_filters = {}

for family, kmer_set in family_kmer_sets.items():
    if len(kmer_set) == 0:
        print(f"Warning: Family '{family}' has no k-mers and will be skipped.")
        continue

    bloom_filter = BloomFilter(capacity=len(kmer_set), error_rate=0.01)
    for kmer in kmer_set:
        bloom_filter.add(kmer)
    family_bloom_filters[family] = bloom_filter

# === Parse the FASTA File ===
fasta_file = "hku5.fasta"  # Path to your FASTA file
sequences = list(SeqIO.parse(fasta_file, "fasta"))

# === Processing Loop ===
y_true = []
y_scores = []
predictions = []

print("Sequence_ID\tBest_Family\tPrediction\tPrediction_Score\tCoverage")
for record in sequences:
    sequence_id = record.id
    sequence = str(record.seq).upper()  # Convert sequence to uppercase
    actual = 0  # Update this if you have ground truth labels (e.g., 0 or 1)
    y_true.append(actual)

    # === Generate k-mers for Testing ===
    test_kmers = set()
    for k in range(6, 7):  # k-mer range: 7 to 8
        kmers = filter_kmers(generate_kmers(sequence, k))
        test_kmers.update(kmers)

    # === Family Matching ===
    best_family = None
    max_overlap = -1

    for family, bloom_filter in family_bloom_filters.items():
        candidate_kmers = {kmer for kmer in test_kmers if kmer in bloom_filter}
        overlap = len(candidate_kmers.intersection(family_kmer_sets[family]))

        if overlap > max_overlap:
            max_overlap = overlap
            best_family = family

    # === Coverage Calculation ===
    covered_positions = set()
    if best_family:
        for k in range(6, 7):
            kmers = filter_kmers(generate_kmers(sequence, k))
            for i, kmer in enumerate(kmers):
                if kmer in family_kmer_sets[best_family]:
                    for pos in range(i, i + k):
                        covered_positions.add(pos)

    coverage = len(covered_positions) / len(sequence) if sequence else 0.0


    # === Probability Calculation ===
    if best_family not in family_kmers:
        posterior = 0.5
    else:
        family_data = family_kmers[best_family]
        total_homo = total_kmers[best_family]['homo']
        total_non_homo = total_kmers[best_family]['non_homo']
        total_family = total_homo + total_non_homo

        if total_family == 0:
            posterior = 0.5
        else:
            valid_ks = family_valid_ks.get(best_family, set())
            if not valid_ks:
                posterior = 0.5
            else:
                prior_homo = prior_non = 0.5
                log_p_homo = log_p_non = 0.0
                unique_positions_contributed = set()  # Track unique positions

                for k in valid_ks:
                    kmers = filter_kmers(generate_kmers(sequence, k))
                    vocab_size = 20 ** k  # Protein-based assumption

                    for i, kmer in enumerate(kmers):
                        if kmer not in family_kmer_sets[best_family]:
                            continue

                        # Check if any position covered by this k-mer has already contributed
                        positions_covered = set(range(i, i + k))
                        if positions_covered.isdisjoint(unique_positions_contributed):  # No overlap
                            h = family_data['homo'].get(kmer, 0)
                            nh = family_data['non_homo'].get(kmer, 0)

                            # Replace Laplace smoothing with adaptive smoothing
                            smoothing_factor = 0.1  # Reduce smoothing factor
                            p_homo = (h + smoothing_factor) / (total_homo + smoothing_factor * vocab_size)
                            p_non = (nh + smoothing_factor) / (total_non_homo + smoothing_factor * vocab_size)

                            # Apply temperature scaling
                            temperature = 1.5
                            p_homo = p_homo ** temperature
                            p_non = p_non ** temperature

                            # Normalize probabilities
                            p_total = p_homo + p_non
                            p_homo /= p_total
                            p_non /= p_total

                            log_p_homo += math.log(p_homo)
                            log_p_non += math.log(p_non)

                            # Mark these positions as contributed
                            unique_positions_contributed.update(positions_covered)

                # Final posterior probability
                if len(unique_positions_contributed) == 0:
                    posterior = 0.5
                else:
                    log_p_homo /= len(unique_positions_contributed)
                    log_p_non /= len(unique_positions_contributed)

                    log_likelihood_homo = math.log(prior_homo) + log_p_homo
                    log_likelihood_non = math.log(prior_non) + log_p_non
                    max_log = max(log_likelihood_homo, log_likelihood_non)

                    denominator = math.exp(log_likelihood_homo - max_log) + math.exp(log_likelihood_non - max_log)
                    posterior = math.exp(log_likelihood_homo - max_log) / denominator

    posterior = np.clip(posterior, 0.0, 1.0)
    y_scores.append(posterior)
    predictions.append(1 if posterior > 0.5 else 0)

    # Print Results
    print(f"{sequence_id}\t{best_family}\t{predictions[-1]}\t{posterior:.4f}\t{coverage:.4f}")

