import pandas as pd
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from bloom_filter2 import BloomFilter
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# === Function Definitions ===
def generate_kmers(sequence, k):
    """Generate k-mers from a given sequence."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def filter_kmers(kmers):
    """Filter k-mers based on required criteria (e.g., remove ambiguous characters)."""
    return [kmer for kmer in kmers if 'X' not in kmer]

# === Load the Test Dataset ===
test_df = pd.read_csv('out_of_sample.csv')

# === Load the Reference Database ===
with open('animal_reference_database_including_out_of_sample.pkl', 'rb') as f:  # Add plant reference database when dealing with plant data 
    data = pickle.load(f)

family_kmers = data['family_kmers']
total_kmers = data['total_kmers']

# === Create k-mer Sets for Each Family ===
family_kmer_sets = {}
family_valid_ks = {}

for family, kmers_obj in family_kmers.items():
    family_kmer_sets[family] = set(kmers_obj['host_positive'].keys()).union(set(kmers_obj['host_negative'].keys()))
    valid_kmers = set(len(k) for k in kmers_obj['host_positive'].keys()).union(set(len(k) for k in kmers_obj['host_negative'].keys()))
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

# === Processing Loop ===
best_families = []
y_true = []
y_scores = []
predictions = []

for idx, row in test_df.iterrows():
    sequence = row['Sequence']
    actual = row['Human']  # 'Plant' if plant data
    y_true.append(actual)

    # === Generate k-mers for Testing ===
    test_kmers = set()
    for k in range(6, 7):  # k-mer range: adjust if needed
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

    best_families.append(best_family)

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
    test_df.at[idx, 'Coverage'] = coverage

    # === Probability Calculation ===
    if best_family not in family_kmers:
        posterior = 0.5
    else:
        family_data = family_kmers[best_family]
        total_host_positive = total_kmers[best_family]['host_positive']
        total_host_negative = total_kmers[best_family]['host_negative']
        total_family = total_host_positive + total_host_negative

        if total_family == 0:
            posterior = 0.5
        else:
            valid_ks = family_valid_ks.get(best_family, set())
            if not valid_ks:
                posterior = 0.5
            else:
                prior_positive = prior_negative = 0.5
                log_p_positive = log_p_negative = 0.0
                unique_positions_contributed = set()

                for k in valid_ks:
                    kmers = filter_kmers(generate_kmers(sequence, k))
                    vocab_size = 20 ** k  # protein assumption

                    for i, kmer in enumerate(kmers):
                        if kmer not in family_kmer_sets[best_family]:
                            continue

                        positions_covered = set(range(i, i + k))
                        if positions_covered.isdisjoint(unique_positions_contributed):
                            h = family_data['host_positive'].get(kmer, 0)
                            nh = family_data['host_negative'].get(kmer, 0)

                            smoothing_factor = 0.1
                            p_positive = (h + smoothing_factor) / (total_host_positive + smoothing_factor * vocab_size)
                            p_negative = (nh + smoothing_factor) / (total_host_negative + smoothing_factor * vocab_size)

                            # Temperature scaling
                            temperature = 1.5
                            p_positive = p_positive ** temperature
                            p_negative = p_negative ** temperature

                            # Normalize
                            p_sum = p_positive + p_negative
                            p_positive /= p_sum
                            p_negative /= p_sum

                            log_p_positive += math.log(p_positive)
                            log_p_negative += math.log(p_negative)

                            unique_positions_contributed.update(positions_covered)

                if len(unique_positions_contributed) == 0:
                    posterior = 0.5
                else:
                    log_p_positive /= len(unique_positions_contributed)
                    log_p_negative /= len(unique_positions_contributed)

                    log_likelihood_positive = math.log(prior_positive) + log_p_positive
                    log_likelihood_negative = math.log(prior_negative) + log_p_negative
                    max_log = max(log_likelihood_positive, log_likelihood_negative)

                    denom = math.exp(log_likelihood_positive - max_log) + math.exp(log_likelihood_negative - max_log)
                    posterior = math.exp(log_likelihood_positive - max_log) / denom

    posterior = np.clip(posterior, 0.0, 1.0)
    y_scores.append(posterior)
    predictions.append(1 if posterior > 0.5 else 0)

# === Evaluation ===
accuracy = accuracy_score(y_true, predictions)
roc_auc = roc_auc_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# === Save Results ===
test_df['Best_Family'] = best_families
test_df['Prediction'] = predictions
test_df['Prediction_Score'] = y_scores
test_df.to_csv('reference_database.csv', index=False)
