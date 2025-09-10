import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pandas as pd

def generate_kmers(sequence, k):
    """Generate overlapping k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def filter_kmers(kmers):
    """Filter out k-mers containing 'X'."""
    return [kmer for kmer in kmers if 'X' not in kmer]

# Load reference database
with open('animal_reference_database_excluding_out_of_sample.pkl', 'rb') as f:
    data = pickle.load(f)
family_kmers = data['family_kmers']
total_kmers = data['total_kmers']

# Load test dataset
test_df = pd.read_csv('animal_test.csv')  # or plant_test.csv

# Define k-range based on saved k-mers
all_kmers = []
for family_data in family_kmers.values():
    all_kmers.extend(list(family_data['host_positive'].keys()))
    all_kmers.extend(list(family_data['host_negative'].keys()))
k_range = list(set(len(kmer) for kmer in all_kmers))

# Prepare sets of k-mers for faster lookup
family_kmer_sets = {}
for family, kmers_obj in family_kmers.items():
    family_kmer_sets[family] = set(kmers_obj['host_positive'].keys()).union(
                               set(kmers_obj['host_negative'].keys()))

# Store results
y_true = []
y_scores = []
predictions = []

for _, row in test_df.iterrows():
    sequence = row['Sequence']
    actual = row['Human']
    y_true.append(actual)

    # Generate k-mers
    test_kmers = set()
    for k in k_range:
        kmers = filter_kmers(generate_kmers(sequence, k))
        test_kmers.update(kmers)

    # Find best matching family based on overlap
    best_family = None
    max_overlap = -1
    for family, kmers_set in family_kmer_sets.items():
        overlap = len(test_kmers.intersection(kmers_set))
        if overlap > max_overlap:
            max_overlap = overlap
            best_family = family

    # Calculate coverage (optional for analysis)
    covered_positions = set()
    if best_family:
        for k in k_range:
            kmers = filter_kmers(generate_kmers(sequence, k))
            for i, kmer in enumerate(kmers):
                if kmer in family_kmer_sets[best_family]:
                    covered_positions.update(range(i, i+k))
    coverage = len(covered_positions) / len(sequence) if sequence else 0.0
    test_df.at[_, 'Coverage'] = coverage

    # Calculate probability using Laplace smoothing and log transformation
    if not best_family or best_family not in family_kmers:
        posterior = 0.5
    else:
        family_data = family_kmers[best_family]
        total_pos = total_kmers[best_family]['host_positive']
        total_neg = total_kmers[best_family]['host_negative']
        total = total_pos + total_neg
        if total == 0:
            posterior = 0.5
        else:
            log_p_pos = 0.0
            log_p_neg = 0.0
            unique_positions_contributed = set()

            valid_ks = set(len(kmer) for kmer in family_kmer_sets[best_family])
            for k in valid_ks:
                kmers = filter_kmers(generate_kmers(sequence, k))
                vocab_size = 20 ** k  # Approximate for amino acids
                for i, kmer in enumerate(kmers):
                    if kmer not in family_kmer_sets[best_family]:
                        continue
                    positions_covered = set(range(i, i + k))
                    if positions_covered.isdisjoint(unique_positions_contributed):
                        h = family_data['host_positive'].get(kmer, 0)
                        nh = family_data['host_negative'].get(kmer, 0)
                        smoothing = 0.1
                        p_pos = (h + smoothing) / (total_pos + smoothing * vocab_size)
                        p_neg = (nh + smoothing) / (total_neg + smoothing * vocab_size)
                        temp = 1.5
                        p_pos **= temp
                        p_neg **= temp
                        norm = p_pos + p_neg
                        p_pos /= norm
                        p_neg /= norm
                        log_p_pos += math.log(p_pos)
                        log_p_neg += math.log(p_neg)
                        unique_positions_contributed.update(positions_covered)

            if not unique_positions_contributed:
                posterior = 0.5
            else:
                log_p_pos /= len(unique_positions_contributed)
                log_p_neg /= len(unique_positions_contributed)
                log_like_pos = math.log(0.5) + log_p_pos
                log_like_neg = math.log(0.5) + log_p_neg
                max_log = max(log_like_pos, log_like_neg)
                denom = math.exp(log_like_pos - max_log) + math.exp(log_like_neg - max_log)
                posterior = math.exp(log_like_pos - max_log) / denom

    posterior = np.clip(posterior, 0.0, 1.0)
    y_scores.append(posterior)
    predictions.append(1 if posterior > 0.5 else 0)

# Assign predictions back to the DataFrame
test_df['Prediction'] = predictions
test_df['Prediction_Score'] = y_scores

# Evaluate
accuracy = accuracy_score(y_true, predictions)
roc_auc = roc_auc_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Animal/Plant Classification')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Save results
test_df.to_csv('animal_test_results.csv', index=False)
print("✅ Final test results saved to 'animal_test_results.csv'")
