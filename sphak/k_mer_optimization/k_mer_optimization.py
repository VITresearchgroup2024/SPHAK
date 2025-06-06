import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pandas as pd

# Function 1
def generate_kmers(sequence, k):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# Function 2
def filter_kmers(kmers):
    return [kmer for kmer in kmers if 'X' not in kmer]

# Function 3
def process_kmers_by_family(dataframe, k_range):
    family_kmers = defaultdict(lambda: {'host_positive': {}, 'host_negative': {}})
    family_kmer_counts = defaultdict(lambda: defaultdict(lambda: {'host_positive': set(), 'host_negative': set()}))
    total_kmers = defaultdict(lambda: {'host_positive': 0, 'host_negative': 0})

    for _, row in dataframe.iterrows():
        sequence, family, host_status = row['Sequence'], row['Family'], row['Human']
        for k in k_range:
            kmers = filter_kmers(generate_kmers(sequence, k))
            for kmer in kmers:
                if host_status == 1:
                    family_kmer_counts[family][kmer]['host_positive'].add(sequence)
                else:
                    family_kmer_counts[family][kmer]['host_negative'].add(sequence)

    for family, kmer_dict in family_kmer_counts.items():
        for kmer, counts in kmer_dict.items():
            pos_count = len(counts['host_positive'])
            neg_count = len(counts['host_negative'])

            if pos_count + neg_count > 40:
                if counts['host_positive']:
                    family_kmers[family]['host_positive'][kmer] = pos_count
                else:
                    family_kmers[family]['host_negative'][kmer] = neg_count
                total_kmers[family]['host_positive'] += pos_count
                total_kmers[family]['host_negative'] += neg_count

    return family_kmers, total_kmers

# Function 4
def remove_shared_and_overlapping_kmers(family_kmers):
    shared_kmers = set()
    kmer_to_families = defaultdict(set)

    for family, kmers_obj in family_kmers.items():
        for kmer in kmers_obj['host_positive']:
            kmer_to_families[kmer].add(family)
        for kmer in kmers_obj['host_negative']:
            kmer_to_families[kmer].add(family)

    for kmer, families in kmer_to_families.items():
        if len(families) > 1:
            shared_kmers.add(kmer)

    for family, kmers_obj in family_kmers.items():
        kmers_obj['host_positive'] = {
            k: v for k, v in kmers_obj['host_positive'].items() if k not in shared_kmers}
        kmers_obj['host_negative'] = {
            k: v for k, v in kmers_obj['host_negative'].items() if k not in shared_kmers}

    for family, kmers_obj in family_kmers.items():
        pos_set = set(kmers_obj['host_positive'].keys())
        neg_set = set(kmers_obj['host_negative'].keys())
        overlap = pos_set.intersection(neg_set)
        kmers_obj['host_positive'] = {k: v for k, v in kmers_obj['host_positive'].items() if k not in overlap}
        kmers_obj['host_negative'] = {k: v for k, v in kmers_obj['host_negative'].items() if k not in overlap}

    return family_kmers


# Load training dataset
train_df = pd.read_csv('./train_data.csv')

# Define k-range for optimization
k_range_full = list(range(6, 7))  # From k=3 to k=251 or analysing each kmers one at a time

# Dictionary to store results per k
all_k_results = {}

for k in k_range_full:
    print(f"Processing k = {k}...")

    # Step 1: Process k-mers for current k
    family_kmers, total_kmers = process_kmers_by_family(train_df, [k])

    # Step 2: Remove shared and overlapping k-mers
    family_kmers_cleaned = remove_shared_and_overlapping_kmers(family_kmers)

    # Step 3: Store in dict instead of saving to file
    all_k_results[k] = {
        'family_kmers': dict(family_kmers_cleaned),
        'total_kmers': dict(total_kmers)
    }

print("✅ All k-mer training results stored in memory.")


# Choose best k for testing (based on evaluation logic or manually e.g., k=6)
best_k_for_testing = 6

# Retrieve best model from dict
data = all_k_results.get(best_k_for_testing, None)
if data is None:
    raise ValueError(f"No data found for k={best_k_for_testing}. Please check training.")

family_kmers = data['family_kmers']
total_kmers = data['total_kmers']


# Create k-mer sets
family_kmer_sets = {}
family_valid_ks = {}

for family, kmers_obj in family_kmers.items():
    family_kmer_sets[family] = set(kmers_obj['host_positive'].keys()).union(
        set(kmers_obj['host_negative'].keys()))
    valid_kmers = set(len(k) for k in kmers_obj['host_positive'].keys()).union(
        set(len(k) for k in kmers_obj['host_negative'].keys()))
    family_valid_ks[family] = valid_kmers


# Load test data
test_df = pd.read_csv('test_data.csv')

# Initialize lists for evaluation
best_families = []
y_true = []
y_scores = []
predictions = []

for _, row in test_df.iterrows():
    sequence = row['Sequence']
    actual = row['Human']
    y_true.append(actual)

    # Generate test k-mers
    test_kmers = set()
    for k in [best_k_for_testing]:
        kmers = filter_kmers(generate_kmers(sequence, k))
        test_kmers.update(kmers)

    # Match to family
    best_family = None
    max_overlap = -1
    for family, kmers_set in family_kmer_sets.items():
        candidate_kmers = {kmer for kmer in test_kmers if kmer in kmers_set}
        overlap = len(candidate_kmers.intersection(family_kmer_sets[family]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_family = family
    best_families.append(best_family)

    # Coverage calculation
    covered_positions = set()
    if best_family:
        for k in [best_k_for_testing]:
            kmers = filter_kmers(generate_kmers(sequence, k))
            for i, kmer in enumerate(kmers):
                if kmer in family_kmer_sets[best_family]:
                    for pos in range(i, i + k):
                        covered_positions.add(pos)
    coverage = len(covered_positions) / len(sequence) if sequence else 0.0
    test_df.at[_, 'Coverage'] = coverage

    # Probability calculation
    if best_family not in family_kmers:
        posterior = 0.5
    else:
        family_data = family_kmers[best_family]
        total_pos = total_kmers[best_family]['host_positive']
        total_neg = total_kmers[best_family]['host_negative']
        total = total_pos + total_neg
        if total == 0:
            posterior = 0.5
        else:
            valid_ks = family_valid_ks.get(best_family, set())
            if not valid_ks:
                posterior = 0.5
            else:
                log_p_pos = log_p_neg = 0.0
                unique_positions_contributed = set()

                for k in valid_ks:
                    kmers = filter_kmers(generate_kmers(sequence, k))
                    vocab_size = 20 ** k
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


# Assign predictions back to DataFrame
test_df['Best_Family'] = best_families
test_df['Prediction'] = predictions
test_df['Prediction_Score'] = y_scores


# Evaluate
accuracy = accuracy_score(y_true, predictions)
roc_auc = roc_auc_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Calculate FP and FN
false_positives = sum((pred == 1) & (true == 0) for pred, true in zip(predictions, y_true))
false_negatives = sum((pred == 0) & (true == 1) for pred, true in zip(predictions, y_true))

print(f"\nFalse Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")


# Count FP and FN per family
from collections import defaultdict
family_stats = defaultdict(lambda: {'FP': 0, 'FN': 0})

for idx, row in test_df.iterrows():
    true_label = row['Human']
    pred_label = row['Prediction']
    family = row['Best_Family'] if 'Best_Family' in row else 'Unknown'

    if pred_label == 1 and true_label == 0:
        family_stats[family]['FP'] += 1
    elif pred_label == 0 and true_label == 1:
        family_stats[family]['FN'] += 1

print("\nFalse Positives and False Negatives per Family:")
for fam, stats in family_stats.items():
    print(f"Family: {fam}, FP: {stats['FP']}, FN: {stats['FN']}")

# Count how many families had at least one FP or FN
fp_families = set(fam for fam, stat in family_stats.items() if stat['FP'] > 0)
fn_families = set(fam for fam, stat in family_stats.items() if stat['FN'] > 0)

print(f"\nNumber of Families with False Positives: {len(fp_families)}")
print(f"Number of Families with False Negatives: {len(fn_families)}")


# Count incorrect family predictions
incorrect_families = (test_df['Family'] != test_df['Best_Family']).sum()
total_samples = len(test_df)

print(f"\nIncorrect Family Predictions: {incorrect_families} out of {total_samples} "
      f"({(incorrect_families / total_samples) * 100:.2f}%)")

# Save final results
test_df.to_csv('sphak_result.csv', index=False)
print("\n✅ Final predictions saved to 'test_predictions_with_families_deduplication.csv'")
