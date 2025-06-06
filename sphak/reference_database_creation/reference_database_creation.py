import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import pandas as pd

def generate_kmers(sequence, k):
    """Generate overlapping k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def filter_kmers(kmers):
    """Filter out k-mers containing 'X'."""
    return [kmer for kmer in kmers if 'X' not in kmer]

def process_kmers_by_family(dataframe, k_range):
    """
    Process k-mers from sequences, considering only those appearing in >3 unique sequences per family.
    Classify a k-mer as 'host_positive' if it has ever appeared in the host_positive set, else as 'host_negative'.
    """
    family_kmers = defaultdict(lambda: {'host_positive': {}, 'host_negative': {}})
    family_kmer_counts = defaultdict(lambda: defaultdict(lambda: {'host_positive': set(), 'host_negative': set()}))
    total_kmers = defaultdict(lambda: {'host_positive': 0, 'host_negative': 0})

    # Step 1: Collect k-mer occurrences across sequences
    for _, row in dataframe.iterrows():
        sequence, family, human = row['Sequence'], row['Family'], row['Human'] # Change to Plant, while dealing with plant data
        for k in k_range:
            kmers = filter_kmers(generate_kmers(sequence, k))
            for kmer in kmers:
                if human == 1:
                    family_kmer_counts[family][kmer]['host_positive'].add(sequence)
                else:
                    family_kmer_counts[family][kmer]['host_negative'].add(sequence)

    # Step 2: Classify k-mers based on their presence in the host_positive set
    for family, kmer_dict in family_kmer_counts.items():
        for kmer, counts in kmer_dict.items():
            host_pos_count, host_neg_count = len(counts['host_positive']), len(counts['host_negative'])
            if host_pos_count + host_neg_count > 40:  # Only consider k-mers appearing in >40 unique sequences in case of animals and >5 unique sequences in case of plants
                if counts['host_positive']:  # If the k-mer has ever appeared in the host_positive set
                    family_kmers[family]['host_positive'][kmer] = host_pos_count
                else:  # Otherwise, classify it as host_negative
                    family_kmers[family]['host_negative'][kmer] = host_neg_count

                # Update total k-mer counts
                total_kmers[family]['host_positive'] += host_pos_count
                total_kmers[family]['host_negative'] += host_neg_count

    return family_kmers, total_kmers

def plot_kmers_distribution_combined(family_kmers, k_range):
    """Plot unique k-mer distributions for all families on a single graph."""
    plt.figure(figsize=(12, 7))

    for family, kmers_obj in family_kmers.items():
        host_pos_counts = [len([kmer for kmer in kmers_obj['host_positive'] if len(kmer) == k]) for k in k_range]
        host_neg_counts = [len([kmer for kmer in kmers_obj['host_negative'] if len(kmer) == k]) for k in k_range]

        plt.plot(k_range, host_pos_counts, marker='o', linestyle='-', label=f'{family} - Host Positive', alpha=0.8)
        plt.plot(k_range, host_neg_counts, marker='x', linestyle='--', label=f'{family} - Host Negative', alpha=0.8)

    plt.title('Unique K-mer Distribution Across Families')
    plt.xlabel('K-mer Size')
    plt.ylabel('Unique K-mer Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# Load dataset
train_df = pd.read_csv('./animal_train.csv') # or plant_train
k_range = list(range(6,7))  # Consider k-mers from size 6 to 6 (adjust as needed)
# Process k-mers
family_kmers, total_kmers = process_kmers_by_family(train_df, k_range)

# Remove shared k-mers (k-mers seen in more than one family)
shared_kmers = set()
kmer_to_families = defaultdict(set)

# Identify shared k-mers
for family, kmers_obj in family_kmers.items():
    for kmer in kmers_obj['host_positive']:
        kmer_to_families[kmer].add(family)
    for kmer in kmers_obj['host_negative']:
        kmer_to_families[kmer].add(family)

# Mark k-mers that appear in more than one family
for kmer, families in kmer_to_families.items():
    if len(families) > 1:
        shared_kmers.add(kmer)

# Remove shared k-mers from family_kmers
for family, kmers_obj in family_kmers.items():
    kmers_obj['host_positive'] = {kmer: count for kmer, count in kmers_obj['host_positive'].items() if kmer not in shared_kmers}
    kmers_obj['host_negative'] = {kmer: count for kmer, count in kmers_obj['host_negative'].items() if kmer not in shared_kmers}

# Remove k-mers present in both host_positive and host_negative within the same family
for family, kmers_obj in family_kmers.items():
    host_pos_kmers = set(kmers_obj['host_positive'].keys())
    host_neg_kmers = set(kmers_obj['host_negative'].keys())
    overlapping_kmers = host_pos_kmers.intersection(host_neg_kmers)

    # Remove overlapping k-mers from both host_positive and host_negative
    kmers_obj['host_positive'] = {kmer: count for kmer, count in kmers_obj['host_positive'].items() if kmer not in overlapping_kmers}
    kmers_obj['host_negative'] = {kmer: count for kmer, count in kmers_obj['host_negative'].items() if kmer not in overlapping_kmers}

# Plot results
plot_kmers_distribution_combined(family_kmers, k_range)

# Save results
with open('animal_reference_database_excluding_out_of_sample.pkl', 'wb') as f:
    pickle.dump({'family_kmers': dict(family_kmers), 'total_kmers': dict(total_kmers)}, f)

print("Filtered k-mers with occurrences and total k-mer counts saved to 'animal_reference_database_excluding_out_of_sample.pkl'")
