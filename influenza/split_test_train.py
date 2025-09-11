import pandas as pd
from sklearn.model_selection import train_test_split 

# Load dataset
try:
    df = pd.read_csv('/content/influenza_curated_protein_dataset.csv', encoding='latin-1')
except FileNotFoundError:
    print("The specified file was not found.")
    raise

print("Dataset loaded successfully...")

# Check the original dataset
print("Dataset size:", len(df))

print("Loading for filtration...")

# Verify required columns exist
required_columns = ['Host_agg', 'Family', 'Species']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' is missing in the dataset.")

# Filter out categories with fewer than 2 samples in 'Host_agg', 'Family', and 'Species'
valid_hosts = df['Host_agg'].value_counts()[df['Host_agg'].value_counts() > 1].index
valid_families = df['Family'].value_counts()[df['Family'].value_counts() > 1].index
valid_species = df['Species'].value_counts()[df['Species'].value_counts() > 1].index

df_filtered = df[
    df['Host_agg'].isin(valid_hosts) &
    df['Family'].isin(valid_families) &
    df['Species'].isin(valid_species)
]

# Check the filtered dataset
print("Filtered dataset size:", len(df_filtered))

# Create a stratification column by combining 'Host_agg', 'Family', and 'Species'
df_filtered = df_filtered.copy()  # Avoid SettingWithCopyWarning
df_filtered['Stratify_col'] = (
    df_filtered['Host_agg'] + "_" +
    df_filtered['Family'] + "_" +
    df_filtered['Species']
)

# Filter out classes in 'Stratify_col' with fewer than 2 samples
stratify_counts = df_filtered['Stratify_col'].value_counts()
valid_stratify_classes = stratify_counts[stratify_counts > 1].index
df_filtered = df_filtered[df_filtered['Stratify_col'].isin(valid_stratify_classes)]

# Check the filtered dataset size again
print("Filtered dataset size after removing single-sample stratify classes:", len(df_filtered))

# Split dataset
train_df, test_df = train_test_split(
    df_filtered,
    test_size=0.20,  # Adjust as needed
    stratify=df_filtered['Stratify_col'],  # Stratify by the chosen column
    random_state=42  # Ensure reproducibility
)
print("Stratification completed...")
print("Splitting test and train completed...")

# Verify the split
print("Train set size:", len(train_df))
print("Test set size:", len(test_df))
print("Train set distribution:")
print(train_df['Stratify_col'].value_counts())
print("Test set distribution:")
print(test_df['Stratify_col'].value_counts())

# Save the datasets
test_df.to_csv("influenza_test_data.csv", index=False)
print("Test data saved to 'influenza_test_data.csv'")
train_df.to_csv("influenza_train_data.csv", index=False)
print("Train data saved to 'influenza_train_data.csv' ")
