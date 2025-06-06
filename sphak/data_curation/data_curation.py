import os
import pandas as pd
from Bio import SeqIO

# Function 1
def load_metadata(csv_file):
    """
    Load metadata from a CSV file into a DataFrame.
    """
    return pd.read_csv(csv_file)

# Function 2
def load_sequences(fasta_file):
    """
    Load sequences from a FASTA file into a DataFrame.
    """
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    sequence_data = [{"ID": record.id, "Sequence": str(record.seq), "Length": len(record.seq)} for record in sequences]
    return pd.DataFrame(sequence_data)

# Function 3
def merge_data(metadata_df, sequences_df):
    """
    Merge metadata and sequence data on cleaned ID.
    """
    sequences_df['ID_cleaned'] = sequences_df['ID'].str.split('.').str[0]
    merged_df = pd.merge(metadata_df, sequences_df, how='inner', left_on='Accession', right_on='ID_cleaned')
    return merged_df

# Function 4
def extract_virus_column(df, source_col, target_col):
    """
    Extract virus information from a column using regex.
    """
    df.loc[:, target_col] = df[source_col].str.extract(r'\[(.*?)\]')
    return df

# Function 5
def remove_unwanted_columns(df, columns_to_remove):
    """
    Remove unwanted columns from the DataFrame if they exist.
    """
    return df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

# Function 6
def load_host_mapping_csv(mapping_file):
    """
    Load host-to-Host_agg mapping from a CSV file into a DataFrame.
    """
    return pd.read_csv(mapping_file)

# Function 7
def map_host_to_agg_with_mapping(df, host_col, target_col, mapping_df):
    """
    Map the host column to 'human' or 'non_human' using a mapping DataFrame.
    """
    # Create a dictionary from the mapping DataFrame
    mapping_dict = mapping_df.set_index('Host')['Host_agg'].to_dict()

    # Map the Host values using the dictionary
    df[target_col] = df[host_col].map(mapping_dict)
    return df

# Function 8
def add_human_column(df, host_col, human_col):
    """
    Add a column indicating whether the species is Homo sapiens.
    """
    df[human_col] = df[host_col].apply(lambda x: 1 if x == "human" else 0)
    return df

# Function 9
def drop_duplicates_on_sequence(df):
    """
    Drop duplicate sequences from the DataFrame.
    """
    return df.drop_duplicates(subset=['Sequence'])

# Function 10
def remove_blank_host_rows(df, host_col):
    """
    Remove rows where the host column is blank or NaN.
    """
    return df[df[host_col].notnull() & (df[host_col].str.strip() != '')]

# Main script
if __name__ == "__main__":
    # File paths
    csv_files = ["./coronaviridae_raw_data.csv"]  # Add more CSV paths as needed also add files need to be curated
    fasta_files = ["./coronaviridae_raw_data.fasta"]  # Add more FASTA paths as needed also add files need to be curated
    mapping_file = "./host_mapping_animal.csv" # File path for the host mapping CSV file, change name and load 'host_mapping_plant.csv', if handling with plant data

    # Check if CSV and FASTA files are of the same length
    if len(csv_files) != len(fasta_files):
        raise ValueError("The number of CSV files must match the number of FASTA files.")

    # Initialize an empty list to collect DataFrames
    all_data_frames = []

    # Process each CSV and FASTA pair
    for csv_file, fasta_file in zip(csv_files, fasta_files):
        # Load metadata and sequences
        metadata_df = load_metadata(csv_file)
        sequences_df = load_sequences(fasta_file)

        # Merge data
        merged_df = merge_data(metadata_df, sequences_df)

        # Extract virus information
        df_modified = extract_virus_column(merged_df, source_col="GenBank_Title", target_col="Virus")

        # Remove rows with blank Host values
        df_no_blank_host = remove_blank_host_rows(df_modified, host_col='Host')

        # Remove unwanted columns
        unwanted_columns = [
            "Unnamed: 0", "Organism_Name", "GenBank_RefSeq", "Assembly", "Nucleotide", "SRA_Accession",
            "Submitters", "Release_Date", "Isolate", "Genus", "Molecule_type", "Length_x",
            "Genotype", "Segment", "Publications", "Protein", "Geo_Location", "USA",
            "Tissue_Specimen_Source", "Collection_Date", "BioSample", "BioProject", "GenBank_Title", "ID", "Length_y",
            "ID_cleaned"
        ]
        df_col_removed = remove_unwanted_columns(df_no_blank_host, unwanted_columns)

        # Load the mapping
        host_mapping_df = load_host_mapping_csv(mapping_file)

        # Apply the mapping
        df_host_mapped = map_host_to_agg_with_mapping(
            df_col_removed, host_col='Host', target_col='Host_agg', mapping_df=host_mapping_df
        )

        # Add Human column, add Plant column if its plant data
        df_human_added = add_human_column(df_host_mapped, host_col='Host_agg', human_col='Human')

        # Append the cleaned DataFrame to the list
        all_data_frames.append(df_human_added)

    # Concatenate all data frames into one large DataFrame
    all_df_merged = pd.concat(all_data_frames, ignore_index=True)

    # Remove duplicate sequences
    final_cleaned_df = drop_duplicates_on_sequence(all_df_merged)

    # Save the final cleaned dataset
    output_file = "coronaviridae_curated_dataset.csv" 
    final_cleaned_df.to_csv(output_file, index=False)
    print(f"Data processing completed. Cleaned data saved to {output_file}.")
