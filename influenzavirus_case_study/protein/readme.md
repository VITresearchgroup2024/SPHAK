### Influenza Virus Case Study

This part explain the application of SPHAK using Influenza.


1️. Data Curation (data_curation.py)

This script processes raw data to create a clean, merged dataset suitable for modeling:

Loads virus metadata and sequence data.

Extracts virus information from descriptions.

Maps host species to broader categories (e.g., human, animal).

Removes duplicates and unnecessary columns.

Generates a final curated dataset (influenza_curated_protein_dataset.csv) for training and testing.

Key functions:

Load metadata and sequences.

Merge on accession identifiers.

Remove blank or irrelevant entries.

Map hosts to categories using a lookup CSV.

Add columns identifying human infections.

Save cleaned dataset for downstream analysis.

2️. Train-Test Split (split_test_train.py)

The curated dataset is further processed to ensure robust evaluation:

Filters out rare categories that may cause bias.

Combines 'Host_agg', 'Family', and 'Species' to stratify data.

Splits into training and testing sets with 80/20 ratio.

Ensures class balance for fair representation.

Saves final datasets as influenza_train_data.csv and influenza_test_data.csv.

3️. Reference Database Creation (training/)

This stage generates the essential knowledge base for prediction:

Extracts k-mers from sequences (length 6 by default).

Filters out low-frequency and ambiguous sequences.

Identifies k-mers that are host-specific (human vs. animal).

Removes k-mers shared across multiple virus families or overlapping between classes.

Stores curated k-mers for each virus family in animal_reference_database.pkl.

Note: For this project, the full animal reference database is used during testing rather than generating a new one.


▶ How to Run

Data Curation:

``bash
python data_curation.py
``

This generates the curated dataset. Change the paths if needed.

Train-Test Split:

``bash
python split_test_train.py
``

This produces stratified training and testing datasets.Change the paths if needed.


Reference Database Creation:

``bash
python training_script.py
``

📊 Results

The final output allows prediction models to:

Identify sequences likely to spill over to human hosts.

Analyze viral families with host-specific sequence patterns.

Provide insights for surveillance and preventive strategies.

📂 Dataset Sources

Sequence data and metadata from public repositories.

Host mapping curated from existing biological classification resources.

🔬 Notes

The project is designed for research and bioinformatics applications.

The approach focuses on sequence patterns without ecological or geographic data.

Thresholds for filtering k-mers are adjustable depending on dataset size and research requirements.
