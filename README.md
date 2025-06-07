# SPHAK

**SPHAK** is a simple proteome-based sequence similarity framework that could quantify spillover risk and predict viral family in animal and plant kingdom.

## Project Overview

Traditional models rely on ecological or phenotypic features, or focus primarily on genomic sequences, to predict host spillover. SPHAK identifies proteome-level sequence patterns specific to animal- and plant-infecting viruses to:

- Predict whether a virus originates from an animal or plant host
- Score the likelihood of host switching or spillover
- Generalize across diverse virus families and protein types

## 🔧 Pipeline Overview

SPHAK involves the following steps:

### 1. **k-mer size optimization**
- To identify an optimal k-mer size that captures discriminative sequence patterns in both animal- and plant-infecting viruses, enabling accurate downstream host prediction and spillover analysis. The k-mer size of 6 is fixed as an optimum k-mer size in animal and plant viruses.

### 2. **Training**
- Training is performed to generate the reference database by extracting host-specific k-mers from curated proteomes. A k-mer size of 6 is used, with a minimum occurrence threshold of 40 for the animal dataset and 5 for the plant dataset, ensuring the selection of representative and high-confidence host-specific sequence patterns.

### 3. **Reference Database Preparation**
- The reference database serves as the foundation of SPHAK. It is built from host-specific proteome sequences, and is used to perform predictions by comparing k-mer patterns in viral protein sequences against known host classes (e.g., animal or plant).
- **Process**:
  - Split each host proteome into overlapping **k-mers** (default: 6-mers)
  - Filter out low-complexity or ambiguous k-mers
  - Store host-specific k-mer dictionaries with frequency counts


### 4. **Testing**
- Apply the SPHAK method to new or unlabelled viral protein sequences.
The method outputs predicted viral family and spillover risk through SP score(Spillover Potential score).
- **SP score calculation**: 
- **Formula**:

SP score = e^(log P(A) − max_log) / (e^(log P(A) − max_log) + e^(log P(B) − max_log))


               



---

## ⚙️ Installation

```bash
git clone https://github.com/VITresearchgroup2024/SPHAK.git
cd SPHAK
pip install -r requirements.txt

Adjust installation steps as needed for your environment.

```
---

📚 Citation

- Manuscript under preparation. Please contact the authors before citing.
Developed by the VIT Research Group (2024–2025).

📬 Contact
For questions, please open an issue or email:
✉️ vibin@cmscollege.ac.in
✉️ vinning372@gmail.com
✉️ ananyaprakash0105@gmail.com
✉️ kavyasree6424@gmail.com

