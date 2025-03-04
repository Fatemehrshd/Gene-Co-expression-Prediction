import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from itertools import combinations
import os
from sklearn.model_selection import train_test_split

random.seed(42)
random_state = 42

# Read the correlation matrix

corr_mat = "../correlation_matrix.csv"

corr_mat = pd.read_csv(corr_mat, index_col=0)

gene_names = corr_mat.index.tolist()
random.shuffle(gene_names)
print(f"---------------------Number of genes: {len(gene_names)}")

corr_mat = corr_mat.loc[gene_names, gene_names]

# Generate pairs

pairs = [(gene1, gene2, corr_mat.loc[gene1, gene2]) for gene1, gene2 in combinations(gene_names, 2)]

df = pd.DataFrame(pairs, columns=["Gene1", "Gene2", "Corr"])

print(f"------------------Number of pairs (before dropna): {len(df)}")

# Extract genomic data in the .gtf file

columns = [
    'seqname', 'source', 'feature', 'start', 'end', 'score', 
    'strand', 'frame', 'attribute'
]

gtf_file = "/share_large/lbcg/data/yeast/Saccharomyces_cerevisiae.R64-1-1.113.gtf"

df_2 = pd.read_csv(gtf_file, sep='\t', comment='#', header=None, names=columns)

# Filter rows where the feature is "gene" and create a copy
gene_df = df_2[df_2["feature"] == "gene"].copy()

# Extract the gene_id from the attribute column
gene_df["gene_id"] = gene_df["attribute"].str.extract('gene_id "([^"]+)"')

# Calculate the TSS
gene_df["TSS"] = gene_df.apply(lambda gene: gene['start'] if gene['strand'] == '+' else gene['end'], axis=1)

# Extract sequences from the .fa file

genome = Fasta('/share_large/lbcg/data/yeast/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa')

def extract_sequence(row, genome, window=1000):
    chr_name = row['seqname']
    tss = row['TSS']
    start = max(0, tss - window)
    end = tss + window
    seq = genome[chr_name][start:end].seq
    return seq

gene_df['sequence'] = gene_df.apply(lambda row: extract_sequence(row, genome), axis=1)

# Merge sequences with pairs

def merge_gene_sequence(df, gene_df):
    df = pd.merge(df, gene_df[["gene_id", "sequence"]].rename(columns={"gene_id": "Gene1", "sequence": "Seq1"}), on="Gene1", how="left")
    df = pd.merge(df, gene_df[["gene_id", "sequence"]].rename(columns={"gene_id": "Gene2", "sequence": "Seq2"}), on="Gene2", how="left")
    return df

df = merge_gene_sequence(df, gene_df)

df = df.dropna(subset=['Seq1', 'Seq2'])

print(f"------------------Number of train pairs (after dropna): {len(df)}")

# Split the data based on their values

bins = [0.0, 0.3, 0.5, 1.0]
labels = ["0.0-0.3", "0.3-0.5", "0.5-1.0"]

def split_data(df):
    df["Corr_Bins"] = pd.cut(df["Corr"].abs(), bins=bins, labels=labels)
    bin_counts = df["Corr_Bins"].value_counts().sort_index()

    print("\nNumber of rows in each bin:")
    print(bin_counts)

    x = bin_counts.min()
    print("Number of samples to take from each bin:", x)

    # Sample x samples from each bin
    balanced_df = pd.concat([
        df[df["Corr_Bins"] == labels[0]].sample(n=x, random_state=random_state),
        df[df["Corr_Bins"] == labels[1]].sample(n=x, random_state=random_state),
        df[df["Corr_Bins"] == labels[2]].sample(n=x, random_state=random_state)
    ])

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

df = split_data(df)

# Encode sequences and generate numerical vectors

nucleotide_to_decimal = {
    'A': 1,
    'T': 2,
    'C': 3,
    'G': 4,
}

def decimal_encode_sequence(sequence):
    encoded = np.array([nucleotide_to_decimal[base] for base in sequence])
    encoded = ','.join(map(str, encoded))
    return encoded

df.loc[:, "Seq1"] = df["Seq1"].apply(decimal_encode_sequence)
df.loc[:, "Seq2"] = df["Seq2"].apply(decimal_encode_sequence)

df = df[["Seq1", "Seq2", "Corr"]]


print("----------Train Set Statistics:\n", df["Corr"].describe())
plt.hist(df["Corr"], bins=30, alpha=0.5, label='Train', color='blue')
plt.legend()
plt.xlabel("Target Value (y)")
plt.ylabel("Frequency")
plt.title("Distribution of Target Variable (y)")
plt.savefig("describe.png")
plt.close()

train, test = train_test_split(df, test_size=0.2, random_state=random_state)

os.mkdir("data")

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)