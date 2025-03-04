import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from itertools import combinations

random.seed(42)
random_state = 42

# Read the correlation matrix

corr_mat = "../correlation_matrix.csv"

corr_mat = pd.read_csv(corr_mat, index_col=0)

gene_names = corr_mat.index.tolist()
random.shuffle(gene_names)
print(f"---------------------Number of genes: {len(gene_names)}")

# Split train and test genes

genes_train = gene_names[:4000]
genes_test  = gene_names[4000:]

print(f"---------------------Number of train genes: {len(genes_train)}")
print(f"---------------------Number of test genes:  {len(genes_test)}")

corr_train = corr_mat.loc[genes_train, genes_train]
corr_test  = corr_mat.loc[genes_test, genes_test]

# Generate pairs for train and test sets

train = [(gene1, gene2, corr_train.loc[gene1, gene2]) for gene1, gene2 in combinations(genes_train, 2)]
test  = [(gene1, gene2, corr_test.loc[gene1, gene2]) for gene1, gene2 in combinations(genes_test, 2)]

train_df = pd.DataFrame(train, columns=["Gene1", "Gene2", "Corr"])
test_df  = pd.DataFrame(test, columns=["Gene1", "Gene2", "Corr"])

print(f"------------------Number of train pairs (before dropna): {len(train_df)}")
print(f"------------------Number of test pairs  (before dropna): {len(test_df)}")

# Extract genomic data in the .gtf file

columns = [
    'seqname', 'source', 'feature', 'start', 'end', 'score', 
    'strand', 'frame', 'attribute'
]

gtf_file = "/share_large/lbcg/data/yeast/Saccharomyces_cerevisiae.R64-1-1.113.gtf"

df = pd.read_csv(gtf_file, sep='\t', comment='#', header=None, names=columns)

gene_df = df[df["feature"] == "gene"]

gene_df["gene_id"] = gene_df["attribute"].str.extract('gene_id "([^"]+)"')
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

# Merge the sequences and the pairs

def merge_gene_sequence(df, gene_df):
    df = pd.merge(df, gene_df[["gene_id", "sequence"]].rename(columns={"gene_id": "Gene1", "sequence": "Seq1"}), on="Gene1", how="left")
    df = pd.merge(df, gene_df[["gene_id", "sequence"]].rename(columns={"gene_id": "Gene2", "sequence": "Seq2"}), on="Gene2", how="left")
    return df

train_df = merge_gene_sequence(train_df, gene_df)
test_df  = merge_gene_sequence(test_df, gene_df)

train_df = train_df.dropna(subset=['Seq1', 'Seq2'])
test_df  = test_df.dropna(subset=['Seq1', 'Seq2'])

print(f"------------------Number of train pairs (after dropna): {len(train_df)}")
print(f"------------------Number of test pairs  (after dropna): {len(test_df)}")

# ------------------------ Split data ------------------------

bins = [0.0, 0.3, 0.5, 1.0]
labels = ["0.0-0.3", "0.3-0.5", "0.5-1.0"]

def split_data(df):
    df["Corr_Bins"] = pd.cut(df["Corr"].abs(), bins=bins, labels=labels)
    bin_counts = df["Corr_Bins"].value_counts().sort_index()

    print("\nNumber of rows in each bin:")
    print(bin_counts)

    x = bin_counts.min()
    print("Number of samples to take from each bin:", x)

    balanced_df = pd.concat([
        df[df["Corr_Bins"] == labels[0]].sample(n=x, random_state=random_state),
        df[df["Corr_Bins"] == labels[1]].sample(n=x, random_state=random_state),
        df[df["Corr_Bins"] == labels[2]].sample(n=x, random_state=random_state)
    ])

    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

print("\nSplit TRAINING set:")
train_df = split_data(train_df)
print("------------------------")
print("\nSplit TEST set:")
test_df = split_data(test_df)


# -------------------------- Embedding --------------------------

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

train_df.loc[:, "Seq1"] = train_df["Seq1"].apply(decimal_encode_sequence)
train_df.loc[:, "Seq2"] = train_df["Seq2"].apply(decimal_encode_sequence)

test_df.loc[:, "Seq1"] = test_df["Seq1"].apply(decimal_encode_sequence)
test_df.loc[:, "Seq2"] = test_df["Seq2"].apply(decimal_encode_sequence)

# ------------------------- End Embedding ---------------------------

# -------------------------------------------------------------------

train_df = train_df[["Seq1", "Seq2", "Corr"]]
test_df  = test_df[["Seq1", "Seq2", "Corr"]]


print("----------Train Set Statistics:\n", train_df["Corr"].describe())
print("----------Test Set Statistics:\n", test_df["Corr"].describe())

plt.hist(train_df["Corr"], bins=30, alpha=0.5, label='Train', color='blue')
plt.hist(test_df["Corr"], bins=30, alpha=0.5, label='Test', color='green')
plt.legend()
plt.xlabel("Target Value (y)")
plt.ylabel("Frequency")
plt.title("Distribution of Target Variable (y)")
plt.savefig("describe.png")


os.mkdir("data")

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)