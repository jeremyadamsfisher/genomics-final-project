"""merge gene ontology and sequence data"""

from Bio import SeqIO
import pandas as pd

ontology_fp = "./data/raw/fly_shim/QuickGO-annotations-1587408787815-20200420.tsv"
seq_fp = "./data/raw/fly_shim/uniprot-yourlist%3AM20200422A94466D2655679D1FD8953E075198DA86FF07ED.fasta"
ontology_with_seqs_fp = "./data/intermediary/drosophila_full_protein_ontology_and_seqs.csv"

df = pd.read_csv(ontology_fp, delimiter="\t")
df.columns = df.columns.map(lambda s: s.replace(" ", "_").lower())
df = df[["gene_product_id", "symbol", "qualifier", "go_name"]]
genes = df["gene_product_id"].unique()

records = SeqIO.parse(seq_fp, "fasta")
gene2seq = {}
for record in records:
    _, gene, _ = record.name.split("|")
    gene2seq[gene] = str(record.seq)

df["seq"] = df.gene_product_id.map(gene2seq)
df.dropna().to_csv(ontology_with_seqs_fp, index=False)