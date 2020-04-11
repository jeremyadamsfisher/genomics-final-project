#!/usr/bin/env python3

"""from the gene ontology ascension numbers, look up protein
sequence on Entrez"""

import time
import pandas as pd
from Bio import Entrez, SeqIO
from tqdm import tqdm
Entrez.email = "jeremyf@cmu.edu"

ontology_fp = "data/raw/zebrafish_protein_ontology.tsv"
ontology_with_seqs_fp = "data/intermediary/zebrafish_protein_ontology_and_seqs.tsv"

df = pd.read_csv(ontology_fp, delimiter="\t")
df.columns = df.columns.map(lambda s: s.replace(" ", "_").lower())
df = df[["gene_product_id", "symbol", "qualifier", "go_name"]]
gene2seq = {}
genes = df["gene_product_id"].unique()
for gene in tqdm(genes, unit="genes"):
    try:
        with Entrez.efetch(db="protein", id=gene, rettype="fasta", retmax=1) as query:
            record = next(SeqIO.parse(query, "fasta"))
            gene2seq[gene] = str(record.seq)
    except:  # HTTPError...shh -- I know this is terrible practice 🤫
        gene2seq[gene] = None
    finally:
        time.sleep(1)
df["seq"] = df["gene_product_id"].map(gene2seq)
df.dropna().to_csv(ontology_with_seqs_fp)