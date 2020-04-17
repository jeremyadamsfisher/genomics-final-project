import pandas as pd
import tensorflow as tf

ontology_and_seq_fp = "data/intermediary/drosophila_protein_ontology_and_seqs.csv"

# preprocess data so its more vector-y for tensorflow ingest

df = pd.read_csv(ontology_and_seq_fp)
relevant_subset = df[df.qualifier.isin(["enables", "involved_in"])].dropna()
interesting_go_names = [name for (name, freq) in relevant_subset.go_name.value_counts().to_dict().items() if 1 < freq]  # <- probably need to change the filter step !!
df = df[df.go_name.isin(interesting_go_names)]
one_row_per_gene = pd.DataFrame(index=df.seq.unique(), columns=interesting_go_names).fillna(0)
for _, row in df.iterrows():
    one_row_per_gene.loc[row.seq, row.go_name] = 1
one_row_per_gene = one_row_per_gene.reset_index().rename(columns={"index": "seq"})
breakpoint()
dataset = tf.data.Dataset.from_tensor_slices(
    [one_hot[col] for col in one_row_per_gene.columns]
)