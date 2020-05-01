default: train

init:
	pyenv local 3.7.6
	poetry install
	poetry run nb-clean configure-git

dataclean:
	dvc run \
		-d poetry.lock \
		-d ./data/raw/fly_shim/QuickGO-annotations-1587408787815-20200420.tsv \
		-d ./data/raw/fly_shim/uniprot-yourlist%3AM20200422A94466D2655679D1FD8953E075198DA86FF07ED.fasta \
		-o ./data/intermediary/drosophila_full_protein_ontology_and_seqs.csv \
		-f pipeline/clean_data.dvc \
		poetry run python pipeline/clean_data.py

train:
	dvc run \
		-d pipeline/train.py \
		-d conda.yaml \
		-d ./data/intermediary/drosophila_full_protein_ontology_and_seqs.csv \
		-o data/model_artifacts/drosophila_subset.csv.gz \
		-o data/model_artifacts/running_metrics.json \
		-f pipeline/train.dvc \
		python pipeline/train.py
