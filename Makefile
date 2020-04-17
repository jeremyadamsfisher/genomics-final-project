default: train

init:
	pyenv local 3.7.6
	poetry install
	poetry run nb-clean configure-git

dataclean:
	dvc run \
		-d data/raw/fly_shim_subset/QuickGO-annotations-1586973806005-20200415.tsv \
		-o data/intermediary/drosophila_protein_ontology_and_seqs.csv \
		poetry run python pipeline/clean_data.py

train:
	poetry run python pipeline/train.py