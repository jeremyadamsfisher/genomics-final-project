PO=poetry run
PY=poetry run python

default: train

init:
	pyenv local 3.7.6
	poetry install
	$(PO) nb-clean configure-git

dataclean:
	$(PY) pipeline/clean_data.py

train:
	$(PY) pipeline/train.py