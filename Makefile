default: train

init:
	pyenv local 3.7.6
	poetry install
	poetry run nb-clean configure-git

dataclean:
	$(PY) pipeline/clean_data.py

train:
	$(PY) pipeline/train.py