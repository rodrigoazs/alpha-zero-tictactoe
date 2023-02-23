.PHONY: all tests clean

install-dev:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type pre-push
	make build-dev

build-dev:
	pip install --no-cache-dir -U pip pipenv
	pipenv install --dev

tests:
	pipenv run python -m pytest -v tests

format:
	pipenv run isort .
	pipenv run black .

check:
	pipenv run isort . -c
	pipenv run black . --check

run:
	pipenv run python main.py