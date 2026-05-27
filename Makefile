.PHONY: install shell notebook test clean lint format

install:
	pip install -r requirements.txt

shell:
	nix develop

notebook:
	jupyter lab

test:
	pytest tests/

lint:
	ruff .

format:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete