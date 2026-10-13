.PHONY: install data audit baseline transformer llm report test app all

install:
	pip install -e ".[transformer,llm,app,dev]"

data:
	python -m fakenews.data

audit:
	python -m fakenews.audit

baseline:
	python -m fakenews.models.baseline

transformer:
	python -m fakenews.models.transformer

# Example: make llm MODEL=<model-name> PRICE_IN=0.15 PRICE_OUT=0.60 SHOTS=0
llm:
	python -m fakenews.models.llm --model $(MODEL) --shots $(or $(SHOTS),0) --price-in $(PRICE_IN) --price-out $(PRICE_OUT)

report:
	python -m fakenews.report

test:
	pytest -q

app:
	python app/app.py

all: data audit baseline report
