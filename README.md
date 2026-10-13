# Fake News Detection: Do You Really Need an LLM?

Benchmark of three ways to detect fake news on the [ISOT Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), comparing **quality, latency and cost** on the same test articles:

| Approach | What it is | Runs on |
|---|---|---|
| **TF-IDF + logistic regression** | Classical baseline, word 1–2 grams | CPU, milliseconds |
| **DistilBERT, fine-tuned** | Small transformer trained on the task | GPU to train, CPU to serve |
| **LLM, zero-shot / few-shot** | General-purpose model prompted through an API | Paid API |

The question behind it is the one teams ask before shipping: *is a large model worth its cost here, or does a cheaper model do the job?*

## The dataset is leaky, so the first step is fixing it

Most public notebooks on this dataset report ~99% accuracy. That number mostly measures **where the articles were scraped**, not whether they are fake:

- almost every real article starts with a Reuters dateline (`WASHINGTON (Reuters) - …`);
- fake articles carry their own source marks (`Featured image via …`, `(VIDEO)` in titles, `21st Century Wire`);
- the `subject` column alone predicts the label perfectly, because its categories never overlap between classes;
- thousands of articles are duplicated, so the same text can land in train and test.

[`src/fakenews/data.py`](src/fakenews/data.py) strips these markers, drops `subject`, removes empty and duplicated articles, then splits the data (stratified random, or temporal with `--strategy temporal`). [`src/fakenews/audit.py`](src/fakenews/audit.py) quantifies the effect by training the same baseline on raw and cleaned text.

<!-- RESULTS:AUDIT -->
*Leakage audit results: run `make audit` (see [results/leakage_audit.md](results/leakage_audit.md)).*
<!-- /RESULTS:AUDIT -->

## Results

<!-- RESULTS:MAIN -->
*Benchmark results: run the pipeline below (see [results/results.md](results/results.md)).*
<!-- /RESULTS:MAIN -->

All models are scored on the same cleaned test set. The LLM is evaluated on a stratified subsample to keep API costs low, and every other model is re-scored on exactly those articles for the head-to-head table. "Fake" is the positive class; FPR is the share of real articles wrongly flagged as fake.

## Demo

`app/app.py` is a Gradio app: paste a headline and an article, and every trained model gives its verdict and confidence, with the words that weighed most in the TF-IDF model's decision.

```bash
python app/app.py          # LLM column appears if OPENAI_API_KEY and LLM_MODEL are set
```

## Reproduce

```bash
git clone https://github.com/aminecharfeddine/fake-news-llm-benchmark.git
cd fake-news-llm-benchmark
pip install -e ".[transformer,llm,app,dev]"

# 1. Data (needs a Kaggle API token)
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
make data          # clean + split  -> data/processed/
make audit         # leakage audit  -> results/leakage_audit.md

# 2. Models
make baseline      # TF-IDF + LogReg, C tuned on validation
make transformer   # DistilBERT; use a GPU (see notebooks/train_transformer_colab.ipynb)
make llm MODEL=<openai-model> PRICE_IN=<$ per 1M input tokens> PRICE_OUT=<$ per 1M output tokens> SHOTS=0

# 3. Compare
make report        # tables + figures -> results/
make test          # unit tests
```

Prices are passed explicitly because they change: check the provider's pricing page. LLM answers are cached in `results/llm_cache/`, so re-running never pays twice for the same article.

## Project structure

```
src/fakenews/
  data.py              loading, leakage cleaning, deduplication, splits
  audit.py             raw vs cleaned comparison, subject-only leak
  evaluation.py        shared metrics and prediction files
  report.py            comparison tables and figures
  models/
    baseline.py        TF-IDF + logistic regression, feature importance, per-article explanation
    transformer.py     DistilBERT fine-tuning (plain PyTorch loop) and inference
    llm.py             zero/few-shot LLM classifier with cost, latency and caching
app/app.py             Gradio demo
notebooks/             GPU training on Colab
tests/                 unit tests on a synthetic dataset that reproduces the leaks
```

## Limitations

- The articles are US political news from 2016–2017. A model trained here learns the style of that period's fake news sites; it is not a general fact-checker.
- Labels are assigned by source (Reuters vs sites flagged by fact-checkers), not article by article. Some "fake" articles are opinion pieces rather than fabricated stories.
- The LLM is evaluated on a subsample, so its metrics have wider confidence intervals than the other models'.

## License

MIT
