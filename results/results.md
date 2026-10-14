# Results

### Full test set (5,621 articles)

| Model | n | Accuracy | F1 (fake) | Recall (fake) | FPR | Latency (ms/article) | Cost / 1k articles |
|---|---:|---:|---:|---:|---:|---:|---:|
| DistilBERT (fine-tuned) | 5,621 | 99.82% | 0.9980 | 99.68% | 0.06% | 8.14 | $0.00 |
| TF-IDF + LogReg | 5,621 | 98.74% | 0.9858 | 98.05% | 0.71% | 0.83 | $0.00 |

![Confusion matrices](confusion_matrices.png)
