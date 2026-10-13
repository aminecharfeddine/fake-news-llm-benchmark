# Leakage audit

| Setting | Articles | Test accuracy | F1 (fake) | FPR |
|---|---:|---:|---:|---:|
| `subject` column only (no text) | 44,898 | 100.00% | - | - |
| TF-IDF + LogReg, raw text | 37,554 | 99.06% | 0.9894 | 0.51% |
| TF-IDF + LogReg, cleaned text | 37,467 | 98.51% | 0.9831 | 0.67% |
| TF-IDF + LogReg, cleaned text + weekday names masked | 37,467 | 98.13% | 0.9789 | 0.96% |
| TF-IDF + LogReg, cleaned text, temporal split (test 2017-11-04 to 2018-02-19) | 37,467 | 98.22% | 0.9283 | 1.73% |

Random splits are stratified (44.6% fake in every split). The temporal split trains on the oldest 70% of articles and tests on the most recent 15%, where only 661 of 5,621 test articles are fake: the class balance shifts over time, which is part of what makes it harder.

**Strongest "real" n-grams, raw text:** `reuters`, `said`, `washington reuters`, `said on`, `on wednesday`, `washington`, `reuters the`, `on tuesday`, `said in`, `president donald`

**Strongest "real" n-grams, cleaned text:** `said`, `said on`, `on wednesday`, `president donald`, `on tuesday`, `said in`, `on thursday`, `on friday`, `on monday`, `said the`

**Strongest "fake" n-grams, raw text:** `via`, `video`, `read more`, `image`, `featured`, `featured image`, `just`, `this`, `read`, `us`

**Strongest "fake" n-grams, cleaned text:** `just`, `this`, `gop`, `us`, `president trump`, `the us`, `watch`, `obama`, `you`, `here`
