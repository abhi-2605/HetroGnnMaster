# Sequential BiLSTM vs Existing TGNN

| Model | Clean Acc / Missed | Unigram Attack Acc / Missed | Random Rename Attack Acc / Missed | Reorder Attack Acc / Missed | Bigram Attack Acc / Missed | Combined All Attack Acc / Missed |
|---|---:|---:|---:|---:|---:|---:|
| Sequential BiLSTM | 89.23% / 4 missed | 22.77% / 0 missed | 15.08% / 0 missed | 82.46% / 5 missed | 41.23% / 1 missed | 22.46% / 0 missed |
| Existing TGNN | 85.54% / 0 missed | 84.31% / 0 missed | 84.62% / 0 missed | 85.23% / 0 missed | 84.62% / 0 missed | 80.31% / 0 missed |