# Static GNN vs Existing TGNN

| Model | Clean Acc / Missed | Bigram Attack Acc / Missed | Structure Attack Acc / Missed | Combined Attack Acc / Missed |
|---|---:|---:|---:|---:|
| Static GCN strict structure-only | 82.74% / 0 missed | 82.74% / 0 missed | 78.74% / 28 missed | 59.58% / 155 missed |
| Static GAT strict structure-only | 88.42% / 0 missed | 88.63% / 0 missed | 58.74% / 144 missed | 54.32% / 167 missed |
| Existing TGNN | 89.26% / 0 missed | 88.42% / 7 missed | 84.42% / 0 missed | 82.32% / 2 missed |