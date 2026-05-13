# Static GNN vs Existing TGNN

| Model | Clean Acc / Missed | Bigram Attack Acc / Missed | Structure Attack Acc / Missed | Combined Attack Acc / Missed |
|---|---:|---:|---:|---:|
| Static GCN strict structure-only | 82.74% / 0 missed | 82.74% / 0 missed | 77.89% / 32 missed | 59.58% / 152 missed |
| Static GAT strict structure-only | 88.42% / 0 missed | 88.63% / 0 missed | 58.53% / 144 missed | 54.11% / 168 missed |
| Existing TGNN | 89.47% / 0 missed | 89.05% / 0 missed | 89.26% / 0 missed | 88.63% / 0 missed |