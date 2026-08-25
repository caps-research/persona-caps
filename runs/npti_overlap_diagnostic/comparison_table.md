# PB-overlap comparison

Pairing: same-trait, polarities pooled (PB has no per-Q polarity).
PB n = 459 questions over 5 traits.

| Source | cosine top-1 mean | cosine top-5 mean | trigram max mean | trigram mean mean |
|---|---|---|---|---|
| IPIP-NEO-300 | 0.364 | 0.314 | 0.000 | 0.000 |
| P^2 paragraphs | 0.358 | 0.337 | 0.005 | 0.000 |
| CAPS atoms | 0.391 | 0.335 | 0.001 | 0.000 |
| **P^2 / CAPS ratio** | 0.915x | 1.007x | 7.711x | 5.646x |

## Per-trait top-1 cosine mean (rows = source, cols = trait)

| Source | E | A | C | N | O |
|---|---|---|---|---|---|
| IPIP-NEO-300 | 0.405 | 0.330 | 0.393 | 0.355 | 0.340 |
| P^2 paragraphs | 0.444 | 0.301 | 0.375 | 0.331 | 0.341 |
| CAPS atoms | 0.474 | 0.378 | 0.418 | 0.371 | 0.317 |