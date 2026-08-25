## 1. Internal diversity (pairwise cosine distance within cell)

Higher = atoms cover a more diverse situation space within the cell.
Identity = 0 distance. Random unrelated sentences ≈ 0.7-0.9.

| Cell | n | mean dist | median dist | min dist | max dist |
|---|---|---|---|---|---|
| high_extraversion | 20 | 0.525 | 0.584 | 0.000 | 0.831 |
| low_extraversion | 19 | 0.624 | 0.642 | 0.019 | 0.938 |
| high_agreeableness | 12 | 0.600 | 0.640 | 0.193 | 0.869 |
| low_agreeableness | 20 | 0.605 | 0.639 | -0.000 | 0.960 |
| high_conscientiousness | 11 | 0.566 | 0.661 | -0.000 | 0.921 |
| low_conscientiousness | 20 | 0.712 | 0.720 | 0.134 | 1.023 |
| high_neuroticism | 19 | 0.634 | 0.678 | 0.064 | 1.049 |
| low_neuroticism | 7 | 0.529 | 0.593 | 0.239 | 0.719 |
| high_openness | 18 | 0.547 | 0.589 | 0.000 | 0.913 |
| low_openness | 20 | 0.687 | 0.730 | 0.137 | 0.980 |

## 2. Test-query coverage (Mini-IPIP items into target cell, threshold=0.5)

For each Mini-IPIP item targeting dim D, max cosine sim into the
(D, high) and (D, low) atom pools. The persona used at test time
matches the polarity, so coverage is the polarity-matched column.

`covered` = # items with max sim ≥ 0.5 in matched cell.

| Cell | n_items | mean max-sim | median max-sim | min max-sim | covered@thresh |
|---|---|---|---|---|---|
| high_extraversion | 4 | 0.414 | 0.399 | 0.321 | 1/4 |
| low_extraversion | 4 | 0.395 | 0.403 | 0.281 | 0/4 |
| high_agreeableness | 4 | 0.382 | 0.367 | 0.329 | 0/4 |
| low_agreeableness | 4 | 0.381 | 0.368 | 0.332 | 0/4 |
| high_conscientiousness | 4 | 0.343 | 0.354 | 0.174 | 0/4 |
| low_conscientiousness | 4 | 0.366 | 0.380 | 0.188 | 1/4 |
| high_neuroticism | 4 | 0.379 | 0.380 | 0.296 | 0/4 |
| low_neuroticism | 4 | 0.339 | 0.312 | 0.235 | 0/4 |
| high_openness | 4 | 0.230 | 0.236 | 0.166 | 0/4 |
| low_openness | 4 | 0.432 | 0.387 | 0.345 | 1/4 |

## 3. Per-item nearest-cell vs target cell

For each test query, what cell does its nearest atom actually live in?
If the test polarity is `high D`, ideally the nearest atom is in `high_D`.

| Item | Target dim | Top cell (high+low pool) | Argmax sim |
|---|---|---|---|
| e1: "I am the life of the party." | extraversion | high_extraversion | 0.321 |
| e2: "I don't talk a lot." | extraversion | high_openness | 0.473 |
| e3: "I talk to a lot of different people at parties." | extraversion | high_openness | 0.634 |
| e4: "I keep in the background." | extraversion | low_neuroticism | 0.439 |
| a1: "I sympathize with others' feelings." | agreeableness | low_openness | 0.445 |
| a2: "I am not really interested in other people's probl" | agreeableness | high_openness | 0.523 |
| a3: "I feel others' emotions." | agreeableness | low_openness | 0.519 |
| a4: "I am not really interested in others." | agreeableness | high_openness | 0.425 |
| c1: "I get chores done right away." | conscientiousness | low_extraversion | 0.562 |
| c2: "I often forget to put things back in their proper " | conscientiousness | high_agreeableness | 0.428 |
| c3: "I like order." | conscientiousness | high_openness | 0.236 |
| c4: "I make a mess of things." | conscientiousness | high_agreeableness | 0.418 |
| n1: "I have frequent mood swings." | neuroticism | low_conscientiousness | 0.568 |
| n2: "I am relaxed most of the time." | neuroticism | high_agreeableness | 0.809 |
| n3: "I get upset easily." | neuroticism | low_agreeableness | 0.365 |
| n4: "I seldom feel blue." | neuroticism | low_openness | 0.317 |
| o1: "I have a vivid imagination." | openness | low_openness | 0.607 |
| o2: "I am not interested in abstract ideas." | openness | high_extraversion | 0.397 |
| o3: "I have difficulty understanding abstract ideas." | openness | low_openness | 0.345 |
| o4: "I do not have a good imagination." | openness | low_openness | 0.415 |

## 4. Cross-cell discrimination per atom

For each atom, mean of (max sim within own cell to OTHERS) − (max sim outside own cell).
Positive = atom is uniquely closer to its own cell. Negative = atom is also similar
to atoms outside its label, suggesting noisy or ambiguous polarity.

| Cell | n | own-cell mean inner sim | best other-cell sim | margin |
|---|---|---|---|---|
| high_extraversion | 20 | 0.784 | 0.857 | -0.073 |
| low_extraversion | 19 | 0.665 | 0.750 | -0.085 |
| high_agreeableness | 12 | 0.702 | 0.769 | -0.068 |
| low_agreeableness | 20 | 0.845 | 0.861 | -0.016 |
| high_conscientiousness | 11 | 0.710 | 0.826 | -0.116 |
| low_conscientiousness | 20 | 0.647 | 0.784 | -0.137 |
| high_neuroticism | 19 | 0.771 | 0.863 | -0.092 |
| low_neuroticism | 7 | 0.671 | 0.853 | -0.182 |
| high_openness | 18 | 0.803 | 0.883 | -0.080 |
| low_openness | 20 | 0.689 | 0.780 | -0.091 |
