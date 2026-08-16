# A — Baseline (fixed-size, dense-only, top_k=5)

- Config id: `a_baseline_fixed_dense`
- Chunking: `fixed_size`
- Retrieval: `dense`
- top_k: 5
- Questions: 19 (10 flagged **no ground truth answer yet**)
- Context recall scored on 6 items with a real `expected_answer`

## Overall

| faithfulness | context precision | context recall | answer relevancy |
| --- | --- | --- | --- |
| 0.865 | 0.451 | 0.518 | 0.800 |

## By category

| category | n | n with GT | faithfulness | context precision | context recall | answer relevancy |
| --- | --- | --- | --- | --- | --- | --- |
| factual_lookup | 10 | 0 | 0.900 | 0.233 | — | 0.790 |
| definitional | 9 | 9 | 0.816 | 0.815 | 0.518 | 0.817 |
| multi_document_synthesis | 0 | 0 | — | — | — | — |
| adversarial_out_of_scope | 0 | 0 | — | — | — | — |

## Per question

| id | category | ground truth | faithfulness | ctx precision | ctx recall | answer relevancy |
| --- | --- | --- | --- | --- | --- | --- |
| fact_01 | factual_lookup | no ground truth answer yet | 0.000 | 1.000 | — | 0.515 |
| fact_02 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.863 |
| fact_03 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.726 |
| fact_04 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.910 |
| fact_05 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.700 |
| fact_06 | factual_lookup | no ground truth answer yet | 1.000 | 0.333 | — | 0.750 |
| fact_07 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.739 |
| fact_08 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.851 |
| fact_09 | factual_lookup | no ground truth answer yet | 1.000 | 1.000 | — | 0.901 |
| fact_10 | factual_lookup | no ground truth answer yet | 1.000 | 0.000 | — | 0.945 |
| def_01 | definitional | yes | 1.000 | 1.000 | 1.000 | 0.807 |
| def_02 | definitional | yes | 0.857 | 1.000 | 0.600 | 0.812 |
| def_03 | definitional | yes | 0.667 | 1.000 | 0.400 | 0.792 |
| def_04 | definitional | yes | 0.909 | 1.000 | 0.250 | 0.770 |
| def_05 | definitional | yes | 0.778 | 0.887 | 0.857 | 0.899 |
| def_06 | definitional | yes | 0.500 | 0.000 | 0.000 | 0.821 |
| def_07 | definitional | yes | 1.000 | — | — | — |
| def_08 | definitional | yes | — | — | — | — |
| def_09 | definitional | yes | — | — | — | — |
