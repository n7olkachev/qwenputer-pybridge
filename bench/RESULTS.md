# Results

Model: Qwen3.5-4B, qwen3.5:4b, qwenputer-rv via Ollama (Q4_K_M, non-thinking, greedy). Items per cell: 50. Accuracy is exact match after task-specific normalization (`bench/tasks.py`). GPU columns (`base`, `weights`) appear once bench/run_gpu.py has been run.

| Task | Base (no tool, Ollama Q4) | python tool, in-weights env (ref machine, Ollama Q4) | python tool, host CPython (Ollama Q4) | Base bf16, no tool (GPU) | Qwenputer + python in-weights (GPU) | same, responses re-tokenized (GPU) | tool calls / item | instructions / item | trace tokens / item | ≈ L4 s / item |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count_char | 68% | 100% | 100% | 78% | 100% | – | 1.02 | 131,831 | 164,048 | 33 |
| reverse | 0% | 96% | 96% | 0% | 20% | 100% | 1.00 | 63,416 | 78,772 | 16 |
| nth_char | 94% | 98% | 98% | 98% | 98% | – | 0.64 | 38,691 | 48,365 | 10 |
| swap_chars | 24% | 64% | 74% | 38% | 30% | 100% | 2.72 | 241,932 | 301,844 | 60 |
| word_sorting | 60% | 98% | 98% | 72% | 96% | 90% | 1.00 | 166,082 | 207,576 | 42 |
| dyck_languages | 26% | 24% | 28% | 48% | 6% | 10% | 1.28 | 41,398 | 52,564 | 10 |
| boolean_expressions | 86% | 92% | 96% | 94% | 92% | – | 1.28 | 91,924 | 114,629 | 23 |
| date_understanding | 52% | 62% | 64% | 84% | 54% | – | 1.24 | 212,456 | 258,250 | 53 |
| multistep_arithmetic_two | 98% | 100% | 100% | 100% | 100% | – | 0.92 | 64,345 | 79,892 | 16 |
| object_counting | 86% | 82% | 82% | 96% | 86% | – | 0.72 | 23,463 | 29,586 | 6 |

Cost columns are for the in-weights environment: the model's calls executed on the reference machine with the real firmware; seconds assume the author's executor (~4K instructions/s on an L4).
