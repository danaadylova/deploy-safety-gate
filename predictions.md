# Predictions (write before running the main experiments)

Fill this in and commit it **before** the Nov 14–15 main runs. The commit date is the proof.
In the write-up, compare each prediction with the result, including the ones that were wrong.

**Date written:**
**Models:**
**Question sets and sizes:**
**Margin (smallest drop I care about):**

| Change | What I expect (direction and size) | What would surprise me | Result |
| --- | --- | --- | --- |
| 4-bit vs 16-bit | | | |
| vLLM vs llama.cpp (defaults matched) | | | |
| No chat template | | | |
| Wrong chat template | | | |
| Missing system prompt | | | |
| Temperature 0 → 0.7 | | | |
| Speculative decoding on | | | |

## Setup definitions (decided before the main runs)
- **Wrong chat template** = a copied ChatML setup: the ChatML wrapper **and** its stop token
  `<|im_end|>`, which replaces the model's own stop list. Two differences from the baseline
  (input, generation settings), kept together on purpose because they are copied together in
  practice. Decided 2026-11-04.

## Noise
- How many questions do I expect I need to detect a 3-point drop?
- How much will answers flip between repeats at temperature 0.7?

## Judge
- How often do I expect the judge to disagree with my hand labels?
