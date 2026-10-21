# deploy-safety-gate

**Does a deployment change make a model less safe, and can an automatic check tell?**

Companies constantly change *how* a model is served: compression, a different serving program, batching, speculative decoding, new defaults. They check speed and accuracy; safety is rarely re-checked, and setup mistakes (a wrong chat template, a doubled start marker, a dropped system prompt) go unnoticed because nothing crashes.

This repo is a small, honest gate for that:

1. **Run** the same safety questions on two setups (a baseline and a candidate).
2. **Judge** every answer (refused or not, and how useful the answer would be to someone with bad intent).
3. **Compare** them question by question, with error bars, and return **PASS / FAIL / INCONCLUSIVE**.
4. **Check the configuration** for known setup mistakes before running anything.

## Setup (Mac, ~5 minutes)

Needs [Ollama](https://ollama.com) running and Python 3.10+.

```bash
git clone <this repo> && cd deploy-safety-gate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # runs against a fake Ollama: no model needed

ollama list                    # see which models you have
dsg fetch-data                 # downloads XSTest (450 questions) + StrongREJECT (313)
```

## First run (the smoke test)

```bash
dsg check-config gemma4:e4b --save-reference refs/gemma4-e4b.json   # setup-mistake check
dsg run configs/smoke.yaml                                          # 100 questions, ~10-20 min
dsg judge <run-id> --judge-model qwen3.5:9b                         # grade the answers
dsg summarize <run-id>                                              # rates, error bars, speed
dsg label <run-id> -n 30                                            # hand-check 30: is the judge right?
```

Then plant a setup mistake and see if the gate catches it:

```bash
dsg run configs/mistake_no_template.yaml
dsg judge <that-run-id>
dsg compare <smoke-run-id> <that-run-id> --margin 0.05
```

`<run-id>` is printed by `dsg run` (folders under `outputs/`).

## What each command does

| Command | What it does |
| --- | --- |
| `dsg fetch-data` | Downloads the question sets into `data/` |
| `dsg run CONFIG` | Asks every selected question (with repeats), saves each answer with the exact request, settings and speed. Rerun with `--run-id` to resume. |
| `dsg judge RUN` | Grades each answer with a judge model, StrongREJECT-style: `refused`, plus a 0-1 `harm_score` that catches "refuse, then help anyway". |
| `dsg summarize RUN` | Rates with 95% intervals, how much answers flip across repeats, noise from question choice vs model randomness, speed. |
| `dsg compare BASE CAND` | The gate. Exit code 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 UNMEASURED, so it can block a CI job. `--json` writes a report. |
| `dsg check-config MODEL` | Lists the template, default settings, system prompt and compression level Ollama uses, warns about known mistakes, and diffs against a saved reference. |
| `dsg label RUN` / `dsg agreement RUN` | Hand-label a sample; see how often the judge agrees with you. |

## Metrics

| Metric | Questions | Bad direction |
| --- | --- | --- |
| `refusal_rate` | harmful (StrongREJECT + XSTest "unsafe") | goes down |
| `harm_score` | harmful | goes up |
| `over_refusal` | harmless (XSTest "safe") | goes up |

## How the gate decides

- **The question is the unit of evidence.** Repeats of one question are averaged first, so 10 repeats never count as 10 independent data points.
- **Paired.** Both setups answer the same questions (same `data.seed`), and only the per-question difference matters. That removes most of the noise from which questions were picked.
- **Bootstrap intervals over questions**, at a one-sided level of `alpha / number_of_metrics` per bound (a Bonferroni correction for checking several metrics at once).
- **Three outcomes**, where "worse" is the change in the bad direction:
  - **PASS**: the upper bound of "worse" is below `--margin`, so any regression is smaller than what you care about (an equivalence / non-inferiority test, not just "no significant difference").
  - **FAIL**: the lower bound is above 0, so it's confidently worse.
  - **INCONCLUSIVE**: not enough evidence either way; add questions or repeats.
- With one answer per question, it also reports an exact **McNemar** test on the questions whose answer flipped.

## Configs

| File | Purpose |
| --- | --- |
| `configs/smoke.yaml` | Baseline: chat template on, temperature 0, 100 questions |
| `configs/mistake_no_template.yaml` | Planted mistake: no chat template (Ollama raw mode) |
| `configs/mistake_wrong_template.yaml` | Planted mistake: another model family's template |
| `configs/noise.yaml` | Temperature 0.7, 5 repeats: measures how much scores wobble |
| `configs/vllm_example.yaml` | For later: vLLM or llama.cpp's `llama-server` (OpenAI-compatible API) |

Change `model:` to test another model. Keep `data.seed` the same across runs you want to compare.

## Handling harmful content

`outputs/` contains model answers to harmful questions and is in `.gitignore`. Publish scores and labels, not raw answers.

## Limits (be honest about these in the write-up)

- The judge is a model and makes mistakes. Measure it with `dsg label` and report its agreement rate.
- Bootstrap intervals are rough below ~30 paired questions per metric.
- A PASS means "no regression larger than the margin *on these questions*", not "safe".
- The config check reads what Ollama reports. It can't yet see the exact tokens sent, or compare against the model's official Hugging Face template (planned).

## Roadmap

- [ ] Smoke run end to end on the Mac
- [ ] Measure judge accuracy (`dsg label`, 100+ answers)
- [ ] Noise study: how many questions and repeats to detect a 3-point drop
- [ ] Config check against the official Hugging Face chat template; count start markers in real token ids
- [ ] vLLM vs llama.cpp on a rented GPU; 4-bit vs 16-bit
- [ ] Validate the gate: planted mistakes measured on a large run, then gate accuracy on random subsets
- [ ] GitHub Action running `pytest` and a tiny `dsg compare` on every push
