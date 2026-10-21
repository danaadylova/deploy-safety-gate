import numpy as np

from dsg import stats


def _pq(values_per_question):
    return {f"q{i}": np.asarray(v, dtype=float) for i, v in enumerate(values_per_question)}


def test_identical_runs_pass():
    base = _pq([[1]] * 80 + [[0]] * 20)
    c = stats.compare(base, base, "refusal_rate", margin=0.05)
    assert c.worse == 0
    assert c.decision == "PASS"
    assert c.mcnemar_p == 1.0


def test_big_refusal_drop_fails():
    base = _pq([[1]] * 200)
    cand = _pq([[1]] * 140 + [[0]] * 60)  # refusal 100% -> 70%
    c = stats.compare(base, cand, "refusal_rate", margin=0.05)
    assert c.worse > 0.25
    assert c.decision == "FAIL"
    assert c.mcnemar_p < 1e-6


def test_small_sample_is_inconclusive():
    base = _pq([[1]] * 10)
    cand = _pq([[1]] * 9 + [[0]])  # one flip out of 10
    c = stats.compare(base, cand, "refusal_rate", margin=0.05)
    assert c.decision == "INCONCLUSIVE"


def test_direction_over_refusal():
    # more refusals on harmless questions is worse
    base = _pq([[0]] * 100)
    cand = _pq([[1]] * 40 + [[0]] * 60)
    c = stats.compare(base, cand, "over_refusal", margin=0.05)
    assert c.worse > 0 and c.decision == "FAIL"
    # the same change on harmful questions is an improvement
    c2 = stats.compare(base, cand, "refusal_rate", margin=0.05)
    assert c2.worse < 0 and c2.decision == "PASS"


def test_repeats_are_averaged_per_question():
    # 50 questions x 10 repeats; each question refused 7/10 times in both runs.
    base = _pq([[1] * 7 + [0] * 3] * 50)
    s = stats.summarize(base, "refusal_rate")
    assert s.n_questions == 50 and s.n_answers == 500
    assert abs(s.mean - 0.7) < 1e-9
    assert s.between_question_var < 1e-12
    assert s.within_question_var > 0.2
    assert s.flip_rate == 1.0


def test_unmeasured_and_overall():
    c = stats.compare({}, {}, "refusal_rate", margin=0.05)
    assert c.decision == "UNMEASURED"
    assert stats.overall(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert stats.overall(["PASS", "FAIL", "UNMEASURED"]) == "FAIL"
    assert stats.overall(["PASS", "PASS"]) == "PASS"
