"""MAD-356: the pipeline was training on its own eval holdout.

Two bugs, both silent:

1. The purpose router had no "eval" arm and ended in `.get(purpose, training_f)`, so a
   source declaring purpose: "eval" was routed straight into training.jsonl. The declared
   holdout was trained on and nothing said so.

2. `_split_train_eval` used a bare `random.shuffle(lines)` -- unseeded. The train/eval
   split was different on every run, so the 8 MAD-160 cells were not even being scored on
   the same eval set. MAD-160's success criterion IS a cross-cell eval comparison, so the
   experiment was not well-posed.
"""
import json
import random

import pytest

from pipeline.executors.dataset_prep import _split_train_eval


def _write(p, n):
    p.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": str(i)}]})
                           for i in range(n)) + "\n")


def test_split_is_deterministic_across_runs(tmp_path):
    """Same seed -> same split. The 8 cells must be scored on the SAME eval set."""
    src = tmp_path / "training.jsonl"
    _write(src, 100)

    outs = []
    for _ in range(2):
        tr, ev = tmp_path / "t.jsonl", tmp_path / "e.jsonl"
        _split_train_eval(src, tr, ev, 0.9, seed=160)
        outs.append((tr.read_text(), ev.read_text()))

    assert outs[0] == outs[1], "split is not reproducible; cells get different eval sets"


def test_split_differs_by_seed(tmp_path):
    """And the seed is actually load-bearing, i.e. it isn't accidentally ignored."""
    src = tmp_path / "training.jsonl"
    _write(src, 100)
    tr1, ev1 = tmp_path / "t1.jsonl", tmp_path / "e1.jsonl"
    tr2, ev2 = tmp_path / "t2.jsonl", tmp_path / "e2.jsonl"
    _split_train_eval(src, tr1, ev1, 0.9, seed=160)
    _split_train_eval(src, tr2, ev2, 0.9, seed=161)
    assert ev1.read_text() != ev2.read_text()


def test_split_does_not_disturb_global_rng(tmp_path):
    """It uses random.Random(seed), not random.seed(), so it must not perturb the global
    stream -- otherwise seeding this quietly reseeds everything else in the process."""
    src = tmp_path / "training.jsonl"
    _write(src, 50)
    random.seed(1234)
    before = [random.random() for _ in range(3)]
    random.seed(1234)
    _split_train_eval(src, tmp_path / "t.jsonl", tmp_path / "e.jsonl", 0.9, seed=999)
    after = [random.random() for _ in range(3)]
    assert before == after


def test_train_and_eval_are_disjoint(tmp_path):
    """The fallback split is in-distribution, but it must at least not OVERLAP."""
    src = tmp_path / "training.jsonl"
    _write(src, 100)
    tr, ev = tmp_path / "t.jsonl", tmp_path / "e.jsonl"
    _split_train_eval(src, tr, ev, 0.9, seed=160)
    train = set(tr.read_text().splitlines())
    evalset = set(ev.read_text().splitlines())
    assert train and evalset
    assert not (train & evalset), "eval rows also appear in train -- holdout is contaminated"
    assert len(train) + len(evalset) == 100


@pytest.mark.parametrize("purpose", ["training", "eval", "context", "calibration"])
def test_known_purposes_are_routed(purpose):
    """All four must have a destination. 'eval' had none, which is why it fell through
    to the training file."""
    import inspect

    from pipeline.executors import dataset_prep

    src = inspect.getsource(dataset_prep.DatasetPrepExecutor.run)
    # Strip comments -- the fix's own comment quotes the buggy expression, and matching
    # that would make this test pass/fail on prose rather than on code.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())

    assert f'"{purpose}": ' in code, f"purpose {purpose!r} has no destination"
    assert ".get(purpose, training_f)" not in code, (
        "the silent default is back: an unknown purpose will be routed into the "
        "training set instead of raising"
    )
    assert "raise ValueError" in code, "an unknown purpose must fail loudly, not default"
