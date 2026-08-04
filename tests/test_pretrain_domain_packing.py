"""D2/MAD-322: the domain label must survive packing and reach the model per-token.

TRL's packing=True concatenates documents and keeps only the text column, so a
per-document domain_id cannot become per-token supervision. Without the path these tests
cover, the domain gate receives NO gradient and stays a fixed random projection for the
entire run -- with a perfectly plausible loss curve, which is why it went unnoticed.

MAD-364 is the trap that makes this delicate: domain labels are supervision ONLY and
must never enter the forward pass as an input. An in-stream domain tag was deliberately
cut from the design because under causal attention every token attends to a tag at
position 0 and Mamba's recurrent state carries it forward, so the gate would learn a
trivial lookup and then collapse at inference where no tag exists. The test that
input_ids is unaffected by the label is therefore not a nicety.
"""
import pytest

datasets = pytest.importorskip("datasets")

from pipeline.executors._pretrain_worker import _pack_with_domains  # noqa: E402


class _FakeTokenizer:
    """One token per character, id = ord(c). Deterministic and inspectable, so a test can
    say exactly which document a token came from."""

    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _ds(rows):
    return datasets.Dataset.from_list(rows)


def test_packs_and_aligns_domain_ids():
    ds = _ds([{"text": "aaaa", "domain_id": 0}, {"text": "bbbb", "domain_id": 2}])
    out = _pack_with_domains(ds, _FakeTokenizer(), seq_len=4)
    assert len(out) >= 1
    for row in out:
        assert len(row["input_ids"]) == 4
        assert len(row["domain_ids"]) == 4
        assert len(row["labels"]) == 4


def test_each_token_keeps_its_own_documents_domain():
    ds = _ds([{"text": "aaa", "domain_id": 0}, {"text": "bbb", "domain_id": 2}])
    out = _pack_with_domains(ds, _FakeTokenizer(), seq_len=2)
    for row in out:
        for tok, dom in zip(row["input_ids"], row["domain_ids"]):
            if tok == 1:            # eos separator
                assert dom == -100
            elif tok == ord("a"):
                assert dom == 0
            elif tok == ord("b"):
                assert dom == 2
            else:
                pytest.fail(f"unexpected token {tok}")


def test_labels_equal_input_ids_unshifted():
    """The model shifts internally for CLM; shifting here too would train on the wrong
    target and still produce a falling loss curve."""
    ds = _ds([{"text": "abcdefgh", "domain_id": 1}])
    out = _pack_with_domains(ds, _FakeTokenizer(), seq_len=4)
    for row in out:
        assert row["labels"] == row["input_ids"]


def test_domain_ids_do_not_perturb_input_ids():
    """MAD-364 guard: labels are supervision, never input. The token stream must be
    identical to what a label-free packer would emit."""
    rows = [{"text": "hello", "domain_id": 0}, {"text": "world", "domain_id": 4}]
    out = _pack_with_domains(_ds(rows), _FakeTokenizer(), seq_len=4)
    got = [t for row in out for t in row["input_ids"]]

    expected = []
    for r in rows:
        expected += [ord(c) for c in r["text"]] + [1]
    expected = expected[: (len(expected) // 4) * 4]
    assert got == expected


def test_missing_domain_id_column_raises_rather_than_training_unsupervised():
    """A corpus without domain_id must stop the run, not silently produce a gate that
    gets no gradient while the config claims supervision."""
    ds = _ds([{"text": "abc"}])
    with pytest.raises(ValueError, match="domain_id"):
        _pack_with_domains(ds, _FakeTokenizer(), seq_len=2)


def test_missing_eos_raises():
    class _NoEos(_FakeTokenizer):
        eos_token_id = None

    ds = _ds([{"text": "abc", "domain_id": 0}])
    with pytest.raises(ValueError, match="eos_token_id"):
        _pack_with_domains(ds, _NoEos(), seq_len=2)


def test_out_of_range_domain_id_is_rejected():
    """A stale corpus carrying an id from an older enum must not become a valid-looking
    class index."""
    ds = _ds([{"text": "abc", "domain_id": 99}])
    with pytest.raises(ValueError):
        _pack_with_domains(ds, _FakeTokenizer(), seq_len=2)
