"""D2/MAD-322: the domain label must survive packing and reach the model per-token.

TRL's packing=True concatenates documents and keeps only the text column, so a
per-document domain_id cannot become per-token supervision. Without the path these tests
cover, the domain gate receives NO gradient and stays a fixed random projection for the
entire run -- with a perfectly plausible loss curve, which is why it went unnoticed.

MAD-364 is the trap that makes this delicate: domain labels are supervision ONLY and must
never enter the forward pass as an input. An in-stream domain tag was deliberately cut
because under causal attention every token attends to a tag at position 0 and Mamba's
recurrent state carries it forward, so the gate would learn a trivial lookup and then
collapse at inference where no tag exists. The test that input_ids is unaffected by the
label is therefore not a nicety.

These exercise `iter_packed_domain_rows`, which takes any object exposing column_names /
__len__ / iteration -- deliberately NOT a datasets.Dataset. The first version of this file
imported `datasets`, and the only environment that can import both `datasets` and
`mlambaformer` is a 40 GB image that is not built on this box, so the tests could not run
anywhere. A test that cannot run is not a test.
"""
import pytest

from pipeline.executors._pretrain_worker import iter_packed_domain_rows


class _FakeTokenizer:
    """One token per character, id = ord(c). Deterministic and inspectable, so a test can
    say exactly which document a token came from."""

    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class _Rows(list):
    """Minimal stand-in for a HF Dataset: a list of dicts that also reports its columns."""

    @property
    def column_names(self):
        return sorted({k for row in self for k in row}) if self else []


def _ds(rows):
    return _Rows(rows)


def _pack(rows, seq_len, tokenizer=None):
    return list(iter_packed_domain_rows(_ds(rows), tokenizer or _FakeTokenizer(), seq_len))


def test_packs_and_aligns_domain_ids():
    out = _pack([{"text": "aaaa", "domain_id": 0}, {"text": "bbbb", "domain_id": 2}], 4)
    assert out
    for row in out:
        assert len(row["input_ids"]) == 4
        assert len(row["domain_ids"]) == 4
        assert len(row["labels"]) == 4


def test_each_token_keeps_its_own_documents_domain():
    out = _pack([{"text": "aaa", "domain_id": 0}, {"text": "bbb", "domain_id": 2}], 2)
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
    for row in _pack([{"text": "abcdefgh", "domain_id": 1}], 4):
        assert row["labels"] == row["input_ids"]


def test_domain_ids_do_not_perturb_input_ids():
    """MAD-364 guard: labels are supervision, never input. The token stream must be
    identical to what a label-free packer would emit."""
    rows = [{"text": "hello", "domain_id": 0}, {"text": "world", "domain_id": 4}]
    got = [t for row in _pack(rows, 4) for t in row["input_ids"]]

    expected = []
    for r in rows:
        expected += [ord(c) for c in r["text"]] + [1]
    expected = expected[: (len(expected) // 4) * 4]
    assert got == expected


def test_a_window_may_span_two_documents():
    out = _pack([{"text": "aa", "domain_id": 0}, {"text": "bb", "domain_id": 1}], 6)
    assert out
    assert {d for d in out[0]["domain_ids"] if d != -100} == {0, 1}


def test_missing_domain_id_column_raises_rather_than_training_unsupervised():
    """A corpus without domain_id must stop the run, not silently produce a gate that
    gets no gradient while the config claims supervision."""
    with pytest.raises(ValueError, match="domain_id"):
        _pack([{"text": "abc"}], 2)


def test_missing_eos_raises():
    class _NoEos(_FakeTokenizer):
        eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        _pack([{"text": "abc", "domain_id": 0}], 2, tokenizer=_NoEos())


def test_out_of_range_domain_id_is_rejected():
    """A stale corpus carrying an id from an older enum must not become a valid-looking
    class index."""
    with pytest.raises(ValueError):
        _pack([{"text": "abc", "domain_id": 99}], 2)


def test_too_many_documents_refuses_the_inline_path():
    """Dataset.from_generator materialises an int64 Arrow cache (~24 bytes/token). At 50B
    tokens that is 1.2 TB against 407 GB free, so the inline path must refuse rather than
    fill the disk partway through a multi-hour tokenisation."""

    class _Huge(_Rows):
        def __len__(self):
            return 500_001

        @property
        def column_names(self):
            return ["text", "domain_id"]

    with pytest.raises(ValueError, match="pack_corpus"):
        list(iter_packed_domain_rows(_Huge(), _FakeTokenizer(), 4))
