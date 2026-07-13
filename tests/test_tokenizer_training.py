"""MAD-327: a raw-text corpus produced a 261-token tokenizer, silently.

_text_iter read ONLY obj["messages"]:

    obj  = json.loads(line)
    msgs = obj.get("messages", [])
    for m in msgs:
        yield m.get("content", "")

A pretraining corpus is {"text": ...} -- that is what data_format: "text" and
dataset_text_field="text" mean. `.get("messages", [])` returns [], the loop yields
nothing, and the `except` only catches JSON *parse* failures, so a perfectly valid
{"text": ...} line parsed fine and contributed no training text.

BPE then trained on an EMPTY ITERATOR and produced 256 byte tokens + 5 specials = 261.
And _load_architecture(arch, len(tokenizer)) takes the model's vocab_size straight from
len(tokenizer) -- so the model would have been built with a 261-token vocabulary instead
of 32,000, with nothing raised and nothing logged.
"""
import json
import random
import string

import pytest

from pipeline.executors.pretrain import _train_tokenizer

# A corpus with enough LEXICAL DIVERSITY that BPE can actually learn merges. A handful of
# repeated words cannot produce >512 tokens no matter how many lines you write, and a
# fixture like that tests the guard rather than the fix.
_RNG = random.Random(0)
_LEXICON = [
    "".join(_RNG.choice(string.ascii_lowercase) for _ in range(_RNG.randint(3, 10)))
    for _ in range(3000)
]


def _text(i):
    r = random.Random(i)
    return " ".join(r.choice(_LEXICON) for _ in range(40))


def _corpus(tmp_path, shape, n=800):
    p = tmp_path / "training.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            if shape == "text":
                f.write(json.dumps({"text": _text(i)}) + "\n")
            elif shape == "messages":
                f.write(json.dumps({"messages": [{"role": "user", "content": _text(i)}]}) + "\n")
            elif shape == "empty":
                f.write(json.dumps({"id": i, "meta": "no text anywhere"}) + "\n")
    return tmp_path


def test_raw_text_corpus_trains_a_real_tokenizer(tmp_path):
    """THE REGRESSION. A {"text": ...} corpus used to yield a 261-token tokenizer."""
    events = []
    tok = _train_tokenizer(
        _corpus(tmp_path, "text"), tmp_path / "tok", 2000,
        lambda e, d: events.append((e, d)),
    )
    assert tok.vocab_size > 512, (
        f"got a {tok.vocab_size}-token tokenizer from a raw-text corpus -- "
        "_text_iter is ignoring the 'text' field again (261 = 256 bytes + 5 specials)"
    )
    trained = dict(events)["tokenizer_trained"]
    assert trained["docs"] == 800, "documents were not counted / not read"


def test_messages_corpus_still_works(tmp_path):
    """The original format must keep working -- this is a fix, not a swap."""
    tok = _train_tokenizer(
        _corpus(tmp_path, "messages"), tmp_path / "tok", 2000, lambda e, d: None
    )
    assert tok.vocab_size > 512


def test_empty_corpus_raises_instead_of_shipping_a_byte_tokenizer(tmp_path):
    """The guard. A byte-level BPE trained on nothing still 'succeeds' and returns 261
    tokens -- which then becomes the model's vocab_size. It must fail loudly."""
    with pytest.raises(ValueError, match="ZERO documents"):
        _train_tokenizer(
            _corpus(tmp_path, "empty"), tmp_path / "tok", 2000, lambda e, d: None
        )


def test_plain_txt_lines_are_read(tmp_path):
    """Non-JSON .txt lines are legitimate corpus text and must not be dropped."""
    (tmp_path / "corpus.txt").write_text(
        "\n".join(_text(i) for i in range(800)) + "\n"
    )
    tok = _train_tokenizer(tmp_path, tmp_path / "tok", 2000, lambda e, d: None)
    assert tok.vocab_size > 512
