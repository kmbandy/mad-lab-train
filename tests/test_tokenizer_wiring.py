"""MAD-327: MAD-160 must use the pinned 48k slice, and the model geometry must match
the tokenizer -- never silently build the wrong vocab_size (the 261-token bug class)."""
import pytest

from mlambaformer.tokenization import get_tokenizer_dir
from pipeline.executors.pretrain import assert_vocab_matches


def test_pinned_slice_is_resolvable():
    d = get_tokenizer_dir("mad160-48k")
    assert (d / "tokenizer.json").exists()


def test_vocab_mismatch_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        assert_vocab_matches(arch_vocab_size=32000, tokenizer_len=48000)


def test_vocab_match_ok():
    assert_vocab_matches(arch_vocab_size=48000, tokenizer_len=48000) is None
