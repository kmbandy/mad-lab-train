"""MAD-327: bits-per-byte is the tokenizer-independent metric that makes the nested
tokenizer family comparable across model sizes."""
import math

from pipeline.executors.eval import bits_per_byte


def test_bpb_matches_hand_calc():
    # 10 nats total over 8 bytes -> (10/ln2)/8 bits per byte
    assert bits_per_byte(10.0, total_tokens=4, total_bytes=8) == \
        (10.0 / math.log(2)) / 8


def test_bpb_zero_bytes_is_zero():
    assert bits_per_byte(5.0, total_tokens=2, total_bytes=0) == 0.0
