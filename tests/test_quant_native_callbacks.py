"""QAT cooldown must be opt-in (MAD-336).

QATCooldownCallback installs torch.nn.utils.parametrize wrappers mid-run. The
resulting checkpoint is unloadable: with tie_word_embeddings=True (the default)
save_pretrained RAISES on the shared lm_head/embed_tokens tensor; with tying off
it saves, but from_pretrained silently reinitializes every parametrized Linear
because the keys moved to *.parametrizations.weight.original.

Until that boundary is fixed, the callback must never be wired by default.

Run under the mlambaformer venv, which has both pytest and mlambaformer:
    PYTHONPATH=~/GitHub/mad-lab-train \
      ~/GitHub/mlambaformer/.venv/bin/pytest tests/test_quant_native_callbacks.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.executors._quant_native_callbacks import mlambaformer_quant_callbacks


def _model(**config_fields):
    config_fields.setdefault("model_type", "mlambaformer")
    return SimpleNamespace(config=SimpleNamespace(**config_fields))


def _names(callbacks):
    return {type(cb).__name__ for cb in callbacks}


def test_non_mlambaformer_gets_no_callbacks():
    assert mlambaformer_quant_callbacks(_model(model_type="qwen2")) == []


def test_qat_cooldown_is_not_wired_by_default():
    """The MAD-336 regression guard. A default mlambaformer run must not
    install fake-quant parametrizations, because the checkpoint it writes
    cannot be reloaded."""
    assert "QATCooldownCallback" not in _names(mlambaformer_quant_callbacks(_model()))


def test_qat_cooldown_is_wired_when_explicitly_enabled():
    cbs = mlambaformer_quant_callbacks(_model(qat_cooldown_enabled=True))
    assert "QATCooldownCallback" in _names(cbs)


def test_kurtosis_gauge_remains_opt_in():
    assert "KurtosisGaugeCallback" not in _names(mlambaformer_quant_callbacks(_model()))
    cbs = mlambaformer_quant_callbacks(_model(log_kurtosis=True))
    assert "KurtosisGaugeCallback" in _names(cbs)


def test_enabling_both_wires_both():
    cbs = mlambaformer_quant_callbacks(
        _model(log_kurtosis=True, qat_cooldown_enabled=True)
    )
    assert _names(cbs) == {"KurtosisGaugeCallback", "QATCooldownCallback"}
