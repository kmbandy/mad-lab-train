"""Wire mlambaformer quant-native training callbacks into the SFTTrainer.

Returns the FFN-intermediate kurtosis gauge + W4A4 QAT-cooldown callbacks when
the model is an mlambaformer; returns [] for any other model. The callbacks
live in the mlambaformer package (installed in the training container); the
import is lazy and guarded so non-mlambaformer runs never require the dependency.

Both callbacks are opt-in, gated on model.config flags:
- ``log_kurtosis`` -> KurtosisGaugeCallback
- ``qat_cooldown_enabled`` -> QATCooldownCallback

Notes / pre-run gates:
- QATCooldownCallback is DISABLED BY DEFAULT (MAD-336). It installs weight
  parametrizations + activation pre-hooks mid-run, which relocates the state_dict
  keys to ``*.parametrizations.weight.original``. The resulting checkpoint cannot
  be reloaded: with tie_word_embeddings=True (the default) save_pretrained raises
  on the shared lm_head/embed_tokens tensor; with tying off it saves, but
  from_pretrained silently REINITIALIZES every parametrized Linear and reports it
  as a benign "newly initialized" notice. Do not set qat_cooldown_enabled until
  that checkpoint boundary is fixed.
- When it is re-enabled: it resolves its start step from state.max_steps via a
  fraction (default 0.93 = final ~7% of steps). This is validated for the
  single-GPU pretrain path. Under multi-GPU DeepSpeed ZeRO-2 (the _pretrain_worker
  path), registering parametrizations mid-training is NOT yet validated against
  DeepSpeed's parameter partitioning — confirm before the multi-GPU 1B run.
- The W4A4 fake-quant scheme in mlambaformer.training.fake_quant must be
  reconciled against the real ml8-4 quantizer config before any committed run
  (see the quant-native design spec risk "ml8-4 config drift").
"""
from __future__ import annotations


def mlambaformer_quant_callbacks(model):
    """Extra TrainerCallbacks for an mlambaformer model; [] otherwise."""
    if getattr(model.config, "model_type", None) != "mlambaformer":
        return []
    try:
        from mlambaformer.training.callbacks import (
            KurtosisGaugeCallback,
            QATCooldownCallback,
        )
    except ImportError:
        # mlambaformer not installed in this environment — nothing to wire.
        return []

    callbacks = []
    if getattr(model.config, "log_kurtosis", False):
        callbacks.append(KurtosisGaugeCallback(log_every=50))
    # Opt-in: the cooldown writes checkpoints that cannot be reloaded (MAD-336).
    if getattr(model.config, "qat_cooldown_enabled", False):
        callbacks.append(
            QATCooldownCallback(
                cooldown_fraction=0.93,
                bits_w=getattr(model.config, "qat_bits_w", 4),
                bits_a=getattr(model.config, "qat_bits_a", 4),
            )
        )
    return callbacks
