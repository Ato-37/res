#!/usr/bin/env python3
"""Run run_nextgqa_annotation_key_bias with an fp32 lm_head readout.

The base runner scores five-choice margins from bf16 logits; at |logit| in [16, 32) bf16 spacing is 0.125 nat,
so per-row margin contrasts are quantized and effects of ~0.01-0.03 nat sit below one quantum. This wrapper leaves
the base runner file (and the hashes recorded in its receipts) untouched and only replaces the final linear readout
with an fp32 matmul of the same bf16 final hidden state and the same weights. Every condition uses the identical
readout. It changes the measurement, not the intervention; label receipts accordingly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

import run_nextgqa_annotation_key_bias as base

READOUT = "fp32_lm_head_matmul_of_bf16_final_hidden_state"
_original = Qwen2_5_VLForConditionalGeneration.from_pretrained.__func__


def _patched(cls, *args, **kwargs):
    model = _original(cls, *args, **kwargs)
    head = model.lm_head
    if head.bias is not None or head.weight.dtype != torch.bfloat16:
        raise RuntimeError("unexpected_lm_head_for_fp32_readout")
    weight = head.weight

    def forward(hidden):
        return F.linear(hidden.float(), weight.float())

    head.forward = forward
    model._fp32_readout = READOUT
    return model


Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(_patched)

def _consume_layers_option(argv):
    """Pop an optional `--layers all` and set the base module's intervention layers to 1..28 (1-based)."""
    if "--layers" not in argv:
        return argv
    i = argv.index("--layers")
    if argv[i + 1] != "all":
        raise SystemExit("layers_option_must_be_all")
    base.LAYERS_1_BASED = tuple(range(1, 29))
    return argv[:i] + argv[i + 2:]


def _consume_bias_option(argv):
    """Pop an optional `--bias-nats X` (requested additive attention-score bias) and set it on the base module."""
    if "--bias-nats" not in argv:
        return argv
    i = argv.index("--bias-nats")
    value = float(argv[i + 1])
    if value == 0.0 or not -8.0 <= value <= 4.0:
        raise SystemExit("bias_nats_out_of_range")
    base.BIAS_NATS = value
    return argv[:i] + argv[i + 2:]


if __name__ == "__main__":
    sys.argv = _consume_layers_option(_consume_bias_option(sys.argv))
    raise SystemExit(base.main())
