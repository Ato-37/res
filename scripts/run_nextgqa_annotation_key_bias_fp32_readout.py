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

if __name__ == "__main__":
    raise SystemExit(base.main() if hasattr(base, "main") else base.run())
