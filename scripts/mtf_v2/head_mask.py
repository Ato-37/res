"""Head-level cross-frame blocking for the official LLaVA-NeXT-Video code (plan 5, step 0).
Pure-tensor helpers (CPU-testable) + a forward-wrapper installer. A subset S of (layer, head) pairs blocks the cross-frame vision x vision
cells; all other heads keep the plain causal mask. The block mask `block` is the official [1,1,n,n] additive mask from precompute_attention_masks
(cross-frame cells), `causal` the plain causal one; both come from the same call with/without ranges.
"""
import functools, random
import torch


def sample_subset(layers, H, p, seed):
    """Random subset: each (layer, head) in layers x range(H) is included independently with prob p. Returns {layer: sorted head list}."""
    rng = random.Random(seed); out = {}
    for l in layers:
        hs = [h for h in range(H) if rng.random() < p]
        if hs: out[l] = hs
    return out


def layer_mask(causal, block, heads, H, buf=None):
    """[1,H,n,n] additive mask: heads in `heads` use `block`, others `causal`. Reuses `buf` (restored by caller via restore_layer_mask)."""
    n = causal.shape[-1]
    if buf is None: buf = causal.expand(1, H, n, n).clone()
    buf[:, heads] = block
    return buf


def restore_layer_mask(buf, causal, heads):
    buf[:, heads] = causal


def install(model_layers, masks):
    """masks: {layer: [1,H,n,n]}. Wraps self_attn.forward for those layers (first forward only, as the official wrapper). Returns undo()."""
    saved = {}
    def wrap(fn, m):
        @functools.wraps(fn)
        def w(*a, **kw):
            if kw["position_ids"].size(-1) == kw["attention_mask"].size(-1): kw["attention_mask"] = m
            return fn(*a, **kw)
        return w
    for l, m in masks.items():
        saved[l] = model_layers[l].self_attn.forward
        model_layers[l].self_attn.forward = wrap(saved[l], m)
    def undo():
        for l, f in saved.items(): model_layers[l].self_attn.forward = f
    return undo
