"""Path patching for attention knockout: close detour routes by freezing attention outputs to their clean values.
The unit is the input of o_proj ([1, seq, H*d], per-head contexts). In a patched run
  - sender (layer, head) at sender positions is computed live (under the knockout mask installed by the caller),
  - receiver positions are computed live (all heads) in layers after the first sender layer,
  - every other attention output is replaced by the clean value. MLPs and norms are always recomputed."""
import torch


def record(layers):
    """Hook every layer; returns (cache {layer: cpu tensor}, undo). Run one clean forward, then call undo()."""
    cache, hooks = {}, []
    for i, layer in enumerate(layers):
        def pre(mod, args, i=i): cache[i] = args[0].detach().cpu().clone()
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
    return cache, lambda: [h.remove() for h in hooks]


def freeze(layers, cache, senders, sender_pos, recv_pos, H):
    """senders: {layer: iterable of heads}. Returns undo()."""
    first = min(senders); hooks = []
    for i in range(first, len(layers)):
        def pre(mod, args, i=i):
            live = args[0]; d = live.shape[-1] // H
            out = cache[i].to(live.device)
            assert out.shape == live.shape
            out = out.clone()
            for h in senders.get(i, ()):
                out[0, sender_pos, h * d:(h + 1) * d] = live[0, sender_pos, h * d:(h + 1) * d]
            if i > first and recv_pos: out[0, recv_pos] = live[0, recv_pos]
            return (out,)
        hooks.append(layers[i].self_attn.o_proj.register_forward_pre_hook(pre))
    return lambda: [h.remove() for h in hooks]
