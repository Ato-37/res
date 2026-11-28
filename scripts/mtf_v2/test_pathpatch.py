import torch
from transformers import LlamaConfig, LlamaForCausalLM
from head_mask import install
from pathpatch import record, freeze

torch.manual_seed(0)
H, L, n = 4, 5, 12
cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=H)
cfg._attn_implementation = "eager"
m = LlamaForCausalLM(cfg).double().eval(); layers = m.model.layers
ids = torch.randint(0, 50, (1, n)); video, question, last = [0, 1, 2, 3], [5, 6, 7], n - 1

def mask(q=(), k=()):
    a = torch.full((n, n), torch.finfo(torch.float64).min, dtype=torch.float64).triu(1)
    for i in q:
        for j in k: a[i, j] = torch.finfo(torch.float64).min
    return a[None, None]
causal, block = mask(), mask(question, video)

@torch.no_grad()
def run(masks=None, senders=None, spos=None, rpos=None):
    u1 = install(layers, masks) if masks else (lambda: None)
    u2 = freeze(layers, cache, senders, spos, rpos, H) if senders else (lambda: None)
    try: return m(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
    finally: u2(); u1()

def one(l, h):
    x = causal.expand(1, H, n, n).clone(); x[:, h] = block[:, 0]; return {l: x}

cache, undo = record(layers)
clean = run(); undo()
# no knockout -> patched run equals clean
assert torch.allclose(run({1: causal.expand(1, H, n, n).clone()}, {1: [0]}, question, [last]), clean, atol=1e-12)
# the knockout does something when nothing is frozen
assert not torch.allclose(run(one(1, 0)), clean, atol=1e-9)
# sender at question positions, nothing downstream live -> cannot reach the last position
assert torch.allclose(run(one(1, 0), {1: [0]}, question, []), clean, atol=1e-12)
# with the last position live it reaches the output, but differently from the unfrozen run
p = run(one(1, 0), {1: [0]}, question, [last])
assert not torch.allclose(p, clean, atol=1e-9) and not torch.allclose(p, run(one(1, 0)), atol=1e-9)
# everything live -> identical to the plain knockout
every = list(range(n))
assert torch.allclose(run(one(1, 0), {1: list(range(H))}, every, every), run(one(1, 0)), atol=1e-12)
# last layer sender: no downstream attention, so frozen == unfrozen
assert torch.allclose(run(one(L - 1, 2), {L - 1: [2]}, question, []), run(one(L - 1, 2)), atol=1e-12)
print("pathpatch tests passed")
