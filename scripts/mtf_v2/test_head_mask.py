import torch, types, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from head_mask import sample_subset, layer_mask, restore_layer_mask, install
n, H = 12, 4; NEG = torch.finfo(torch.float32).min
causal = ((1 - torch.tril(torch.ones(n, n))) * NEG)[None, None]
blk = torch.tril(torch.ones(n, n)); blk[6:, 2:5] = 0; block = ((1 - blk) * NEG)[None, None]
S = sample_subset(range(3), H, 0.5, 7); assert S == sample_subset(range(3), H, 0.5, 7) and S != sample_subset(range(3), H, 0.5, 8)
assert sample_subset(range(3), H, 0.0, 1) == {} and all(len(v) == H for v in sample_subset(range(3), H, 1.0, 1).values())
buf = layer_mask(causal, block, [1, 3], H)
assert torch.equal(buf[:, 1], block[:, 0]) and torch.equal(buf[:, 3], block[:, 0]) and torch.equal(buf[:, 0], causal[:, 0]) and torch.equal(buf[:, 2], causal[:, 0])
restore_layer_mask(buf, causal, [1, 3]); assert torch.equal(buf, causal.expand(1, H, n, n))
# wrapper: replaces mask on first forward only, and undo restores
class A:  # dummy attention
    def forward(self, *a, **kw): return kw["attention_mask"]
ORIG = torch.zeros(1, 1, 1, n + 5)  # decode-step mask length != position length
layers = [types.SimpleNamespace(self_attn=A()) for _ in range(3)]
m = {1: layer_mask(causal, block, [0], H)}; undo = install(layers, m)
first = layers[1].self_attn.forward(position_ids=torch.zeros(1, n), attention_mask=torch.zeros(1, 1, n, n))
dec = layers[1].self_attn.forward(position_ids=torch.zeros(1, 1), attention_mask=ORIG); assert dec is ORIG
other = layers[0].self_attn.forward(position_ids=torch.zeros(1, n + 5), attention_mask=ORIG)
assert torch.equal(first, m[1]) and other is ORIG
undo(); assert layers[1].self_attn.forward(position_ids=torch.zeros(1, n + 5), attention_mask=ORIG) is ORIG
print("head_mask tests OK")
