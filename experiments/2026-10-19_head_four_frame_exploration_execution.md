# Four-frame head-resolution diagnostic execution (2026-10-19)

## Purpose and scope

Executed the frozen 4-frame exploratory diagnostic in
`2026-10-19_head_four_frame_resolution_protocol.md` because the GPU was idle.
It is an attention-shape measurement only. The closed confirmatory head
faithfulness branch remains closed; this run cannot establish faithfulness,
causal relevance, or grounding.

## Setup and execution

- Manifest: `data/nextgqa_samples_big.json`, 61 ordered rows; manifest hash was
  verified against the output.
- Model: local Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit, BF16 compute, eager
  attention, 4 sampled frames, `max_pixels=200704`.
- A three-row smoke completed 3/3 in 3.452 s inference, observed two temporal
  bins, and reached 6.740 GiB peak allocated VRAM.
- Full execution requested 61 and completed 58; three rows failed with sampled
  frame-read errors. Inference took 32.064 s, with 6.743 GiB peak allocated
  VRAM. Hardware was the RTX 5070 Ti. These are allocator readings, not total
  system memory use.
- The local JSON artifact is mode `0600` under the private data directory:
  `data/private_nextgqa/head_four_frame_full_20261019_01.json`, SHA-256
  `87deff184e919681a316c4d38b35fb6f4525ee8ce94954e51c31101794cbb3c3`.
  Raw media and derived head metrics remain local; rights status is unverified.

## Verified result

The scan computed all 784 head summaries. The fixed sink rule retained zero
heads: minimum observed sink fraction was exactly 0.5, and all 784 values were
at least 0.5. This is structurally forced with only two temporal bins: for any
head, the more frequent of two peak bins must occur at least half the time.
Because the rule requires `<0.5`, no head can pass. The elbow retained 104
heads, but the subsequent sink filter removed all of them; therefore there is
no top-20 list and no meaningful 4-vs-8 or 4-vs-16 Jaccard comparison. Do not
interpret empty-set overlaps as evidence that rankings disagree.

Independent CPU checks verified output and manifest hashes, source-code hashes,
private file/directory permissions, the complete unique 784-head inventory,
and exact recomputation of elbow and sink-filter counts. The result itself
therefore identifies a design-domain limit of the existing heuristic at two
bins; it does not support retuning the threshold after observing this run.
The exact runner SHA-256 stored in the artifact was independently reproduced
from the committed source at `9c0208b`; the runner was later extended with a
6-frame option without changing this run's recorded code identity.

## Faithfulness progress and next action

Direct explanation-faithfulness remains unchanged: formal claim review **0/3**,
verified clock/frame correspondence **0/15**, prediction-link judgment **0/3**,
and direct score **0/3**. No direct effect estimate, interval, or test exists.
The confirmatory head test remains negative (79/79; −0.0756 nat; 95% t CI
[−0.1578, +0.0065]) and closed.

Do not run another two-bin head ranking with this sink threshold or adjust the
threshold based on this outcome. Return to the direct-faithfulness/review
pipeline; the GPU is now available for the next scientifically justified,
pre-specified workload.
