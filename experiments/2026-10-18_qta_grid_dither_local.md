# QTA 32-frame grid-dither stability diagnostic (local result)

## Purpose and preregistered diagnostic

After the fixed-cohort 16/32-frame comparison showed partial selector support
overlap, while repeated forwards on the same batch and a separate process
replication were exact, this exploratory diagnostic changed only a frozen
sampling-grid perturbation while retaining 32 frames and the same cohort. It
tests selection stability to this specific input-sampling perturbation. It is
not an explanation-faithfulness score, causal intervention effect, QA metric,
or population estimate.

The baseline is the existing endpoint-inclusive uniform 32-frame decoded-index
grid. The diagnostic grid preserves both endpoints, shifts each odd 1-based
interior sample slot forward by one decoded-frame index, and leaves each even
interior slot unchanged. This rule was frozen before outcome inspection. Every
baseline and dither grid had to remain sorted, unique, and within the existing
50 ms actual-PTS uniform-grid tolerance. No row was replaced. The comparison
used the same processor, prompt/question, pinned model, and QTA aggregation;
one untreated forward was run per condition. The primary statistic is
ordinal temporal-group-index-set Jaccard; actual-PTS support Jaccard and
selected-center W1 are supplementary. There is no threshold, p-value, or CI.

The decoder and processor compatibility preflight completed for the fixed
three candidates before model loading. On the baseline grids, maximum PTS
deviation was 16.145 ms for each candidate; on the dither grids it was 46.283,
49.512, and 49.512 ms. All satisfy the frozen 50 ms bound, including the two
near-boundary cases. Token count, processor grid, input IDs, attention mask,
FPS and pixel tensor shape/dtype were checked across each pair before model
loading. A 1-candidate/2-forward GPU smoke passed before the full run.

## Run and provenance

- Cohort: same first three deterministic eligible NExT-GQA test candidates
  used in the preceding QTA stability diagnostics; fixed order, no replacement.
- Model: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
  PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5070 Ti. Seed 20261018.
- Full run: 3/3 pairs and 6/6 forwards completed; no failed or unattempted
  pairs/forwards. All six attention-capture receipts passed one-forward,
  per-layer-once, and normalization checks.
- Runtime: 87.72 s end to end; model snapshot hashing took 13.69 s before load
  and 12.38 s after forwards. Pre-load free VRAM was 14.655 GiB.
- Local-only summary:
  `data/private_nextgqa/qta_grid_dither_20261018_2220/full.json`, mode `0600`
  under mode `0700`, SHA-256
  `37d4f6ed7bca80760d7376d7a018af05c1b6a5a02b412995e7e6e7bc7cf0cb6a`.
  Rights status remains unverified. No raw video, ID, question/answer text,
  selection indices, logits, or video-derived row was saved.
- Provenance: the source manifest digest is
  `1c487cd48db16838265e58849693f30fbb0bde56230b82fcfe9bb31abce15be3`; all
  ten source file hashes and the digest matched a post-run reread. The 14
  candidate-exclusion manifests and hashes also match a post-run reread. The
  model snapshot content hash before load and after forwards was
  `1325edfbc585ee74c5cb8666dc96ce3881fdd1cf2220cc315f9fa39a9f41aa93`.
  An independent local verification recomputed this checkpoint hash, source
  hashes, manifest hashes, counts, PTS bounds, output mode, and metric means.

## Results

| Fixed pair ordinal | Group-index Jaccard | Actual-PTS support Jaccard | Selected-center W1 |
|---:|---:|---:|---:|
| 1 | 1.0000 | 0.99693 | 0.0001981 |
| 2 | 1.0000 | 0.99338 | 0.0004288 |
| 3 | 1.0000 | 0.99054 | 0.0003569 |
| Mean | **1.0000** | **0.99362** | **0.0003279** |

Primary mean was recomputed from the three fixed ordinal rows. The same
readback reproduced both supplementary means exactly at stored precision. The
independent auditor separately matched all three metrics and confirmed counts,
permissions, and privacy schema.

## Interpretation and limits

The selected ordinal group slots were identical across baseline and dither
conditions for these three fixed videos. Actual-PTS support was close but not
identical because the selected slots map to slightly shifted frame times. This
supports stability of the chosen group indices to this particular +1/0
decoded-frame perturbation under the tested model, processor, prompt, and
selector. It does not separate positional metadata from changed image pixels;
sampling perturbation changes both. It does not explain the earlier 16/32-frame
discrepancy, prove that selected frames support an explanation claim, measure
prediction dependence, or establish faithfulness. The tiny reused cohort has
no population inference.

Direct faithfulness remains **0/3** reviewed claims, **0/15** independently
verified browser/player-to-decoder frame correspondences, **0/3** prediction-
link judgments, and **0/3** direct scores. There is no direct `DeltaF`, CI,
or significance test. Human review still awaits genuine role/key assignments,
independently pinned trust root, signed review-only custody, witnessed unopened
handoff, and passing pre-open audit. This diagnostic does not satisfy any human
gate. Keep the 32-frame primary setting frozen and do not publish the protected-
derived JSON or this log.

## Validation

The focused QTA stability, repeatability, grid-dither, and loader-classification
tests passed 32/32. The complete repository suite passed **2,043/2,043** tests
in 85.81 s. Compilation, CLI help, and `git diff --check` passed.
