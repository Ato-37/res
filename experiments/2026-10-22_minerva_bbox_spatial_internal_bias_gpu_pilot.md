# Minerva bbox spatial internal-bias GPU pilot (2026-10-22)

## Question and design

Does a fixed dataset-authored rationale line's teacher-forced likelihood respond
more to a small internal visual-key bias on its annotation-linked object region
than to equal-dose spatial or temporal sham positions? This is an exploratory
prediction-dependence proxy, not a generated-explanation faithfulness measure.

The selected row came from frozen cohort
`data/private_minerva/minerva_ego_bbox_cohort_fresh3_20261022.json`
(SHA-256 `ae45c65ed9a268302b37fc6561631add23190194cd6a4b9ecda52bed2660480a`).
The cohort requested 3 rows; only P05 media was initially available. The
CPU-only cohort preflight receipt `_04` (SHA-256
`45323c2ee93c59e9b08c0e61ae8ac3b6ebe05f911973bb512eb37b870962619b`)
recorded 1 passing row and 2 `video_file_missing` failures. The current
annotation JSON, object association, and mask source hashes match the frozen
cohort. The MP4 SHA-256 is
`4af2c06780f40fa6395599b2d9884b7188959063367fcc0f7e4658598701bb22`.

The runner used the official time map to sample 56 frames over a 27.5-second
clip and inserted the exact annotated source frame 306 (10.2 s) at sample rank
20. The Qwen2.5-VL-7B checkpoint was pinned to `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
4-bit weights with BF16 compute, eager attention, and +ln(2) visual-key bias at
layers 5–16. Hardware/runtime: NVIDIA RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0,
Transformers 5.18.0, qwen-vl-utils 0.0.14. The four conditions were baseline,
six bbox keys, six deterministic non-bbox keys in the same temporal pair, and
the same six bbox-cell offsets at matched non-track temporal group 24. Each
condition applied one scored branch through the same layer hooks.

## Verified results

The full-prompt CPU smoke passed before GPU loading. It used the same 56-frame
video, produced `video_grid_thw=[28,22,22]` (28 temporal groups, 121 merged
cells per group), 3388 video tokens, and mapped the annotation bbox to six
actual prompt video-token positions in group 10. All three nonbaseline
conditions biased exactly six distinct keys. The CPU receipt is
`data/private_minerva/minerva_ego_bbox_spatial_token_preflight_20261022_01.json`
(SHA-256 `ac1c06e127460496fc98bbbdb98cac38d1bd5a9ca312a4bb38f750ff78cb53c7`,
mode 0600).

GPU result: `data/private_minerva/minerva_ego_bbox_spatial_gpu_20261022_01.json`
(SHA-256 `0d0b983baba73e4187e4b35ecc5c496f38ecbbe0020f92869100d3d4ee209b9c`,
mode 0600). It has 37 target tokens and one score per condition:

| Condition | Mean target log probability (nats/token) | Difference from bbox support |
|---|---:|---:|
| Baseline | −2.793748 | +0.013686 |
| Bbox support bias | −2.807434 | 0 |
| Same-pair spatial sham | −2.786545 | +0.020889 |
| Temporal bbox sham | −2.849629 | −0.042195 |

Thus bbox support was 0.013686 nat/token below baseline and 0.020889 below the
same-pair spatial sham; it was 0.042195 above the temporal bbox sham. There is
one video and one rationale line, so no uncertainty interval or inferential
test is estimable. Peak allocated VRAM was 10,159,492,096 bytes; scoring took
12.89 seconds (model loading excluded).

## Interpretation and limits

Observed: the fixed rationale likelihood did not improve under the bbox-targeted
bias relative to baseline or the equal-dose same-time spatial sham. It was
higher than under the same bbox offsets at a different temporal group. This is
consistent with temporal-position sensitivity for this one sentence; it does
not establish spatial grounding improvement, causal explanation faithfulness,
or a population effect. The hashed random spatial sham matches token count, not
object salience, shape, or semantic information.

Qwen's input grid has temporal patch size 2. The annotated 10.2-s frame is
paired with the 10.5-s frame, so the intervention is on the pair representation,
not the isolated annotated frame. Exact-frame insertion makes local sample gaps
0.3–0.7 s although the processor metadata uses nominal 2 fps (`second_per_grid_ts`
is 1 s/group). Recalculation against the nominal 0.5-s frame grid gives a
maximum source-time deviation of 0.2 s at the inserted anchor. Updated receipts
now store this deviation; the GPU output was produced before that correction
and its `maximum_grid_time_error_seconds=0.0` field is stale. The same GPU JSON
also omits an explicit `preflight_receipt_sha256`; its runner hash matches the
decoder-source hash in receipt `_04`, and the command explicitly consumed `_04`.
These provenance gaps are recorded rather than silently repaired in the result.

Independent audit reproduced all four score contrasts, identical target token
IDs, one hook per layer 5–16, and the equal six-key doses. It also found that
the GPU JSON omits `preflight_receipt_sha256` and annotation-source hashes. The
recorded GPU runner SHA-256 (`09711c67…1f618c`) matches the decoder source hash
in `_04`, and the actual invocation in the execution record passed `_04`, but
the result JSON itself is not provenance-closed. The result is retained as an
exploratory observation, not an SSS artifact.

The GPU JSON also recorded zero nominal-grid deviation, which was carried over
from the pre-insertion sampling schedule and is stale. Recalculation from the
official time map gives 0.3–0.7 s gaps and a 0.2 s maximum deviation from the
nominal 0.5 s grid. The full-prompt CPU replay against corrected preflight `_05`
records `second_per_grid_ts=[1.0]`, the source hashes, and the corrected grid
deviation. The runner now checks current annotation sources and bbox coordinates
against cohort/preflight hashes and records these fields for future GPU outputs.
Both receipts and all videos remain local private data; rights remain unverified.

## Next action

## Three-row frozen-cohort completion (2026-10-22)

### Data and protocol repair

The frozen 3-row cohort requested P05, P06, and P07. The official dataset
manifest (SHA-256
`4b2bc0f8bae6c27d3521d7f319b3a9df16652da1f0492be3ff01261fd8197c06`)
verified all six selected MP4/time-map files after local-only retrieval:
2,041,005,727 bytes total, with official MD5 checks passing and two files
reused from earlier local storage. Rights remain unverified and all media and
row-level JSON remain local/private.

The first complete media preflight exposed a real alignment bug: rationale
times written as `01:17-01:46` were read as two isolated one-second bins, so an
exact annotation at 01:29 was incorrectly rejected. The helper now parses
explicit hyphen/en-dash/em-dash/`to` ranges as half-open spans
`[start, end+1)`, reflecting inclusive, second-precision displayed endpoints;
standalone times retain their original one-second-bin rule. It does not relax
the 0.05-second track-boundary tolerance. An independent code/data audit
verified P07's frame 2674 (89.1333s) lies inside both the stated 01:17–01:46
range and its object track. It also identified and we fixed a separate
sub-microsecond timestamp equality issue by requiring the exact sampled source
frame index plus <=1 ms PTS discrepancy.

The corrected CPU cohort receipt is
`data/private_minerva/minerva_ego_bbox_patch_preflight_20261022_08.json`
(SHA-256 `895bba8e0fcb66cf74da16efdc32b0ab179e59c5f980ded2e5f4f3bd701e7b52`):
3 requested, 3 passed. All three full-prompt CPU smokes passed and verified the
actual Qwen video-token geometry and equal per-row bias/sham doses (P05: 6
keys; P06: 15; P07: 9). The two earlier preflights are superseded: `_06` used
the endpoint-only parser, and `_07` selected the correct frame but failed on
exact floating-point equality. Neither was used by the GPU runs.

### Three-row GPU observations

The corrected runner produced four teacher-forced likelihood conditions per
video: baseline, annotation-bbox spatial bias, equal-dose random spatial sham
inside the same temporal pair, and equal-dose bbox-offset sham at another
temporal group. It ran 12 scoring forwards total, with 4-bit Qwen2.5-VL-7B,
BF16 compute, and +ln(2) key bias in layers 5–16. The 3-row run used the same
frozen cohort, official media hashes, source annotation hashes, CPU receipt
hash, and runner source hash. Scoring took 14.35–15.63s per row excluding model
loading; peak allocated VRAM ranged from 10,159,492,096 bytes (P05) to
11,274,653,184 bytes (P06). The exact values and receipt hashes
are emitted by `scripts/analyze_minerva_bbox_spatial_gpu_results.py`.

| Frozen row | bbox − baseline | bbox − same-pair spatial sham | bbox − temporal sham |
|---|---:|---:|---:|
| P05 | −0.013686 | −0.020889 | +0.042195 |
| P06 | −0.052788 | −0.036300 | −0.014265 |
| P07 | −0.018610 | −0.001359 | −0.006439 |
| Mean of 3 rows | −0.028361 | −0.019516 | +0.007164 |

The first two contrasts are negative for all three rows: bbox bias lowered the
fixed rationale likelihood versus baseline and versus the equal-dose same-time
spatial sham. Temporal-sham contrast varies by row and averages near zero.
These are descriptive paired differences for a fixed 3-row exploratory set;
no inferential test or confidence interval is justified here. This run does
not show an improvement from the bbox intervention.

Result JSON SHA-256 values: P05
`40769d4330d0615065922159420109d8d41ad2121981517d6e5081901480c950`, P06
`ae0a8119317eaca4a02228a79b072dce938855c13a18e1de86b48461938145c7`, P07
`f89777f1242d606fc7e31a321584f63537cf45429f9fb7300d179953a3e1fcb0`. The
reproduction command is:

```bash
.venv/bin/python scripts/analyze_minerva_bbox_spatial_gpu_results.py
```

### Interpretation and next action

Observed: the intervention did not raise likelihood of the dataset-authored
rationale on any of the three videos relative to baseline or the equal-dose
same-time spatial sham. This is evidence against this specific positive-bias
setting as an effective way to increase teacher-forced rationale dependence
on the annotation-bbox visual keys. It is not evidence that model explanations
are unfaithful in general: the target is dataset-authored rather than generated
by the model, one line is measured per video, the sham is not matched on
object-salience/shape/semantic content, and Qwen fuses two temporal frames per
group. P07's metadata-range fix establishes only a consistent timestamp/mask
join, not that the object visually supports the action described.

An independent GPU-result audit is pending. No human review was performed.
The earlier claim-support response JSON remains bound to retired artifact `_06`
and its self-reported answers contain no `supports` labels; it cannot validate
these three targets or be counted as a direct faithfulness result. The
independent output audit found no integrity, mapping, dose, or arithmetic
mismatches. Next is an explicitly exploratory polarity follow-up on the same
three rows: apply fixed-magnitude `-ln(2)` bias to bbox keys and equal-dose
spatial and temporal shams. This direction was selected after observing the
positive-bias results, so it is post-hoc and cannot serve as an independent
confirmatory test. The negative-bias run completed for all three rows. Its
bbox-minus-baseline contrasts were P05 +0.017502, P06 +0.004841, and P07
−0.013786 nat/token; bbox-minus-same-pair-sham contrasts were +0.057346,
+0.025910, and −0.036861. Means were +0.002852 and +0.015465 respectively.
Thus two rows moved upward under negative bbox bias and one downward; the
negative-bias selectivity minus positive-bias selectivity was also mixed, with
mean −0.034981 nat/token. These post-hoc observations provide no consistent
sign-sensitive dependence pattern. Receipt hashes: P05
`ba39c919b3c8cd06cd7ffb4e0d37a1bc2d6602eb3311bd6d51b0528e36de78b0`, P06
`c754332420bfe2ba75076b2cfcf9e168226f6810b222e736efb5f337cc5f6cfc`, P07
`b783f24f073110d6491a64fb527066fa590cdf0f1d95b5b6539323de780ac95a`.

The first negative-bias attempt safely failed before scoring because the
shared attention hook rejected negative logit bias; no result was produced.
The hook now accepts any finite nonzero sign, with a unit test verifying the
selected key changes. Before running, the fixed-cohort CPU preflight was
regenerated as `_10`, and all three full-prompt CPU smokes passed. An
independent audit of the new sign-controlled outputs is pending.

### Direct visual-key ablation and independent audit

The announced post-hoc follow-up removed (`-inf`) the selected keys rather than
adding a finite bias. Independent audit matched all three GPU outputs to the
same-row full-prompt CPU token receipts, the `_12` cohort preflight, source and
media hashes, and the expected per-layer deleted-key counts (P05 6, P06 15,
P07 9; matched across spatial evidence and shams). Recomputed score contrasts
match the outputs exactly. The three output SHA-256 values are recorded by
`scripts/analyze_minerva_bbox_spatial_gpu_results.py`; all remain under the
private local data directory.

| Frozen row | bbox ablation − baseline | bbox ablation − spatial sham | bbox ablation − temporal sham |
|---|---:|---:|---:|
| P05 | +0.014217 | +0.010864 | +0.028535 |
| P06 | −0.031523 | +0.001225 | −0.027964 |
| P07 | +0.000996 | +0.000725 | −0.003295 |
| Mean of 3 rows | −0.005437 | +0.004271 | −0.000908 |

The bbox-versus-spatial-sham contrast is positive in all three rows and small
(mean +0.004271 nat/token), so this check does not show stronger target-score
dependence on the annotation bbox than on an equal-dose random spatial sham.
The bbox-versus-baseline direction is mixed. This is descriptive, post-hoc,
three-row fixed-target evidence only; no inferential test or confidence
interval is warranted, and it is not generated-explanation faithfulness.

Audit caught a metadata ambiguity: the legacy GPU receipts labeled
`bias_nats=+ln(2)` although key-ablation applies no finite bias. Their actual
operation and per-layer records correctly specify key deletion. I updated the
runner so new ablation CPU/GPU receipts store `bias_sign` and `bias_nats` as
null and pass zero finite bias into the scorer, and updated the analyzer to
report the effective ablation operation rather than that unused legacy field.
The historical JSONs remain unchanged. The 53 relevant unit tests and Python
compile checks pass; the analyzer reproduces the above values.

A later independent audit found that the analyzer's no-argument default had
paired the positive-bias GPU files with preflight `_12`, although those outputs
bind to `_08`; the documented command failed. The default is now `_08` for the
built-in positive set and `_12` for an explicitly supplied all-ablation set;
custom non-ablation sets require an explicit `--preflight`. The old positive
GPU receipts do not contain an exact token-preflight receipt SHA, and their
saved prompt hashes differ from the available same-row token receipts. Their
source/media/cohort and arithmetic checks pass, but exact full-prompt CPU
binding is incomplete. This is a historical provenance limit, not repaired by
the analyzer change.

### Review JSON recheck

The requested Downloads response still hashes to
`3d9faa75edf9ce69fe9730b01a85094ded411a49bdfdb85761809c00f38cd267`. The
response inspector accepts it against retired artifact `_06`
(`abaa3b8dee8f4fc9c9bd2031d1772d2912502a9c1bafb1f2b03dac90365d8a1d4`)
and rejects it against current cohort artifact `_08` because the fingerprint
and ordered item inventory differ. It has three self-reported reviewers and
nine complete answers: 5 `contradicts`, 2 `not_visible_in_model_frames`, 2
`cannot_judge`, and 0 `supports`. Reviewer identity/independence and semantic
validity remain unverified. This is useful exploratory negative feedback but
does not satisfy the current review cohort or a direct-faithfulness gate.

### Next action

Do not tune more bias or ablation settings on these same three targets: both
finite-bias polarities and direct ablation are post-hoc and show no selective
positive effect. The next useful experiment should use a newly frozen,
independent cohort and a model-generated explanation target, with the
annotation-linked object-pixel removal and a salience/area-matched sham fixed
before outcomes. Until that target can be built and validated, these likelihood
proxies remain exploratory and cannot support an SSS/direct-faithfulness claim.
