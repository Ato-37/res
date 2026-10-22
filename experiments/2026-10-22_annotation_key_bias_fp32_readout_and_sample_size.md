# fp32 readout check and sample-size plan for QA-annotation key bias (2026-10-22, local)

## Purpose and decision context
User decision (2026-10-22): the main limitation is the number of videos per experiment, and the confirmatory design
must NOT require human review. The primary endpoint is therefore the gold-annotation-based, label-free one already
implemented in `run_nextgqa_annotation_key_bias.py` (five-choice gold-minus-baseline-best-foil margin under
annotation-interval key bias minus equal-count disjoint hash sham). It measures dependence on the QA time annotation,
not explanation faithfulness; claims stay scoped to that.

## Finding 1: bf16 readout quantization (verified)
Both earlier val cohorts (n=20 + n=20) have per-row contrasts that are all multiples of 0.125 nat (smallest nonzero
|x| = 0.125). That equals bf16 spacing for logits in [16, 32). The observed mean effects (+0.0125, +0.025) are 1/10-1/5
of one quantum. The base runner computes `out.logits[0,-1].float()`, i.e. it casts after bf16 rounding.

## Check: fp32 lm_head readout on the identical cohort-1 rows (exposed, not confirmatory)
- `scripts/run_nextgqa_annotation_key_bias_fp32_readout.py` wraps the unmodified base runner and replaces only
  `lm_head.forward` with an fp32 matmul of the same bf16 final hidden state and weights, identically for all conditions.
  Base runner and its receipts' hashes are untouched.
- Preflight re-run reproduced the original lock byte-for-byte (SHA-256 prefix 3d49b789...). Smoke 3/3 forwards,
  full 20/20 rows, 60/60 forwards, no failures or replacement; same 20 media in the same order as the bf16 receipt.
- Results (annotation-bias minus sham, nat): bf16 mean +0.0125, SD 0.1140; fp32 mean +0.0194, SD 0.0798,
  descriptive t interval [-0.0180, +0.0567] (includes 0), 10 positive / 10 negative, 20 distinct values (min nonzero
  |x| 0.0002), row-wise correlation with bf16 0.83. Annotation-minus-baseline fp32 mean -0.0040 (SD 0.0730);
  sham-minus-baseline -0.0234 (SD 0.0767).
- Interpretation: the fp32 readout removes the quantization and cuts SD by ~30%, so it roughly halves the required n.
  It does not create an effect: the mean stays small and the sign split is even. Readout change = measurement change;
  results from the two readouts must not be pooled. Remaining variance is between-video.

## Sample-size plan (proposal; delta_min needs user confirmation)
Two-sided alpha 0.05, power 0.80, n = (2.8 * SD / delta)^2, one value per video (video is the unit).
| delta (nat) | n at SD 0.080 (fp32, observed) | n at SD 0.114 (conservative, bf16) |
|---|---|---|
| 0.05 | 20 | 41 |
| 0.025 | 80 | 163 |
| 0.0125 | 320 | 652 |
SD comes from 20 rows (itself uncertain). Proposed: one prespecified confirmatory val cohort of N=160 fresh videos
(covers the conservative SD at delta 0.025), fp32 readout, primary contrast annotation-minus-sham, single analysis, plus
an equivalence (TOST) statement with margin delta_min so a null can be reported as "effect below delta_min". Candidate
pool: NExT-GQA val has 567 videos / 3358 QA, test 990 videos / 5553 QA; the local `NExTVideo.zip` (24 GB, 5600 files)
holds the media; 1,034 raw-media hashes plus 40 used val videos must be excluded. Compute is small: 20 rows = ~4 min.

## Limits / next action
- Cohort 1 is exposed data and a measurement check only; no hypothesis test was run.
- Next: freeze delta_min and N before selection, build the outcome-blind val lock for N=160 (preflight is CPU-only),
  smoke, then one full run; analyze with the existing analyzer extended for fp32 receipts and an equivalence test.
- Faithfulness status unchanged: 直接評価 `未実施`; this endpoint is an annotation-dependence proxy.
