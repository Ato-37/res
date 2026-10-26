# Temporal-order scorer prompt-binding repair (2026-10-21)

## Purpose

Audit all local completed receipts emitted by
`run_saved_claim_frame_permutation_attention_smoke.py` and repair the rows
where model-generated claims recorded `generation_prompt_version=v2_short`
but the scorer had used the legacy v1 suffix. The measurement is an internal
visual-key-bias × temporal-order likelihood interaction; it is not direct
faithfulness.

## Defect and affected scope

The original runner imported `PROMPT_SUFFIX`, which
`generate_nextgqa_claim_review_pilot.py` defines as the v1 legacy suffix. The
three `v2_short` packets below had row-level `generation_prompt_version` set to
`v2_short`. Thus the earlier original-versus-reversed scores used a mismatched
question suffix. The old receipts are retained, but their v2_short estimates
are invalid for that generation-prompt condition.

An inventory of completed local receipts found 11 previously scored valid
rows from four v2_short packets/cohorts requiring correction:

1. Existing exploratory v2_short val packet, 3 rows.
2. Temporal-order val smoke plus remaining packet, 3 rows.
3. Local-history-screened val preflight packet, 3 rows.
4. Temporal-order test smoke and remaining packet, 2 valid scored rows; the
   packet contained one additional structurally invalid generated claim, which
   was not scored and was not replaced.

Other inventoried receipts used v1 generation prompts and were not affected by
this specific mismatch. Their results were not rerun here.

## Repair

The scorer now resolves `PROMPT_SUFFIX_BY_VERSION[record.generation_prompt_version]`
for each row and applies the same resolved suffix to original and reversed
frame batches. It records the prompt version, suffix SHA-256, resolver source
hash, and all prior source hashes in the private receipt. Unknown versions
fail closed. A CPU regression test covers v2_short selection and rejection of
unknown versions. The focused runner suite passed **12/12**. An independent
review of the changed call path and packet binding passed before the GPU runs.

All corrected runs used Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
eager attention, 16 sampled frames, and the fixed `+ln(2)` visual-key bias at
layers 5–16. Each row used four teacher-forced forwards: original/reversed
frames under baseline and biased attention. All were local, protected inputs
and receipts remained mode 0600, and no media, claim, or question text was
published.

## Corrected results

| Fixed cohort / packet | Corrected per-row difference-in-differences (nat/token) | Mean | Positive rows | Completed |
|---|---|---:|---:|---:|
| Existing exploratory v2_short val, n=3 | +0.004429, +0.057039, +0.003445 | +0.021637 | 3/3 | 12/12 forwards |
| Temporal-order v2_short val, n=3 | −0.016760, +0.118283, +0.058883 | +0.053469 | 2/3 | 12/12 forwards |
| Local-history-screened v2_short val, n=3 | +0.011883, +0.063553, +0.050534 | +0.041990 | 3/3 | 12/12 forwards |
| Temporal-order v2_short test, n=2 valid of 3 generated | −0.021464, −0.016327 | −0.018895 | 0/2 | 8/8 forwards |

Across these four **separate** fixed cohorts, all 11 previously scored valid
rows were rescored and all **44/44** forwards completed with no substitutions.
The repaired row-level source video and claim hashes join exactly to their
prior receipts; all new source hashes were recomputed from the current files.
Peak allocated VRAM was 7.26–7.42 GiB. Summed runner elapsed time was 273.0
seconds; this excludes interaction and tool-polling overhead.

For comparison only, the invalid v1-suffix interactions had means −0.027599
(3/3 negative), +0.038378 (2/3 positive), −0.003526 (1/3 positive), and
+0.049029 (2/2 positive), respectively. The corrected values change direction
in three of four cohort summaries. This demonstrates that prompt binding
materially affects the endpoint. Do not average old and corrected results or
pool the corrected cohorts. Each cohort was outcome-exposed by the prior
scoring, including the history-screened packet whose old wrong-prompt scores
had already been inspected; this repair is not confirmatory evidence.

No confidence intervals, hypothesis tests, p-values, or multiplicity-adjusted
claims are reported. These are selected finite-cohort diagnostics with
different packets/splits; the second test cohort has only two scorable rows.
The interaction is conditional target-text likelihood sensitivity to temporal
block reversal. It does not establish claim truth, claim-support, causal use of
the described evidence, or explanation-faithfulness improvement.

## Faithfulness progress

- Direct claim-to-frame faithfulness: **0/3 reviewed claims, 0/15 clock/frame
  correspondences, 0/3 prediction links, 0/3 direct intervention scores**.
- This repair is **補助診断のみ**: it corrected prompt consistency for an
  internal sensitivity proxy, not the semantic support labels required by the
  direct endpoint.
- Human review remains waived as a collection prerequisite for exploratory
  local runs, but no human judgment was inferred. Dataset/model rights remain
  unverified; local-only handling continues under the user's authorization.

## Reproduction and next action

The private-receipt analyzer rechecks exact old/new row joins, all current
source hashes, prompt suffix hashes, forward counts, and difference-in-
differences. Run it separately per frozen cohort; do not mix packet families:

```bash
.venv/bin/python scripts/analyze_promptbound_saved_claim_frame_order.py \
  --new data/private_nextgqa/frame_order_promptbound_v2short_smoke_20261021_02.json \
       data/private_nextgqa/frame_order_promptbound_v2short_row1_20261021_01.json \
       data/private_nextgqa/frame_order_promptbound_v2short_row2_20261021_01.json \
  --old data/private_nextgqa/frame_permutation_attention_v2short_val_smoke_20261021_01.json \
       data/private_nextgqa/frame_permutation_attention_v2short_val_row01_20261021_01.json \
       data/private_nextgqa/frame_permutation_attention_v2short_val_row02_20261021_01.json
```

Apply the same argument shape with matching new/old receipt paths for the
other three cohorts. The next scientific step is not more variants on these
rows: either obtain actual claim-to-frame judgments for the exact reviewed
artifact, or design a new prospectively fixed direct-support cohort. Until
then, report the prompt-corrected temporal-order sensitivity only as an
auxiliary diagnostic.
