# QTA 16/32-frame selector resolution stability (local result)

## Purpose and status

Measure how much the untreated question-token-attention selector's temporal
support changes when the same deterministic NExT-GQA test cases are encoded
with 16 rather than 32 sampled frames. This is an exploratory selector
resolution diagnostic. It is not an explanation-faithfulness score, causal
effect, validation result, or replacement for the frozen 32-frame condition.

The run and this result log remain local because they derive from staged
protected video/question inputs. Rights status remains unverified. No video,
question text, answer, sample identity, or selected group was written to the
result. The private JSON contains only aggregate hashes and metrics indexed by
fixed ordinal; it has mode `0600` under a mode `0700` directory.

## Frozen setup

- Cohort: first three eligible candidates from `choose_videos`, in its existing
  deterministic order; one qid-sorted eligible question per video. Cohort was
  fixed before decoding and no failed row would be replaced.
- Conditions: same video/question at 16 and 32 endpoint-inclusive sampled
  frames; PTS-grid tolerance 50 ms; one untreated forward per condition.
- Selector: question-token attention averaged across heads and layers 5–16;
  select `ceil(0.10*T)` complete processor temporal groups. Qwen temporal patch
  size was 2 and was checked across the image processor, video processor, and
  model config. Selected group intervals use the decoder PTS midpoint-cell
  support, normalized to each video's sampled endpoint span.
- Primary statistic: mean per-video Jaccard between the two selected interval
  unions. Supplementary statistic: Wasserstein-1 between selected patch-center
  distributions. No p-value, population CI, or threshold was prespecified or
  used.
- Model/runtime: Qwen2.5-VL-7B-Instruct pinned revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute
  and BF16 model dtype, CUDA 12.8, PyTorch 2.11.0, Transformers 5.18.0,
  RTX 5070 Ti. Seed: 20261018. Before model load, free VRAM was 14.66 GiB;
  after load it was 8.68 GiB. The recorded whole-run elapsed time was 57.14 s,
  including model load and local decode/preprocessing; attention-capture and
  analysis time was 19.49 s.
- Provenance: private run JSON
  `data/private_nextgqa/qta_resolution_stability_20261018_2137/full_auditable.json`,
  SHA-256 `f63fc9c0c45d87e2aaa6bc0b60e3621a5defed1d379f79840a1c7596635628fb`.
  It records source/dataset/model/processor hashes, the candidate-exclusion
  manifest digest, sampled PTS/RGB aggregate hashes, counts, and ordinal-only
  pair metrics. Runner SHA-256 in the manifest matched the current source.

## Verified results

Requested/effective cohort: **3/3 video pairs**. Requested/completed forwards:
**6/6**. Failed pairs/forwards: **0/0**. The local JSON was independently
read back; the three ordinal metrics reproduce the aggregate estimates to
floating-point precision.

| Fixed pair ordinal | Interval Jaccard | Selected-center W1 |
|---:|---:|---:|
| 1 | 0.3864 | 0.0579 |
| 2 | 0.3570 | 0.0622 |
| 3 | 0.2935 | 0.1613 |
| Mean | **0.3456** | **0.0938** |

Jaccard range was 0.2935–0.3864; W1 range was 0.0579–0.1613. A separate
one-pair smoke completed 2/2 forwards and returned Jaccard 0.3864 and W1
0.0579; the subsequent three-pair run reproduced the same aggregate values
as the first full run before per-pair provenance fields were added.

## Interpretation and limits

On these three fixed cases, the attention selector's selected time support
overlapped only partially between 16- and 32-frame inputs. This is evidence
that the selected support is resolution-sensitive under this exact processor,
model, prompt and selector setup. It does not isolate selector noise: changing
frame count also changes visual tokenization and the attention distribution.
The normalized support is relative to the sampled endpoints, not verified
full-video-duration coverage. The cohort is tiny and fixed; no inferential
claim follows. Attention selection alone says nothing about whether a human
explanation matches valid video evidence or whether a model prediction depends
on that evidence.

Direct faithfulness remains **0/3** reviewed claims, **0/15** independently
verified player/browser-to-decoder frame pairs, **0/3** prediction-link
judgments, and **0/3** direct scores. There is still no direct `DeltaF`, CI,
or statistical test. Real human review remains gated on independently pinned
trust root, real role-to-key assignments, signed rosters, review-only receipt,
witnessed unopened handoff, and passing pre-open audit. The existing candidate
inventory remains non-authorizing.

## Next action

Keep the 32-frame primary configuration fixed. Continue human-free reliability
checks that separate deterministic repeatability from resolution sensitivity;
do not tune frame count or selector settings from these observed outcomes.
Advance human review only after genuine independent role assignments and the
documented release ceremony are supplied; do not simulate identities or
review judgments.
