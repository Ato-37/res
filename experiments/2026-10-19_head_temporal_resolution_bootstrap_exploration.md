# Head temporal-resolution and bootstrap exploration (2026-10-19)

## Purpose

Use available GPU time for bounded exploratory head analysis, as requested,
without reopening the closed preregistered head-faithfulness test. This evaluates
only how an unsupervised attention-shape ranking changes across temporal
resolutions and across resamples of one fixed local video cohort. It does not
measure explanation faithfulness, evidence localization, or causal prediction
dependence.

## Setup and provenance

- Cohort: 61 rows from `data/nextgqa_samples_big.json`; source manifest SHA-256
  `19407828202f7f69c91f97c86051f1e57fe84141360d2c5800d240380284fa60`.
- Model: local Qwen2.5-VL-7B-Instruct, explicit revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
  eager attention. Hardware: RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128.
- The earlier 8-frame and 16-frame full scans each requested 61 rows and
  completed on 58; three rows failed in each condition, with matching recorded
  frame-read errors. No imputation or replacement was made. The separate 16-frame
  bootstrap collection also completed 58/61, but recorded only `RuntimeError`
  types for its three failures; their causes are unresolved and are not assumed
  to match the earlier run.
- Private 8/16-frame result:
  `data/private_nextgqa/head_temporal_resolution_full_20261019_02.json`,
  SHA-256 `fa633888e86cee5451e4251d4008122d501be2544d9bb0f90d018960985841f5`.
  It is mode `0600` beneath a mode-restricted private data directory. The
  recorded runner and helper hashes were verified against the preserved runner
  and `scripts/head_selection_unsupervised.py`.
- At 16 frames, the scan recorded peak allocated VRAM 15.249 GiB and 312.03 s;
  at 8 frames, 8.526 GiB and 55.963 s. These are run-recorded allocator peaks,
  not system-wide maxima.
- Bootstrap used per-row attention metrics from the fixed 16-frame scan and
  resampled unique media-content clusters (58 independent successful media
  clusters), 1,000 replicates, seed `20261019`. It is post-selection,
  descriptive exploration because the full-sample ranked candidates had
  already been inspected.
- Private bootstrap files are local only:
  `data/private_nextgqa/head_bootstrap_full_20261019_01.json` (SHA-256
  `6ecbccda0bf41364c960517cb47664f01753f62f4a14b25c20cea9277d12385e`),
  `.npz` (SHA-256
  `2f4c44c2375f0f33d6acdfa826ea50365335528b8d69682bf726678081541e15`),
  and analysis JSON (SHA-256
  `5bce9d941b81fd6b17dc6f1472e5fc8c825e4426b1708a24ad56abc6b7840874`).
  All three are mode `0600`. A second analysis invocation with the same seed
  and 1,000 replicates was byte-identical to the first.

## Observations

- The independently recomputed 8-vs-16 frame top-20 overlap was 9 of a union
  of 27 candidates (Jaccard `0.3333`). Fourteen non-sink survivors were shared;
  their average-tie Spearman rank correlation was `0.9473`. The selected
  survivor counts were 16 at 8 frames and 28 at 16 frames. These are
  descriptive results conditional on this ranking/filter and cohort.
- In the 16-frame cluster bootstrap, the median non-sink survivor count was 26.
  The highest-ranked candidate appeared in the top 20 in all 1,000 resamples;
  the second-ranked did likewise, and the next three frequencies were 0.999,
  0.996, and 0.988. The exact table is retained only in the private analysis
  artifact.
- The archived analysis reproduced byte-for-byte under an independent second
  invocation. Six focused unit tests passed. The temporal-resolution audit
  independently verified the source hashes, effective/failure counts, all 784
  head inventory, top-20 ordering, overlap, and average-tie rank correlation.
- One audit attempt initially used the wrong helper path and correctly failed
  its hash gate; rerunning with the helper actually imported by the runner
  passed. The failure was an audit-command argument error, not a result
  mismatch.
- A second independent audit of the bootstrap summary and NPZ rehashed the
  listed media and reconstructed all 1,000 resamples; the complete saved top-40
  table and median eligible count matched. It also verified the mode-0600 files,
  mode-0700 parent, and that no raw video, path, or question text was in those
  outputs.
- A local, aggregate-only CPU recheck of the three failed bootstrap rows used
  the same 16 uniformly sampled frame indices. All three again failed to decode
  at least one sampled frame. No path or row identifier was emitted. This
  supports a repeatable media-decoding limitation for these rows, though it
  cannot show whether the videos are corrupt versus a decoder/container issue.

## Interpretation and limits

This supports only that, under one heuristic and one local cohort, some
attention-shape ranks were descriptively stable across resampling and the
selected shared survivors had high rank concordance across 8/16-frame runs.
The candidate list is selected by the same attention data, has no independent
validation set, and receives no inferential CI or hypothesis test. Attention
entropy and video mass are not evidence attribution. The bootstrap is not a
population guarantee, and its three repeatable frame-decode failures could bias
the successful-row result. The preregistered head-faithfulness result remains negative and unchanged:
79/79, estimate `-0.0756` nat, 95% t CI `[-0.1578, +0.0065]`, below its
`+0.25` nat practical threshold. This exploratory work does not reopen that
closed test.

Direct explanation-faithfulness progress is unchanged: 0/3 formal claim reviews,
0/15 verified clock/frame correspondences, 0/3 prediction-link judgments, and
0/3 direct scores. There is no direct effect estimate, interval, or test.
Rights status remains unverified; all video-derived metrics and artifacts stay
local and are not committed or published.

## Next action

Do not spend more GPU on the same all-head ranking or bootstrap: all 784 heads
were already enumerated, and repeating this outcome-blindness-limited exercise
would add little. Continue the direct-faithfulness handoff and independently
audit the current review gate. If a new head experiment is proposed, it must
have a distinct question, fixed before seeing its outcomes, and must not be
framed as confirmation of the failed preregistered branch. Use idle GPU for
such exploration only when that discriminating purpose is clear.
