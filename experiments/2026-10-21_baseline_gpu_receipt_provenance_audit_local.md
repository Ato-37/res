# Baseline GPU smoke receipt provenance recheck (2026-10-21 JST, local)

## Purpose

Determine whether a new compatibility GPU run is needed before building the
real-data path. Revalidate the prior three-forward smoke from its result bytes
and source hashes instead of repeating a completed synthetic run.

## Verified result

- Artifact: `results/synthetic_baseline_only_qwen_gpu_20261019_04.json`, mode
  `0600`, 3,816 bytes, SHA-256
  `e17ad5217d45c747d0dddbaf4b165d10961ec6e1b31542a8912e4cf257dc0d91`.
- The recorded run reports `passed`, 3 requested / attempted / completed
  forwards, 3/3 verified model boundary calls, 3/3 attention captures, RTX
  5070 Ti, 38.646 seconds, peak allocated VRAM 10.724 GiB,
  `faithfulness_result=false`, and `execution_authorization=not_granted`.
- Recomputed all 14 recorded source hashes against the current checkout. Thirteen
  match. `qa_video_batch_builder` differs in the current source because a later
  commit changed it after this run: the result recorded
  `92a6173e...e70b07d`; current file is
  `b978f83d...61c0334`. The old digest exactly matches the file at commit
  `2b34f27` (2026-10-19 07:53 JST), before the run timestamp
  `2026-10-19 09:13 JST`; the changed version arrived in commit `e1f23f1`
  (2026-10-19 20:05 JST). This is explained temporal drift, not a failed run.
- The artifact is already tracked in Git. No new GPU run is justified: repeating
  its three synthetic clips would duplicate existing compatibility evidence and
  would not address the actual review gate.

## Research status and next action

This is valid evidence that the historical pinned model/runtime executed the
synthetic three-row scorer path. It says nothing about real-video faithfulness.
Direct faithfulness remains 0/3 claims reviewed, 0/15 clock/frame
correspondences, 0/3 prediction-link reviews, and 0/3 direct scores, with no
direct effect estimate or inferential result. The actual real-video GPU baseline
still requires trusted role/release inputs, completed claim and frame reviews,
and the exact pre-load transaction. Do not use this synthetic result to bypass
those gates or rerun the same synthetic workload for utilization.
