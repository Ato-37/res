# 2026-10-31 WP4: Logit Lens on evidence-interval tokens (Qwen2.5-VL-7B 4-bit, TVBench Action Sequence)
Plan: `2026-10-31_tvbench_lens_evidence_plan.md`. Receipt (local-only) `data/private_nextgqa/tvb_lensev_full_20261031.json`; derived `results/tvbench_lens_evidence_summary_20261031.json` (`scripts/analyze_lens_evidence.py`, run once). Test 100 requested, 97 analysed; translators trained on 40 disjoint videos.
## Observations
- Primary (mid band, layers 9-20; evidence-group mean log-mass of gold-only minus foil-only words, evidence minus other groups): +0.032 [-0.042, +0.103] (bootstrap), d_z 0.087, t p=0.39, sign-flip p=0.39, Wilcoxon p=0.15; positive in 58.8 % of videos (Wilson 48.8-68.0 %).
- Late band: +0.035 [-0.049, +0.115]; excluding group-0 evidence (n=58): -0.051; baseline-correct only (n=59): +0.050. Layer profile is flat (+0.02 to +0.04 in all 28 layers).
## Interpretation
No evidence that evidence-interval tokens carry more gold-option-word information than other tokens (null/inconclusive; the CI is not shown to be below any equivalence bound). Limits: tokens mix context so the lens is imperfect, group-0 position confound, option words only, one task, 4-bit model, no TOST prespecified. Not explanation faithfulness.
