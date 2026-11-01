# 2026-10-31 WP2-TVB batch A result (window-width expansion, TVBench Action Sequence, Qwen2.5-VL-7B 4-bit)

Plan: `2026-10-31_tvbench_window_expansion_plan.md` (prespecified). Receipt (local-only): `data/private_nextgqa/tvb_window_full_20261031.json`; derived summary `results/tvbench_window_expansion_summary_20261031.json` (script `scripts/analyze_window_expansion.py`, run once).
Counts: 200 requested, 200 recorded, 9 excluded (annotation group budget), 191 analysed, **121 baseline-correct**; 6554 forwards, 3.35 h.

## Primary analysis (condition `eff` = video->text, layers 6-20; per-video OLS of %p change on w/G and o/k)
- Evidence-coverage coefficient (extra drop at equal blocked amount, 0 -> 100 % of the evidence covered): mean **-20.4 pp**, bootstrap 95 % CI [-53.7, +12.6]; t p=0.226, sign-flip p=0.234, Wilcoxon p=0.046; TOST(+-6.7 pp) p=0.79.
- Rule (mean <= -6.7 pp and all three p < 0.05): **not met**; not equivalent to zero either => **inconclusive**.
## Secondary / sensitivity (pre-declared)
- Blocked amount (width/G, per full video): eff -31.1 pp [-56.4, -7.2]; all layers -25.1 [-51.5, -0.1]; non-effective layers -3.5 [-6.1, -1.1]. Blocking more of the video lowers the answer probability, and almost only through layers 6-20.
- Sensitivity with the indicator "window includes temporal group 0": the evidence-coverage coefficient becomes +8.3 pp [-9.5, +27.9] (eff) and +5.7 [-13.7, +26.6] (all), and the group-0 indicator is **-8.5 pp [-15.3, -1.9]** (eff), -9.3 [-16.7, -2.1] (all), -0.5 [-1.2, +0.1] (non-eff). The nominal evidence effect in the primary model is not separable from a position-0 effect.
- Non-effective layers (video->text outside 6-20): evidence-coverage coefficient +1.7 pp [+0.5, +2.9], TOST(+-6.7) p~1e-12 (equivalent to zero); windows in these layers do ~nothing (whole video -1.8 %).
- Descriptive means: whole video -40.0 % (eff), -1.8 % (non-eff), -42.0 % (all layers); width 0.94 windows -25.5 % (eff); width ~0.19-0.38 windows within +-8 %.
## Interpretation (not observation)
1. The effect depends on how much video is blocked (amount) and on layers 6-20; layers outside that band carry almost none: supports the effective-pathway location.
2. Whether the annotated interval matters beyond amount is not established here: the primary estimate is large but very uncertain, and it disappears (changes sign) once the earliest temporal group is controlled. Group 0 itself looks special (blocking windows that include it lowers the answer probability by ~8-9 pp at equal amount), which is a position effect, not necessarily an annotation effect.
3. This agrees with the NExT-GQA computation that the evidence share of the effect (~22 %) equals its share of keys. Limits: one task, 2 options (chance 50 %), 121 baseline-correct videos, contiguous windows, 4-bit model; the evidence start position is not randomised. Not explanation faithfulness.
## Next
Action Localization WP2, Logit Lens v2, batch B (replication, not pooled) and FINE (single-group profile to characterise group 0) are queued. Process note: the first queue start was blocked by a `pgrep -f` pattern that also matched the launching shell; fixed with a PID wait and relaunched at 05:44 JST (about 5 min of idle GPU).
