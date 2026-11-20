# Map the Flow reproduction vs. paper: what differs and by how much (2026-11-10)

Scope: LLaVA-NeXT-7B-Video-FT (official checkpoint), official code (fast copy, mask_num_heads=1), TVBench 5 tasks and TOMATO, local-only data. All our numbers come from per-question outputs under ~/external/outputs (not tracked); paper numbers are read from the Japanese translation of arXiv 2510.13251 (v2) and have not been re-checked against the PDF tables. Descriptive comparison, no hypothesis test unless stated.

## Metric correction (important)
Paper Table 2 reports the drop of accuracy when cross-frame attention is blocked in layers 1-16 ("at least 18% lose the correct answer"). Earlier logs (2026-10-29, 2026-10-30) compared it with the mean relative change of the answer probability and a "< -10%" criterion, which is a different quantity. The correct comparison is the share of baseline-correct questions whose answer changes. From the existing scaled run (table2_first_half, official code, window bottom_up 15-16, same records):

| Task | n correct | ours: answer lost (Wilson 95% CI) | paper | diff (ours - paper) | paper inside our CI |
|---|---|---|---|---|---|
| Action Sequence | 95 | 25.3% [17.6, 34.8] | 20.2 | +5.1 | yes |
| Moving Direction | 40 | 35.0% [22.1, 50.5] | 44.8 | -9.8 | yes |
| Object Count | 78 | 60.3% [49.2, 70.4] | 60.8 | -0.5 | yes |
| Scene Transition | 92 | 19.6% [12.7, 28.8] | 18.0 | +1.6 | yes |
| Action Antonym | 93 | 22.6% [15.3, 32.1] (21/93; added 2026-11-11, 150 q sampled, seed 42) | 24.1 | -1.5 | yes |

Observation: all four reproduced values fall inside their intervals around the paper's values; the ordering (Object Count > Moving Direction > Action Sequence ~ Scene Transition) is the same. The former "I1 not met" verdict came from the wrong metric and should be read as superseded. Inference limit: intervals are wide (n=40-95), so this shows consistency, not equality. Whether the paper counts exactly "answer changed among baseline-correct" is inferred from its text, not verified in code.

## Table 3 (effective-pathway pruning), accuracy %
| Benchmark | arm | ours | paper (7B) | diff |
|---|---|---|---|---|
| TVBench, 5 tasks pooled (n=1322) | full attention | 53.33 | 51.5 | +1.8 |
| | effective pathway | 51.36 | 51.2 | +0.2 |
| | random 58% | 43.04 | 40.1 | +2.9 |
| | pathway - full | -1.97 [-4.08, +0.15], McNemar p=0.081 | -0.3 | -1.7 |
| TOMATO (n=1484) | full attention | 28.37 | 30.2 | -1.8 |
| | effective pathway | 27.09 | 29.2 | -2.1 |
| | random 58% | 22.44 | 23.1 | -0.7 |
| | pathway - full | -1.28 [-3.30, +0.74], McNemar p=0.234 | -1.0 | -0.3 |
Observation: pathway stays close to full attention and above random in both benchmarks; the largest gap is the TVBench pathway-minus-full contrast (-1.97 vs -0.3), whose interval still contains -0.3. Our TOMATO baseline is 1.8 pt below the paper's (video decoding/frame sampling not verified identical). Not established: whether the TVBench gap is real (no prespecified margin, no equivalence test).

## Layer-wise curves vs Table E(b) (Action Antonym, 7B)
Ours: official code, centre window k=9, 40 q sampled (28 baseline-correct), mean relative change of probability (%) averaged over 5-layer bins; paper uses 5-layer bins (window size of the paper's Table E not verified, so only qualitative):
| Interaction | bin | ours | paper |
|---|---|---|---|
| cross-frame | L1-5 / 6-10 / 11-15 / 16-20 | -4.8 / -13.6 / -9.4 / -0.4 | -4.2 / -11.1 / -6.3 / -0.2 |
| video->question | L1-5 / 6-10 / 11-15 / 16-20 / 21-25 | -1.8 / -5.6 / -8.8 / -9.0 / -5.8 | -3.9 / -15.1 / -21.5 / -5.6 / -0.2 |
Observation: cross-frame matches within about 3 pt per bin. Video->question is flatter and later in ours (peak at L11-20, still -5.8 at L21-25) than in the paper (peak L11-15, ~0 by L21-25), and about 2-3x smaller at the peak. Candidate explanations (untested): different window (k=9 vs the paper's bins), n=28, a different seed/sample. The k=5 and k=1/13 runs in the appendix queue address the window explanation.

## Other remaining deviations
- Layer 1 and 32: ours ~0 (window clipped at edges); the paper gives no layer-1/32 statement.
- Logit Lens (7B, Action Sequence, 185 questions): temporal top-1 share 0.12/0.32/0.35% early/mid/late; paper uses 13B and reports a larger rise; direction only.
- Base ImageLLM comparison, Fig 6 maps, Fig 9/10 tracing, open-ended VCGBench, long-video benchmarks, other models: not run (tokenizer loading, data or VRAM).
- Official `vql-to-ql` target also blocks last->last (the paper's note 2 mentions it only in the pruning description); relevant when interpreting that run.

## Housekeeping (2026-11-10)
Freed about 13 GB: removed ~/.cache/pip (1.7 GB) and the redundant ~/external/tomato_dl/videos.zip (12 GB, extracted copy verified: 1417 videos). Kept: Hugging Face blobs (23 GB, contain the pinned Qwen2.5-VL checkpoint), ~/external/models, venvs, outputs. Linux page cache was already small (free memory 20 GiB), nothing to drop.

## Appendix runs finished 2026-11-10 (analysed 2026-11-11; Action Antonym, 7B, official code, local-only outputs ~/external/outputs/appendix_20261110)
- Table 2, AA: first-half block (layers 1-16) changed the answer for 21/93 baseline-correct questions = 22.6% (Wilson 95% [15.3, 32.1]); paper 24.1 lies inside. All five tasks are now consistent with the paper's Table 2 (intervals wide).
- Table B-like, AA: second-half block (layers 17-32) changed 1/93 = 1.1% [0.2, 5.8]: cross-frame dependence is essentially confined to the first half (direction as in the paper; I did not check the paper's AA second-half number).
- Window sensitivity (cross-frame, 40 q sampled, 28 baseline-correct videos; mean relative change of probability %, video = unit, no CI computed yet): k=1: L6-15 -1.1, L16-20 +0.1 (single-layer block is nearly invisible, one outlier -5.1 at L7 1-based); k=9: L6-15 -11.4; k=13: L1-5 -10.5, L6-15 -13.6, L16-20 -1.3 (broader and starts earlier, edge layers already -7). Observation: the effect grows with window size and its location stays in L1-17; the single-layer result is consistent with redundancy across neighbouring layers (inference, untested). The paper's Table E window size remains unverified.
- G4 progressive blocking: the first run (appendix_20261110/G4_progressive_bottom_up) was stopped at user request and is incomplete; rerun started 2026-11-11 under ~/external/outputs/appendix_g4_20261111 (bottom_up then top_down, 40 q).

## G.4 progressive blocking finished (2026-11-11; AA, 7B, window growing from one end, 40 q sampled, 28 baseline-correct videos; outputs ~/external/outputs/appendix_g4_20261111b, local-only)
Mean relative change of answer probability (%), layers 0-based, "blocked through layer L". Descriptive, no CI computed.
- Front-to-back (bottom_up): ~0 through L5, -7 -> -15 over L6-L11, plateau about -16 from L12 to L31.
- Back-to-front (top_down): about -16 through L5, back to 0 by L12; ~0 for L13-L31.
Observation: the cross-frame effect is located in about L6-L12; adding layers beyond L12 changes nothing from either side. Consistent with the window k=9 and k=13 runs. An earlier attempt (appendix_g4_20261111) died with no output (likely WSL restart) and was discarded.
