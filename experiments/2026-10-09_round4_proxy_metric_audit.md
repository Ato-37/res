# 2026-10-09 iterative round 4: attention-overlap proxy audit and literature update

## Purpose

Clarify exactly what the existing attention statistic measures before interpreting it as grounding or faithfulness, and extend the prior-art search specifically around intervention, temporal evidence localization, and claim/evidence faithfulness.

## Code audit and change

The previous code averaged the last prompt-position attention over selected/all layers and heads, summed token weights within each temporal grid chunk, normalized over video chunks, selected the single peak chunk, and counted a hit if that chunk's coarse time interval intersected any ground-truth interval. The last prompt position predicts the first generated token; its attention pattern is not itself a causal test of the generated claim.

The resulting value is a **top-attention-chunk temporal-intersection proxy**. It has material limitations:

- It discards the mass and shape of the rest of the attention distribution after choosing one peak.
- It is sensitive to coarse temporal chunk width and annotation boundary overlap; any positive intersection counts as a hit.
- It is not evidence that removing/changing that interval changes the answer, and attention weights need not be faithful explanations.
- Layer/head averaging can obscure heterogeneous pathways.
- Invalid or empty gold intervals must not silently count as misses in the denominator.

Added a pure, unit-tested attention_overlap_proxy helper. It validates finite positive-duration intervals and finite nonnegative attention, rejects empty gold annotations, normalizes chunk weights, and exposes peak index plus the exact intersection decision. The CLI now labels the score as a proxy and skips invalid annotations rather than folding them into the evaluated denominator. No inference was run as part of this code change.

## Literature findings: direct overlap and boundaries

1. **CRA: Cross-modal Causal Relation Alignment for Video Question Grounding (CVPR 2025).** The official paper describes VideoQG as answering and localizing supporting segments jointly, and introduces explicit front-door visual and back-door language interventions to reduce spurious cross-modal correlations. It reports NextGQA results. This is very close to the broad proposed objective of causal intervention plus grounding and means that objective cannot be presented as new. CRA is a trained weakly supervised feature-alignment architecture, not the same intervention as test-time residual-stream steering in a frozen Qwen2.5-VL, and its reported grounding/accuracy ablations do not by themselves establish explanation faithfulness under a selective claim-dependence test. Source: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Cross-modal_Causal_Relation_Alignment_for_Video_Question_Grounding_CVPR_2025_paper.html

2. **TA-AE: Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering (NeurIPS 2025).** The official abstract explicitly studies internal activation engineering in VideoLLMs, and the paper includes Qwen2.5-VL results. This rules out "first activation intervention in a VideoLLM." Its target is hallucination reduction, not the exact proposed joint localization/selective-claim-dependence outcome. Source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html

3. **CaST-Bench (CVPR 2026).** This benchmark evaluates causal-chain-grounded spatiotemporal reasoning and annotates multiple temporal evidence segments plus box tracks. It makes "causal video answers with temporal evidence" an established benchmark target, though it is not itself an internal intervention method. Source: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_CaST-Bench_Benchmarking_Causal_Chain-Grounded_Spatio-Temporal_Reasoning_for_Video_Question_Answering_CVPR_2026_paper.html

4. **INFACT (ACL 2026).** It separates video-evidence faithfulness from world-knowledge factuality and probes robustness under evidence corruption and temporal intervention. This suggests a useful diagnostic distinction and a potential externally meaningful outcome, but does not test internal activation intervention or interval-level explanation faithfulness. Source: https://aclanthology.org/2026.acl-long.2062/ (official ACL Anthology record; DOI 10.18653/v1/2026.acl-long.2062).

5. **FaithLM (EACL 2026).** It operationalizes explanation faithfulness via a contradiction intervention that should shift the prediction, then optimizes prompts/explanations using that score. This supports measuring behavioral dependence rather than relying on explanation plausibility or attention. It concerns text-only LLM explanation settings, so video adaptation and confounds from visual/OOD interventions require validation. Source: https://aclanthology.org/2026.eacl-long.177/

6. **Map the Flow (ICLR 2026).** It analyzes VideoLLM information pathways and reports early-to-middle cross-frame interactions followed by middle-layer video-language integration. This is mechanistic analysis that can inform where to define a hypothesis, but does not validate a grounding/faithfulness intervention or justify post-hoc layer selection. Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/537b3416a0c8035794a12b3fa596f941-Abstract-Conference.html

7. **IGV: Invariant Grounding for VideoQA (CVPR 2022).** It learns causal-scene grounding and trains with counterfactual replacement of complement scenes; it predates current VideoLLMs and is training-time rather than test-time activation steering. It still establishes that intervention-based causal grounding in video QA is not novel in general. Source: https://openaccess.thecvf.com/content/CVPR2022/html/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.html

8. **Seeing but Not Believing (ICLR 2026).** This paper diagnoses that VLMs may attend to correct image evidence yet answer incorrectly, and introduces a training-free inference-time selective attention-masking intervention that improves accuracy across model families including Qwen. It is image-focused and not temporal grounding, but it directly overlaps with the idea of using internal evidence signals to improve answer use. Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html

9. **STEAR (arXiv:2604.03045, 2026 preprint).** This is an especially close recent method: it selects token-conditioned spatiotemporal evidence from middle layers, reinjects evidence into those layers, and uses temporally perturbed patch-level counterfactual decoding in later layers. The paper reports Qwen2.5-VL and faithfulness/temporal consistency outcomes. It is a preprint, not counted as a top-conference publication in this review, but it substantially overlaps the intended video-specific method and must be treated as direct prior art. Source: https://arxiv.org/abs/2604.03045

10. **Causality Matters: How Temporal Information Emerges in Video Language Models (AAAI 2026).** It traces temporal information through inter-frame attention and query-token integration, then proposes staged cross-modal attention and a temporal exit mechanism. It is mechanistic analysis and efficiency work rather than the proposed intervention, but narrows any novelty claim about causal temporal pathways. Source: https://ojs.aaai.org/index.php/AAAI/article/view/37856

## Research decision

The broad objective is not novel: CRA combines causal intervention and VideoQA grounding on NextGQA; TA-AE performs video-LLM activation engineering including Qwen2.5-VL; and STEAR is a close video-specific inference-time evidence intervention with temporal counterfactual decoding (currently an arXiv preprint). Seeing but Not Believing also uses internal visual evidence signals for inference-time accuracy improvement. Consequently, the current evidence does not support a novel method claim. A possible thesis contribution may instead be a tightly controlled replication/generalization study of an explicitly chosen method and prespecified dual outcome on a fixed cohort, or a validated measurement study; this requires advisor-level scope approval and an exhaustive comparison before commitment. Do not assume a novel combination of known modules is itself a contribution. Do not equate attention overlap, evidence sufficiency, robustness, and explanation faithfulness.

## Verified checks

- 15/15 CPU unit tests passed across attention-proxy semantics, time mapping, and data audit.
- Python byte-compilation and git diff whitespace checks passed.
- The updated one-sample synthetic Qwen forward passed after the proxy-label assertion was updated; frames 4/4, sample 1/1, 4-bit, peak allocated VRAM 6.446 GiB. Synthetic overlap was discarded. No dataset video was used.
- A first rerun caught a stale smoke assertion expecting the former summary label; the model subprocess had completed but the wrapper failed its label assertion. The wrapper was updated, and the full smoke then passed.

## Next action

1. The INFACT record is now verified at ACL Anthology ID 2026.acl-long.2062.
2. Read CRA and TA-AE methods/supplements in a structured intervention/estimand/metric matrix; compare the close STEAR preprint and image-domain ICLR intervention, and do not frame an intervention method as novel from this scan.
3. Implement and test a claim/annotation-fit gate. The ground-truth interval is not automatically evidence for every question-answer claim.
4. Consider a reproducibility/generalization thesis rather than a novel-method claim, subject to advisor approval; freeze outcomes and controls before any pilot. Keep the closed head/layer/alpha search closed.
