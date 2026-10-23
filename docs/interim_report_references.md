# 参考にした論文(中間報告用、2026-10-23)

凡例: **[確認]** = 2026-10-23 に arXiv の公式ページで題名・著者・日付を確認。**[記載]** = リポジトリの文献調査([faithfulness methods review](faithfulness_methods_and_internal_interventions_review.md)、[internal intervention review](internal_intervention_for_video_grounding_review.md)、[image-to-video options](image_to_video_intervention_options_20261022.md))の記載に基づく。本文の精読は一部のみで、**引用前に本文で照合すること**。各論文の知見は、その論文の対象・タスクに限られ、本研究の測定とは区別している。

## A. 対象モデルとデータ
1. Bai et al., *Qwen2.5-VL Technical Report*, arXiv:2502.13923 (2025). **[確認]** — 対象モデル。
2. Xiao et al., *NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions*, CVPR 2021, arXiv:2105.08276. **[確認]** — 動画QAデータ。
3. Xiao et al., *Can I Trust Your Answer? Visually Grounded Video Question Answering* (NExT-GQA), CVPR 2024, arXiv:2309.01327. **[確認]** — 質問の根拠となる時間区間の注釈。本研究のラベルなし指標の基盤。

## B. 説明の faithfulness の評価と、指標の限界
4. DeYoung et al., *ERASER: A Benchmark to Evaluate Rationalized NLP Models*, ACL 2020. **[記載]** — rationale の除去・一致による評価の枠組み。
5. Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023. **[記載]** — 反事実編集と再構成のテスト。
6. Siegel et al., *The Probabilities Also Matter*, ACL 2024. **[記載]** — 反事実入力での予測分布の変化を測る動機(binary flip でなく、確率や margin を使う)。
7. Parcalabescu & Frank, *On Measuring Faithfulness or Self-consistency of Natural Language Explanations*, ACL 2024. **[記載]** — 既存テストは内部 faithfulness より出力の自己一貫性を測る、という論点。
8. Lyu et al., *Are self-explanations from Large Language Models faithful?*, Findings of ACL 2024; Lyu et al., *Towards Faithful Model Explanation in NLP: A Survey*, Computational Linguistics 2024. **[記載]**
9. Hooker et al., *A Benchmark for Interpretability Methods in Deep Neural Networks* (ROAR), NeurIPS 2019. **[記載]** — 削除型の評価の落とし穴(分布外入力)。
10. Gomez et al., *Metrics for Saliency Map Evaluation of Deep Learning Explanation Methods*, arXiv:2201.13291 (2022). **[記載]** — Deletion/Insertion の限界。
11. Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018. **[記載]**
12. Menzat et al., *Counterfactual Tests for Measuring Chain-of-Thought Faithfulness in Visual Language Models*, Findings of EMNLP 2026, arXiv:2609.06704. **[確認]** — 画像VLMの反事実による説明 faithfulness の検証。人手レビューに依存しない設計の動機。
13. Ding et al., *Explanation-Driven Counterfactual Testing for Faithfulness in Vision-Language Model Explanations*, arXiv:2510.00047 (NeurIPS 2025 workshop); Ding et al., *EDCT-Bench*, arXiv:2609.17953 (2026). **[確認]** — 説明由来の介入テスト。
14. Liu et al., *On the Faithfulness of Visual Thinking: Measurement and Enhancement*, arXiv:2510.23482 (2025). **[確認]**; Moll et al., *Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations*, ML4H 2025, arXiv:2510.11196. **[確認]**

## C. 内部介入(mechanistic interpretability)と測定の注意
15. Vig et al., *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*, NeurIPS 2020. **[記載]**
16. Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022. **[記載]**
17. Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models*, ICLR 2024. **[記載]** — corruption 法と指標で結論が変わる、という注意。
18. Makelov et al., *Is This the Subspace You Are Looking for?*, ICLR 2024. **[記載]** — 出力が変わっても元の経路とは限らない、という解釈の注意。
19. Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023; Goldowsky-Dill et al., *Localizing Model Behavior with Path Patching*, arXiv:2304.05969. **[記載]**
20. Yeo et al., *Towards Faithful Natural Language Explanations: A Study Using Activation Patching in Large Language Models*, EMNLP 2025. **[記載]** — 本研究に最も近い、説明 claim と因果寄与の整合の評価。

## D. VLM・動画の内部介入とステアリング(介入の設計の参照)
21. Li et al., *The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering* (VISTA), arXiv:2502.03628 (2025). **[確認]** — 残差ストリームのステアリング(画像VLM)。
22. Fan et al., *STEAR: Layer-Aware Spatiotemporal Evidence Intervention for Hallucination Mitigation in Video Large Language Models*, arXiv:2604.03045 (2026). **[確認]** — 時空間証拠を中間層で扱う動画の介入。
23. Zhang et al., *VERHallu: Evaluating and Mitigating Event Relation Hallucination in Video Large Language Models*, arXiv:2601.10010 (2026). **[確認]** — Qwen2.5-VL の中間層で key frame を強調(KFP)。
24. Leng et al., *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, arXiv:2311.16922 (2023). **[確認]**
25. Liu et al., *HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models*, ACL 2026 Findings, arXiv:2508.00553. **[確認]**
26. Wu et al., *SEASON: Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding*, arXiv:2512.04643 (2025). **[確認]**
27. TA-AE (NeurIPS 2025); RIVS (IJCAI 2026); CausalLens (CVPR 2026); Modular Attribution and Intervention (ICLR 2025); VTI (ICLR 2025); Localization Heads (CVPR 2025). **[記載]** — 詳細は [internal intervention review](internal_intervention_for_video_grounding_review.md) の表を参照(書誌は同表のリンク)。

## E. 動画・VLM の faithfulness に近い研究
28. Yang et al., *INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs*, arXiv:2603.11481 (2026). **[確認]**
29. Wang et al., *Evidence-Backed Video Question Answering*, ECCV 2026, arXiv:2607.11862. **[確認]** — 注釈された証拠領域と、ランダムな非証拠領域を比較する診断。本研究の sham 対照と同じ発想。
30. Fan et al., *Chains That See, Answers That Don't: A Multi-Aspect Evaluation Recipe for Forced Chain-of-Thought on Video-MME*, EvalMG @ SIGIR 2026, arXiv:2606.22862. **[確認]**
31. Song et al., *Visual Attention Faithfulness in Vision-Language Models is Heterogeneous*, EMNLP 2026, arXiv:2609.00830. **[確認]**

## 使い方の整理(本研究との対応)
| 論点 | 主な参照 |
|---|---|
| 説明 faithfulness を、人手に頼らず反事実・介入で測る | 5, 6, 7, 12, 13, 29 |
| 指標の限界(削除型、分布外、丸め) | 9, 10, 11, 17, 18 |
| 内部の key・表現への介入(ステアリング) | 21, 22, 23, 27 |
| 「注釈区間 vs ランダム区間」の対照 | 29, 22(random patch の ablation) |
| 対象データ・モデル | 1, 2, 3 |

## 注意
- 調査はシステマティックレビューではなく、網羅性は保証しない。数値は各論文の報告値で、本研究で再現したものではない。
- 2026 年の論文の多くは arXiv の preprint である(査読状況は、確認できたものだけを記載)。
- [記載] の項目は、各論文のページを再確認していない。発表資料で引用する際は、本文で書誌と主張を確認すること。
