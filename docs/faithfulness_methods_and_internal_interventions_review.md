# 動画VLMの説明 faithfulness と内部介入：論文候補レビュー

調査日: 2026-10-09
対象: Qwen2.5-VLによる動画質問応答の説明を、入力根拠・行動的一貫性・内部計算の因果性からどう評価するか。各論文の知見は対象モデル・タスク・介入の範囲に限定する。

## 要点

区別すべき問いは三つ。(1) 説明で重要とされたフレーム／領域を変えると回答スコアが変わるか（入力依存性）、(2) 説明理由を反転した入力で予測分布が変わるか（行動的一貫性）、(3) その根拠が回答を生む内部経路を媒介したか（内部機構への忠実性）。(1)(2)では入力から回答へ説明文を迂回する直接経路が残り、説明の原因性は示せない。(3)はactivation patching等で踏み込めるが、clean/corruptペア、位置、単位、score、候補選択に強く依存する。マスク後に回答が変わる／headを消すと回答が変わるだけでは、説明が機構を正しく述べた証拠にはならない。

本研究で同じhead/layer/steering量の探索を再開する根拠はない。次の候補は、説明を検証可能なイベントclaimに分解し、claimと時間区間の対応を人手監査した上で「説明claimに対応する時空間証拠の内部媒介効果」という新しいestimandを事前定義できるか検討すること。妥当性ゲートを通過しなければGPU実験へ進まない。

## 方法の比較

| 方法 | 測るもの | 強み | 落とし穴 |
|---|---|---|---|
| Deletion/Insertion、Comprehensiveness/Sufficiency | 領域を隠す・残す・除き回答score変化 | 入力依存性 | mask/blurでOOD、順序・基準値依存。説明faithfulnessそのものではない |
| ROAR | 重要特徴を訓練・評価双方から除き再学習 | test時だけ隠す削除との対照 | 再学習コスト大。元モデルの局所因果効果ではない |
| Sensitivity-n / faithfulness correlation | attribution特徴操作とscore変化の相関 | 順位・強度検査 | 特徴相互作用・摂動・相関解釈に依存 |
| Counterfactual reason test | 理由を反転・除去した入力で回答変化 | 行動的一貫性 | 自然さ・意味保存が難しく、入力→回答の迂回経路を遮断しない |
| Reconstruction / simulatability | 説明から再構成した入力で予測再現 | 説明の十分性 | 再構成器・圧縮・事前確率依存。内部機構説明ではない |
| Activation patching / mediation | clean/corrupt間で内部活性を交換しlogit差 | 内部媒介効果候補 | ペア・位置依存。patchで作るOOD、並列経路、相互作用、候補選択 |
| Ablation / path patching | head/MLP/token/frameまたは経路を置換・遮断 | 成分から経路へ因果仮説を絞る | zero ablationの不自然さ、冗長性・補償。無効果は不使用を意味しない |
| 人手rationale/時間一致 | 説明と人手根拠の重なり | 読みやすさ・根拠提示 | plausibilityであり内部faithfulnessとは別 |

**指標上の注意:** Deletion/Insertion AUCはサリエンス値の強度より順位だけに反応する場合があり、人工摂動はOOD入力を作りうる。SufficiencyとComprehensivenessは非対称の問い。正答性、回答margin、説明claimの妥当性も分けて記録する。好ましい結果の指標だけを選ばず主estimandを事前固定する。

## 論文候補

### A. 説明faithfulnessと行動テスト

1. **ERASER: A Benchmark to Evaluate Rationalized NLP Models** — [DeYoung et al., ACL 2020](https://aclanthology.org/2020.acl-main.408/)。人手rationaleと予測の一致、rationale除去時の挙動を評価。転用: 説明をイベントclaimと時間区間へ分解する枠組み。限界: NLP向けで、人手根拠一致は内部忠実性を証明しない。
2. **Faithfulness Tests for Natural Language Explanations** — [Atanasova et al., ACL 2023](https://aclanthology.org/2023.acl-short.25/)。説明にない理由を入力へ加えるcounterfactual editorと、説明から入力を再構成するテスト。転用: claim反転動画を考える発想。限界: 意味保存が難しく、回答への直接経路が残る。CLEVRER pairの妥当性が未確立なら件数確保で弱いpairを使わない。
3. **The Probabilities Also Matter** — [Siegel et al., ACL 2024](https://aclanthology.org/2024.acl-short.49/)。binary flipでなく反事実入力による予測分布変化を測るCCT。転用: 回答変化率でなく選択肢確率やtarget-vs-foil logit marginを使う動機。限界: 分布変化はpair妥当性を保証しない。
4. **On Measuring Faithfulness or Self-consistency of Natural Language Explanations** — [Parcalabescu & Frank, ACL 2024](https://aclanthology.org/2024.acl-long.329/)。既存テストは内部faithfulnessより出力self-consistencyを測ると論じ、11 LLM・5タスクを比較。CC-SHAPは回答と説明生成への入力寄与を比較する。本研究の最優先精読候補。直接経路が本研究でどう働くかを検討する。
5. **Are self-explanations from Large Language Models faithful?** — [Lyu et al., Findings ACL 2024](https://aclanthology.org/2024.findings-acl.19/)。faithfulnessは説明・モデル・タスク依存と報告。単一質問形式から一般化しない根拠。動画は直接扱わない。
6. **Towards Faithful Model Explanation in NLP: A Survey** — [Lyu et al., Computational Linguistics 2024](https://aclanthology.org/2024.cl-2.6/)。110超の手法を類似度、内部構造、gradient、counterfactual、self-explanatoryに分類。探索の入口だが動画介入は原論文で確認する。

### B. 画像・動画XAI

7. **Sanity Checks for Saliency Maps** — [Adebayo et al., NeurIPS 2018](https://proceedings.neurips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf)。重み・ラベルrandomizationでsaliencyのモデル／データ依存性を検査。転用: 可視化sanity check。通過してもfaithfulnessの証明ではない。
8. **A Benchmark for Interpretability Methods (ROAR)** — [Hooker et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks-Paper.pdf)。重要特徴を訓練・評価双方から除き再学習。test時削除との対照だが動画VLM再学習は高コストで、元モデルの局所因果性は測らない。
9. **Metrics for Saliency Map Evaluation** — [Gomez et al., arXiv 2022](https://arxiv.org/abs/2201.13291)。Deletion/Insertion AUCが順位に偏り値の強度を使わないこと、摂動OODを批判。frame ranking一曲線に頼らない理由。arXiv論文で提案指標も因果ground truthではない。
10. **On the Faithfulness of Vision Transformer Explanations (SaCo)** — [Wu et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html)。異なるsalienceのpixel群をペア比較し、score差に見合う影響差を集約。転用: 重要度帯・時間区間の比較。限界: 画像ViT用で自然言語説明や内部headを測らず、摂動妥当性は別問題。
11. **Token Transformation Matters (TokenTM)** — [Wu et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Token_Transformation_Matters_Towards_Faithful_Post-hoc_Explanation_for_Vision_Transformer_CVPR_2024_paper.html)。attentionだけでなくtokenの大きさ・方向の変化も寄与へ集約。attention mapだけで「使った」と結論しない参考。post-hoc attributionであり因果patchingではない。
12. **Evaluating Faithfulness for Temporal Colour Constancy** — [Rizzo et al., XAI 2022 workshop/arXiv](https://arxiv.org/abs/2211.07982)、[code](https://github.com/matteo-rizzo/saliency-faithfulness-eval)。動画の時間的色恒常性へNLP由来testを適応し、対象設定ではattentionは非faithful、confidence saliencyはfaithfulと報告。時間動画の先例だが低レベル視覚タスクでVLMへ一般化不可。
13. **Understanding Video Transformers via Universal Concept Discovery (VTCD)** — [Kowal et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kowal_Understanding_Video_Transformers_via_Universal_Concept_Discovery_CVPR_2024_paper.html)。動画Transformer内部から時空間concept/tubeletを発見・順位付けし衝突・遮蔽追跡などを可視化。転用: 視覚tokenを時刻・イベントclaimへ対応づける発想。概念可視化は因果必要性やNLE fidelityを示さない。
14. **Don't Lie to Me: Verified Perturbation Analysis (EVA)** — [Fel et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fel_Dont_Lie_to_Me_Robust_and_Efficient_Explainability_With_Verified_Perturbation_Analysis_CVPR_2023_paper.html)。摂動範囲を明示して説明を検証。転用: 黒塗り一種でなく摂動族を定義する視点。主に画像分類。

### C. 内部介入・mechanistic interpretability

15. **Investigating Gender Bias Using Causal Mediation Analysis** — [Vig et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)。neuron/headをmediatorとしてdirect/indirect effectを分け、相乗・抑制も分析。転用: claimに対応する視覚情報から回答marginへの媒介効果の基礎。元は言語バイアス設定。
16. **Locating and Editing Factual Associations in GPT (ROME/causal tracing)** — [Meng et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)。subject tokenをcorruptしclean activationをpatchして事実回答回復位置を探す。転用: clean/corrupt/patch三条件と連続score。限界: 事実想起の局在化は説明faithfulnessではなく、人工corruptionは自然動画反事実でない。
17. **Towards Best Practices of Activation Patching** — [Zhang & Nanda, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html)。corruption法・metricで結論が変わると検証。内部介入の必読。動画frameや視覚tokenを持つQwenで別途確認が必要。
18. **Is This the Subspace You Are Looking for?** — [Makelov et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)。patchingが出力効果を作っても、因果的に離れた部品が休眠並列経路を起動しただけという解釈illusionを示す。steeringの変化を元の判断経路と即断しない警告。
19. **Towards Automated Circuit Discovery (ACDC)** — [Conmy et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)。patching候補から計算graph edgeを剪定し回路を抽出。head単体から時空間token→視覚層→言語層→回答位置の経路仮説へ広げる。task/score/data依存で動画NLE fidelityは未検証。
20. **Path Patching** — [Goldowsky-Dill et al., arXiv 2023](https://arxiv.org/abs/2304.05969)。sender→receiver経路のみ介入。経路仮説に有用だがpreprintで、動画VLMでは探索費用・多重性が大きい。
21. **Towards Faithful Natural Language Explanations: A Study Using Activation Patching in LLMs** — [Yeo et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.529/)。patching由来の因果寄与が説明claimと回答予測で整合するかをCausal Faithfulnessとして評価。本研究に近い最重要候補で全文・付録・コードから定義、pair、score、失敗例を精読する。限界: テキストLLM向けで動画token・時間groundingは別課題。activation swapは自然状態を保証しない。

### D. 動画・マルチモーダル近接研究（目的は別）

22. **Interventional Video Grounding With Dual Contrastive Learning** — [Nan et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Nan_Interventional_Video_Grounding_With_Dual_Contrastive_Learning_CVPR_2021_paper.html)。moment groundingのselection biasをbackdoor adjustment・proxy confounderで扱う。時間区間とqueryの因果図に参考だが、VLM説明faithfulnessではない。
23. **Causal Attention for Vision-Language Tasks** — [Yang et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Causal_Attention_for_Vision-Language_Tasks_CVPR_2021_paper.pdf)。vision-language attentionの交絡を扱う。因果図の参考でありattention mapが説明理由を証明する論文ではない。

## 内部介入の読み違い防止

- Ablationは0・平均・他例活性など置換先で因果質問が異なる。単独zero ablationの無効果は冗長性・補償・OODのため不使用を意味しない。
- Activation patchingでclean活性をcorrupt実行へ移しscoreが戻っても、当該pairの媒介効果候補であり、自然入力での通常利用や人間可読概念の存在を単独で示さない。
- Causal tracingの人工corruptionを意味妥当な動画counterfactualと同一視しない。Path patchingでも経路・剪定基準・データ依存を報告する。
- Steering/model editingは出力制御でありfaithfulness評価ではない。出力変化は説明忠実性の証拠にならない。
- Attention/gradientは寄与推定・可視化で、内部経路の因果効果と同一ではない。

## 本研究への転用案（仮説）と事前ゲート

**仮説案:**「説明から事前規則で抽出し時間区間へ対応づけたイベントclaimは、選択回答marginを生む内部状態を、対照区間より強く媒介するか」。説明全体のfaithfulnessではなく、claimの時間groundingと内部媒介の一致に限定する。曖昧・視覚検証不能claimは無理に対応づけず、除外数を報告する。

GPU前に必要なこと:
1. claim抽出・視覚検証基準を事前定義し、説明を効果確認後に選ばない。
2. claim-to-timeの人手監査、評価者間一致、不一致・除外を記録。説明なし・対応不能を隠さない。
3. 同一動画内対照か意味妥当性を独立確認したcounterfactualを用意。黒塗り・ぼかし・別フレームをsemantic counterfactualと呼ばない。
4. 元入力での回答可否、正答・誤答の扱い、foil選択法を事前指定する。
5. frame/visual token、層、head/MLP、patch位置、corruption、score（例: target-vs-foil logit margin）を固定。旧head確認後にhead/layer/alphaを再探索しない。
6. sham patch、時間ずらし、ランダム位置、直接入力介入を対照とし、並列経路・冗長性・介入外挿を検討。
7. 主効果・不確実性・検定・多重性・除外・失敗規則を先に定め、探索pilotと独立確認を分ける。

推奨順: (i) Parcalabescu & Frank、Zhang & Nanda、Yeo et al.精読、(ii) NExT-GQAの説明・時間ラベル・claim対応可能性をCPU監査、(iii) estimandと対照を事前定義、(iv) hook/cache/patch機構のsmoke、(v) 妥当性ゲート後のみ独立確認。既存の負のhead確認は同じ探索の再開根拠ではない。新estimandは旧結果の上書きでなく、独立確認を要する別の問いとして扱う。

## 結論を限定する範囲

- 結果はNExT-GQA・Qwen2.5-VL・特定介入・scoreに限定し、他モデル／動画一般へ外挿しない。
- 回答marginの媒介効果は内部全体の完全説明や自由生成CoT全体の因果性を示さない。
- mask/blur/patchは介入の定義。差を介入なしの自然効果と同一視しない。
- 人手rationale一致、削除効果、反事実分布変化、activation patchingは異なる問いを測る。
- 探索で選んだ条件を同一データで最終検定せず、post-selectionを明示する。

---

### 2026年の追加候補

**CAuSE: Decoding Multimodal Classifiers using Faithful Natural Language Explanation** — [Bandyopadhyay et al., TACL 2026](https://aclanthology.org/2026.tacl-1.37/)。interchange interventionで学習したcausal abstractionを用いて、既学習multimodal classifier向けfaithful NLEを生成する。multimodal内部表現と説明を因果介入で結ぶ近接候補。classifierと生成型動画VLMでは出力・時間構造が異なり、Qwenへの直接適用は未検証。全文で手法・metric・失敗例を確認する。

**FaithLM: Towards Faithful Explanations for Large Language Models** — [Chuang et al., EACL 2026](https://aclanthology.org/2026.eacl-long.177/)。説明と矛盾するcontrary hintへの予測変化でfaithfulnessを測る介入テスト。maskingに依存しない発想を得られるが、テキストLLM向けで動画claim・内部経路は直接検証しない。説明最適化と同じデータで評価する際は選択バイアスに注意する。

本書は候補論文の設計用整理で、検索式・全DB・重複除去を記録した体系的レビューではない。主要会議の一次資料を優先し、arXiv/workshopは身分を明記した。採用前に本文・付録・実装を照合する。

### 2026-10-15追加：動画介入ベンチマークとの区別

**INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs** — [Yangjunqi et al., ACL 2026](https://aclanthology.org/2026.acl-long.2062/)。実・合成動画9,800問で、Base、視覚劣化、証拠corruption、時間順序介入を分ける。正解が変わらない想定の介入はResist Rate、順序が正解に必要な項目のshuffle/reversalはTemporal Sensitivity Scoreで評価する。Base精度だけでは誘導条件での信頼性を示さない、という診断は本研究の回答marginとfaithfulnessを分ける必要性を補強する。これは回答が動画に依存するかの信頼性評価であり、自由記述claimがモデル判断の根拠を忠実に表すかを直接検証する指標ではない。同論文はlabel-preserving operatorについて、200例/条件を3名のblind annotatorで監査している。

**Evidence-Backed Video Question Answering** — [Wang et al., ECCV 2026](https://arxiv.org/abs/2607.11862)。答えに時間区間とdense tracked maskletsを添える課題と、人手確認付きST-Evidenceを提案する。モデル出力と証拠の照合を可能にする点で近いが、予測された証拠のgrounding精度は、説明claimへの選択的依存や説明の内部faithfulnessとは別の性質である。

**EDCT-Bench** — [Ding et al., arXiv:2609.17953](https://arxiv.org/abs/2609.17953)。画像VLMの説明から視覚概念を抽出し、検証した最小編集を適用して回答と説明の一貫性を測る。説明由来の介入という点は近いが、現時点ではarXiv preprintであり、画像向けの編集・判定を時間構造のある動画へそのまま移せない。論文自身も、モデルごとの説明から介入を選ぶことで難度が変わり得ることと、より多くのhuman-verified記述・介入による確認を課題として挙げる。

**本研究への含意:** 現行のclaim-mask案は説明claimを固定し、claimに人手対応づけした区間と同枚数の時間対照でgold-vs-foil margin差を測るため、INFACTの一般的な動画依存性やE-VQAのgrounding精度と異なる問いを狙う。human claim reviewが完了するまでは、どちらの文献も未実施のfaithfulness実験を代替しない。別案として時間順序介入を検討する場合は、質問が順序依存であること、介入後に正答がどう変わるべきかを結果閲覧前に人手確認し、説明由来介入とは別estimandとして事前登録する。
