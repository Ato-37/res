# 卒業研究案：動画VLMの内部steeringは説明の証拠依存を改善するか

更新日：2026-10-21

対象：Qwen2.5-VL-7B-Instruct、NExT-GQA/NExTVideoのローカル実動画実験

## 研究の問い

**動画VLMの内部visual-key steeringは、説明claimが示す証拠への選択的依存を、無介入および同量のmatched-sham介入より高めるか。**

attentionやlogitが動いたことだけをfaithfulness改善とみなさない。内部介入、モデル出力、claimの証拠依存を別々の段階として測る。対象は説明の見栄えや人の納得感ではなく、固定claimの予測が、そのclaimを支える動画区間を取り除いたとき、長さ・位置・視覚変換を揃えたcontrol区間より選択的に変化するかである。

## 現在までに確認できた結果

| 実験 | 実測 | 読めること | 読めないこと |
|---|---:|---|---|
| 全動画5択QA、27動画 | baseline/QTA/sham全条件15/27正答、予測選択肢変化0/27 | この設定ではsteeringが回答選択を改善しなかった | 説明claimのfaithfulness全般 |
| 質問attention選択窓への介入、固定3動画 | 制約付き区間出力の差0/3、tIoU差0/3。token log-probabilityは動いた | 内部状態・局所logit変化は最終出力改善と同義でない | claim支持や直接faithfulness |
| claim尤度×frame-mean交互作用の先行3件 | 差分の差 +0.03927 nat/token、3/3正 | global biasで空間構造入力へのclaim尤度感度が増えた可能性 | claimの正しさ、証拠区間への選択的依存 |
| 別の固定3件による再現 | 有効2/3、効果 −0.00904 と +0.04469 nat/tokenで逆符号。残り1件はclaim形式不正 | 正方向の一貫性は再現されず、global bias機序は不安定 | 全3件平均、CI、p値、母集団効果 |
| 旧v10 artifactへの探索的レビュー（3人×3問） | 回答適合は5/9「適合」、3/9「不適合」、1/9不明。説明根拠は5/9「不足」、2/9十分、2/9不明 | 正答と説明の十分性は別軸であり、説明が正しいだけでは証拠説明が十分とは限らないという評価動機 | 旧回答のartifact fingerprintがなく、本人性・独立性も認証しない。v10では旧claim集合に結び付くが、正式faithfulness scoreには算入しない |
| claim-specific human review/direct score | 正式レビュー0/25、現行3件0/3 | 直接faithfulnessの実測はまだない | faithfulness改善の肯定的結論 |
| QA区間mask感度の独立小規模追試（val固定10件、6件採点） | QA-key bias−baselineの平均 `DeltaF` −0.00355 nat/token（3/6正）；bias−sham −0.00878（2/6正） | 先行3件の小さな正方向proxy平均は再現せず、この機械的estimandに安定した正効果は確認できない | QA intervalはclaim-supportではなく、n=6・探索的・直接faithfulnessではない |
| 同一estimandの次候補val cohort（7件採点、42 forward） | QA-key bias−baseline −0.00368 nat/token（3/7正）；bias−sham +0.01474（5/7正、範囲−0.04326〜+0.08636） | QA−baselineは再度ほぼ0。QA−shamは正方向平均だが異質性が大きく、別cohortで安定した改善とは確認できない | 同じ不確かな機械QA interval proxy・小標本・探索的。直接faithfulnessではない |
| 3つのval batchの行単位raw記述集計（非重複16件） | QA-key bias−baseline平均 +0.00046 nat/token、8/16正；bias−sham平均 +0.00961、9/16正 | QA−baselineはほぼ0。proxyのraw summaryは再現性が弱く、明確なliftなし | 非確率・逐次・幾何条件付き標本。CI/検定/母集団推定なし。直接faithfulnessではない |
| **事前登録した確証run（2026-10-22）：QA注釈区間 key bias（+ln2）対 同数sham、val 149動画、fp32読み出し、人手ラベル不要** | annotation−sham平均 −0.00064 nat（SD 0.0805、95% CI [−0.0137,+0.0124]、t検定 p=0.92）、正73/負76；TOSTは±0.025 natで同等（p=0.00015）；正答は3条件とも109/149 | この設定ではQA注釈区間への選択的な依存の差はshamと同等（±0.025 nat以内）。bf16読み出しの0.125 nat量子化を解消して測った | 有効149/要求149、447/447 forward、失敗0。δ_min=0.025は筆者設定。**陽性対照（注釈keyを×1/16、×1/256に抑制）も効果なし（Holm p=0.35〜0.55、同等）のため、この操作自体に出力への作用がなく、本結果は「選択的依存なし」の証拠としては情報量がない**（用量反応ln4〜ln16も同様に効果なし）。注釈PTSは未校正。QA注釈が動画の半分以下の動画に限定。直接faithfulnessではない（[結果ログ](../experiments/2026-10-22_annotation_key_bias_confirmatory_val_result.md)） |

数値の一次記録は[全体の卒研リード記録](../experiments/2026-10-21_thesis_lead_internal_selective_dependence.md)、[global-bias追試](../experiments/2026-10-21_test_split_internal_bias_frame_mean_interaction_replication.md)、[質問区間介入](../experiments/2026-10-21_qta_intervened_qa_interval_generation.md)、[QA区間proxy追試](../experiments/2026-10-21_freshval_qa_interval_claim_mask_bias_cohort10.md)にある。各結果は小標本・探索的で、異なるendpoint間を統合していない。

旧v10 review JSONを現物から再検査し、3人×3問のaggregateを再現した。特に第2問は回答適合の不一致が2/3、第3問は根拠不足が2/3だった。今回の版間監査では、v10の質問・claimは旧sentence-explorationのclaim manifestとsource lineageに3/3件結合した。v10からv14のitem内容を比較すると1問目・3問目は同一だが、2問目はitem ID・質問・claimを保ったまま埋込動画byteがv11で差し替わっていた。回答時刻は保存されたv10 artifactの更新後、v11更新前であり、v10を見たことの時系列上の裏付けになるものの、JSONにfingerprintがないため提示artifactの厳密な認証ではない。旧GPU候補receiptは7件とも0 forward/0 score callで、これらの所見を内部介入効果と結び付ける得点は存在しない。したがってこのユーザー／reviewerフィードバックは旧v10 claimの探索的な人手所見としてのみ扱い、後版2問目や現在のGPU生成claim、faithfulness効果・信頼性推定には流用しない。nicknameや自由記述は記録しない。詳細は[探索的レビュー項目集計と版間監査](../experiments/2026-10-21_exploratory_review_item_summary.md)。

## 卒研としての主張候補

現時点で支持可能な結論は、**「広域visual-key biasは、実動画で一部の尤度proxyを変えるが、効果は小標本で混在し、回答・生成区間の改善やclaim固有faithfulnessへ一貫して結び付かなかった」**である。これは肯定的なfaithfulness改善の実証ではない。内部状態の変化をfaithfulnessと取り違えやすい評価上の問題を、実動画、matched sham、失敗を含む分母管理、複数endpointで示す卒研としてまとめられる。

**仮題：**「動画VLMにおける内部視覚steeringと説明faithfulness—証拠依存・出力変化・matched-sham特異性の実動画評価」

## 次に直接検証する最小設計

1. 結果を見ずに固定した動画と生成claimを使い、独立したレビュー担当者がclaimごとの支持区間を判定する。判定不能・複数区間も事前規則に従って記録し、都合の悪い例を黙って除かない。
2. claim・prompt・token列・model revision・layer・介入量を固定し、各claimをbaseline、支持区間key steering、同数かつ非重複の時間matched-sham steeringで採点する。
3. 各条件で固定claimのtoken log-likelihoodを、claim支持区間を除いた動画と、同じフレーム数・マスク方式の非支持control区間を除いた動画で比べる。`F = LL(control除去) − LL(支持区間除去)` とし、主要contrastを `F(支持steering) − F(sham steering)`、副次contrastを `F(支持steering) − F(baseline)` と事前登録する。正の差が狙い方向。
4. claim/videoを解析単位とし、全行・生成/採点失敗・除外理由を報告する。小標本で動画内forwardを独立標本と数えない。単一の主指標・閾値・解析を結果閲覧前に固定する。
5. smokeではsource/data binding、実際に変化したmask/logit、support/controlの長さ一致、出力の再現性を確認する。direct endpointが成立しない場合はproxyとして明示し、直接faithfulnessを測ったとは書かない。

この指標も、支持区間を除くと固定claimの尤度が低下するという**モデル内部の反事実的依存**を測るものであり、説明生成の品質、claimの真偽、人にとっての有用性を単独では証明しない。生成された説明そのもののfaithfulness改善を主張するには、steered generationとbaseline/sham generationを別途比較し、出力claimを条件盲検で判定する必要がある。

## 成立条件と現状の停止点

- 人手レビューの免除はGPU探索を許すもので、claim-supportラベルを作り出すものではない。旧レビューの印象回答は今回のclaimに対する時間区間ラベルとして流用しない。
- ローカルの既知ログを除く次候補は抽出可能だが、exposure manifestは既知JSONに基づく不完全なheuristicであり、事前学習露出や未記録実験履歴を除外できない。従って新候補でも確認的・完全に独立とは呼ばない。
- 直接claim-supportレビューが用意できない場合、卒研の到達点は「faithfulnessを改善した」ではなく、「実動画で内部介入を診断し、proxyの限界と再現性を測った」に置く。肯定結果を捏造せず、負・混合結果を貢献として明示する。
- データ権利は未確認。保護動画・claim・質問・回答・個別receiptはローカルに保持し、公開するのは審査済みの集計値と必要最小限のコードのみ。

## 2026-10-22追記：画像VLMのcounterfactual説明faithfulness

Menzat et al. の *Counterfactual Tests for Measuring Chain-of-Thought Faithfulness in Visual Language Models*（arXiv:2609.06704、Findings of EMNLP 2026採択）は、画像の対象物を削除したcounterfactual pairを用い、固定選択肢に対する元画像・編集画像の回答分布のTVDと、説明が編集対象に言及したかを結び付けるvCCTを提案している。連続的なvCCTはprompting方式を分離した一方、予測反転だけを見るvCTは高値に飽和しやすいと報告されている。編集妥当性を補うため、再構成のみのno-op対照と、GroundingDINO/SAMによるマスク対応フィルタを使い、編集・言及判定を人手監査した結果も報告する。手法・公開実装は[論文](https://arxiv.org/html/2609.06704v1)と[公式コード](https://github.com/neuralisa/cct-vlm)を参照。

この知見は、本研究のQA正答率や固定claim尤度だけでは「説明が予測を反映したか」を測れていない点を明確にする。内部steeringの改善を検証する候補としては、同一動画の品質確認済みイベントcounterfactualに対し、内部介入が説明文のイベント言及と回答分布shiftの対応（vCCT型の連続指標）をbaseline・matched shamより高めるか、という設計が考えられる。ただし原研究は画像対象物編集であり、動画の行為・順序counterfactualは対象外と明記されている。時間的一貫性を保った動画編集、編集artifact対照、イベント変更の妥当性が先に必要である。

現在のローカル資源には適格なイベントcounterfactual動画pairが確認されていない。NExT-GQAの時間区間はQA根拠であってclaim-specific event changeではなく、CLEVRERの既存pair feasibility枝も意味的に有効なmatched pairを確立できず閉じている。したがってvCCT型の動画実験は将来候補に留め、既存proxyを再利用したり画像版の結果を動画faithfulnessへ一般化したりしない。直接faithfulnessは正式レビュー0/25、現行3件0/3のままである。詳しくは[設計監査](../experiments/2026-10-22_counterfactual_faithfulness_design_audit.md)。
