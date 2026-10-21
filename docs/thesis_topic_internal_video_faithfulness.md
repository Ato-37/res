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
| claim-specific human review/direct score | 0/25、0/25 | 直接faithfulnessの実測はまだない | faithfulness改善の肯定的結論 |

数値の一次記録は[全体の卒研リード記録](../experiments/2026-10-21_thesis_lead_internal_selective_dependence.md)、[global-bias追試](../experiments/2026-10-21_test_split_internal_bias_frame_mean_interaction_replication.md)、[質問区間介入](../experiments/2026-10-21_qta_intervened_qa_interval_generation.md)にある。各結果は小標本・探索的で、異なるendpoint間を統合していない。

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
