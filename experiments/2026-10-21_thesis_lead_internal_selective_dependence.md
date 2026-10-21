# 卒研候補：動画VLMの内部証拠誘導は説明の選択的忠実性を高めるか（2026-10-21）

## 現時点の結論

現行QTAの固定予測key-bias適用後、選択的な予測依存を安定して高める証拠は確認できていない。コホート16–18、20–24、26の9つの逐次探索コホート（27行）について、QTA `DeltaF`を元の5択log-probabilityから再計算すると平均 `+0.01389 nat`、中央値はほぼ0、範囲 `[-0.125,+0.250]`、`+0.001 nat`超は7/27だった。同じ行で再計算したQTA−matched-shamは平均 `+0.02315 nat`、中央値はほぼ0、範囲 `[-0.250,+0.250]`、正値は11/27だった。全9組は3/3行・15/15 QTA forwards・6/6 sham forwardsを完了し、QTA/sham行digestは一致、27動画digestもすべて異なる。独立監査はreceipt chain、counts、行結合、算術を再現した。値は逐次post-hoc探索の記述値であり、独立標本の確認実験・母集団推定・有意差ではない。

したがって卒研の中心を「提案介入が既にfaithfulnessを改善した」とは置けない。より妥当な主題は「質問attentionで選んだ動画keyへの一様な内部biasは、選択的な証拠依存を増やすか。増えない場合、matched shamや全体的な確信変化と区別できるか」である。現状の観測は、単純な全層・広範囲key-biasに対する負または不確実な結果として研究可能だが、主張を成立させるには凍結した新規コホートとclaim-support妥当性が要る。

## 卒研としての研究質問と仮説

**RQ:** Qwen2.5-VLが動画QA説明文を生成するとき、説明中claimに対応すると判定された時間区間への内部key routingは、同じ予測・同じ計算量のbaselineおよびmatched-sham routingより、その予測のclaim-evidence選択的margin依存を高めるか。

**主要仮説:** claimに対応する根拠keyへの内部biasによって、固定予測のevidence-ablation効果がmatched-control ablationより増える（`DeltaF_intervention − DeltaF_baseline > 0`）うえ、同じサイズのsham biasでは同程度の増加が起きない。

**反証条件:** 効果が0付近、符号が一貫しない、shamとの差が0付近、または説明claimと区間の対応が妥当でない場合、「内部介入がfaithfulnessを改善した」と結論しない。これは研究の失敗ではなく、粗いattention steeringの限界を示す結果である。

## 実験設計案

1. まだQTA結果に使っていないテスト動画から、結果を見ずに候補を凍結する。現行3行コホート群をconfirmatoryデータに再利用しない。
2. 生成説明をclaim単位に分割する。各claimの支持区間は、少なくとも独立した一名の人間が動画とclaimを照合して記録する。ユーザーのレビュー収集免除は実験着手の障壁を除くものであり、観察していない判断を作る許可ではない。
3. 予測ラベルとfoilを介入前のfull-video passで固定する。evidence区間と同長・近接位置のnon-evidence controlを用意し、動画key budget、層、強度、prompt、sample、seedを事前固定する。
4. baseline、claim-evidence key intervention、同サイズmatched-shamの3条件を比較する。固定予測marginについて `F = margin(control ablation) − margin(evidence ablation)`、`DeltaF = F_intervention − F_baseline`、specificity `DeltaF_intervention − DeltaF_sham`を主指標とする。全動画margin変化も測り、単なるconfidence driftを除外する。
5. 一人一動画を単位に事前に決めた標本数で実行し、全件のpaired値、失敗、bootstrap信頼区間、指定した一つの主要比較を報告する。探索コホートの結果を見て層・強度・群を変えない。

## 既に得られた支援的・反証的成果

- 既存のQTA固定予測key-ablationとmatched-shamを9コホート分、27行で監査再計算した。平均効果は非常に小さく、個体差の範囲内で符号も不安定だった。これは効果を支持する結論ではなく、現行の粗い介入をそのまま卒研の成功主張に使えない証拠である。
- 別のclaim-token尤度テストでは、3行に対しQA intervalと同数temporal-controlをbaseline/bias条件で比較し、介入×区間差の値が全行負、平均 `−0.018260 nat/token`だった。ここでもQA時刻は説明claim固有根拠ではないため、これはclaim-token感度proxyであり、忠実性低下の証拠ではない。
- QA temporal interval生成介入の修正版GPU smokeはRTX 5070 Tiで3/3生成を完了し、各層でDynamicCacheのsquare prefill後に `Q=1, K+1` が続くことを確認した。ところが三条件すべてで生成token SHAが同一、JSON parse成功0/3、tIoU=0だった。これは区間生成プロトコルの失敗であり、接地改善の測定値ではないため、full runは実施しない。
- claim内容が動画ピクセル統計に感度を示す過去の3行テスト（original対gray/mean frame）は補助診断に留まる。画素入力感度であり、claimが意味的に正しい根拠へ依存した証拠ではない。

## faithfulness進捗・制約

- GPU内部介入cohortに結びついた正式なclaim-to-frame支持レビューと直接faithfulness得点は、32-frame群6件と16-frame群19件を合わせて **0/25**。別の旧review artifactには3名分×3項目=9件の非公式な印象評価がある（question clarity: clear 4/9, unclear 4/9, cannot tell 1/9; answer fit: fits 5/9, does not fit 3/9, cannot tell 1/9; evidence sufficiency: sufficient 2/9, insufficient 5/9, cannot tell 2/9）。これは今回の25 claimに対するclaim-specific frame ground truthではない。
- 補助診断のみ: 固定予測の機械選択窓による `DeltaF` 27/27行。対象はQwen2.5-VL-7B-Instruct、NExT-GQAの局所テスト由来逐次コホート、質問attention上位key bias、QA選択窓/同長control ablation。事後選定とclaim-support未確認により意味的faithfulnessの推定ではない。
- 直接介入効果の推定: この27行について平均 `DeltaF=+0.01389 nat`、QTA−sham specificity `+0.02315 nat`。不確実性区間・p値は計算せず、逐次探索・小コホート・選択後評価のため母集団解釈しない。
- 32-frame内部介入では、保存claimのteacher-forced token尤度をQA区間／同数control画素black mask × baseline／QA-key `+ln(2)` bias／disjoint sham-key biasの6条件で、重複しないv2_short動画6件・36 forward測定した。cohort30のQA-bias−baselineは平均 `+0.04374`（3/3正）だったが、cohort29は平均 `−0.02934`（1/3正）で再現しなかった。記述的pooled平均は `+0.00720`（4/6正）、QA-bias−sham平均 `−0.00059`（3/6正）。QA区間はclaim固有supportではないため、これは意味的faithfulness改善を示さない。
- 接地生成smoke: 要求1/effective1、3条件完走だが妥当区間0件、全条件IoU 0。故に接地精度の介入効果は未測定。
- 別16-frame層の結果を含む直接claim-to-frameレビュー・直接faithfulness: 正式review未実施で0/25。レビュー免除は実行の許可であり、review済み結果とは扱わない。
- データ/モデルの権利状態は未確認。保護データと詳細artifactはローカルmode 0600のまま。

## 次の判断

卒研候補は「動画VLMの説明faithfulness proxyに対する内部attention steeringの再現性評価」。具体的な否定可能な主張は「質問区間visual-keyへの固定biasはbaselineよりQA窓black-mask感度を一貫して増やさず、matched shamに対する特異性も安定しない」。現行のn=6はこの仮説に対する初期反証的データを提供する。研究として成立させるには、開発データと分離した新規cohortを事前凍結し、claim固有support intervalを人手で検証し、baseline / claim-evidence bias / matched-shamと全体confidence driftを固定予測下で比較する必要がある。結果は研究の問い・介入・負/不確実なpilot成果を与えるが、SSSの卒研結論水準にはまだ届かない。

## 16-frame内部介入の追加結果と卒研としての提示（2026-10-21追記）

別の16-frame解像度層は、事前定義した幾何条件を満たす全19行で114/114 forwardを完了した。QA-bias−baselineは平均 **−0.002610 nat/token**（中央値 −0.010875、範囲 [−0.055202,+0.113485]、正6/19）、QA-bias−shamは平均 **−0.009011**（中央値 −0.001040、範囲 [−0.166461,+0.097747]、正9/19）だった。独立監査で19動画の一意性、入力・コード・モデル重みハッシュ、114 token列と算術を確認した。後方視的開発データであり、32-frame群とは解像度が異なるため統合推定せず、信頼区間・p値も主張しない。

32-frameの6行ではQA-bias−baseline平均 `+0.00720 nat/token`（4/6正）、QA-bias−sham平均 `−0.00059`（3/6正）だった。初期cohort30の3/3正方向はcohort29で再現せず、16-frame群の平均もほぼ0だった。現時点で卒研として使えるもっとも明確な結果は、「固定した汎用QA-key `+ln(2)` steeringは、QA窓pixel-mask感度という測定可能なproxyを安定して増やさず、matched-shamに対する特異的増加も示さなかった」という再現性・限界の知見である。全ての内部介入の失敗やfaithfulness改善の不可能性を示すものではない。

**仮題:** 「動画質問応答モデルに対する内部attention steeringは説明の証拠依存を改善するか—再現性とmatched-sham特異性の評価」

### 研究としての位置づけ

近接する研究として、ICLR 2026の[「Seeing but Not Believing」](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76818d8d85e05e45ce3a16a8468619d1-Abstract-Conference.html)は、VLMが証拠を視覚的に捉えていても誤答する場合を分析し、深層の証拠領域を選択的attention maskingで強調して画像QA精度を上げたと報告している。本研究ではその知見を動画QAへ無検証に移植せず、(i) 時間方向の証拠領域を使った内部介入が動画QAの正答率を変えるか、(ii) 正答率の変化が説明claimの証拠依存も改善したことを意味するか、を別々の評価軸として検証する。今回の汎用all-video-key biasは再現しなかったため、「広範なbias」から「時間局在・深層限定の選択的介入」へ進む動機は得られている。ただし未実装の介入を既成果として書かず、卒研の後続実験案として扱う。

この区別は注意重みの操作をそのまま説明のfaithfulnessと同一視しないために重要である。[JainとWallaceの一次研究](https://arxiv.org/abs/1902.10186)も、attentionの表示だけでは予測への因果的寄与が保証されない点を指摘している。従って介入によるQA精度の向上が確認されても、claim-specific evidence ablationに対する固定予測marginの選択的変化を別に示すまでは、faithfulness向上とは結論しない。

次の本実験は、結果未使用動画を固定し、生成説明をclaimごとに分割し、claim固有support区間をレビュー担当者が記録したうえで、固定予測に対するbaseline / support-key bias / 同予算shamを比較する。主要評価はsupport区間除去に対するclaim margin変化の介入差とし、同長・近接control、全体margin変化、失敗例も記録する。候補cohortと解析コードは結果閲覧前に凍結する。注釈品質・信頼性が確保できない場合、その測定上の限界を卒研の結論に含める。

faithfulnessの直接測定進捗は **正式なclaim-to-frame支持レビュー0/25、直接semantic-faithfulness得点0/25**。25件のGPU実験はQA interval pixel-mask感度proxyであり、claim固有support依存ではない。したがって研究のめどは「実動画による反証的内部介入結果、検証済み実装、および追試可能な卒研仮説がある」段階まで進んだが、主仮説の直接検証とSSS相当の確定結論には未到達。権利状態は未確認で、データ・詳細receiptはローカル限定。

## 制約付き実動画区間生成の追加GPU結果（2026-10-21追記）

実動画3件で、質問attention top-10% video-key bias（要求量 `+ln(2)`、layers 5–16）、baseline、同数hash-shamを比較するQA区間生成を完走した。選んだ区間の時間IoUは各条件で平均 **0.08300**、paired selected−baseline **[0,0,0]**、selected−sham **[0,0,0]**。3動画すべてで3条件のgreedy生成区間コードが同一だった。BF16表現に丸めたmask bias（`0.69140625`）が、12層×全生成forwardで処置/shamに適用されたことは監査ログで確認済み。これは固定3件のQA-localization null結果で、説明のclaim支持・直接faithfulnessでも、全介入の否定でもない。

卒研で具体的に扱える成果は、(i) 固定予測key-ablationの27行・19行等で選択依存が安定しなかったこと、(ii) 今回3件でも汎用temporal biasはgreedy QA intervalを変えなかったこと、(iii) 不適切なattention可視化やpixel-mask proxyをfaithfulness改善と誤認しない比較設計、の三点を組み合わせた「動画VLMにおける内部証拠誘導と選択的faithfulnessの再現性・sham特異性評価」である。肯定的改善を主張する材料はまだないが、仮説・再現可能な介入・監査可能な実動画null・次の反証条件が揃い、卒研テーマとして成立する。

直接faithfulness進捗は変わらず **0/25 claim-to-frameレビュー、0/25直接得点**。今回のQA intervalをclaim evidenceと読み替えない。直接の卒研主結果を求めるには、結果未使用の新規動画と人間が確認したclaim-support区間が必要である。レビューなしでも今回のGPU feasibility/null pilotは方法・限界の章に使える。詳細とreceipt provenanceは[`2026-10-21_qta_intervened_qa_interval_generation.md`](2026-10-21_qta_intervened_qa_interval_generation.md)に記録し、保護動画・receiptはローカルに限定する。

## 機序診断を含む現在の卒研ネタ

今回の事後logit診断では、固定3動画の各6生成stepで選択介入が同じ出力tokenのlog-probabilityを動かした（動画ごとのselected−baseline平均 `+0.00583, +0.01054, +0.02475 nat/token`、selected−shamは `−0.00115, −0.00003, +0.02375`）。しかし3動画すべてで制約された区間出力は三条件同一、tIoU差も全て0だった。18 token-stepは動画内で反復測定なので独立n=18として扱わず、3動画の探索的な記述値に限る。

これにより卒研の問いを「介入でスコアが上がるか」だけでなく、**内部動画keyへの操作がtoken-level予測分布を変えても、説明が示す動画証拠への選択的依存・接地精度に結びつくのか**へ具体化できる。卒研としての貢献候補は、(1) mask変更、raw token distribution、生成区間、claim-specific evidence-ablationを段階別に測り、(2) 出力確信の変化と証拠依存の改善を分け、(3) baselineとmatched shamに対する新規human-reviewed cohortで再現性を検証すること。いまある結果はこの問題設定と小規模なnull/機序pilotを与えるが、faithfulness改善の肯定的証拠ではない。

現状の直接claim-faithfulness測定は依然 **0/25**。必要な次の実験は新規動画でのclaim-to-frame support interval確認と、同一予測のsupport ablation marginに対するbaseline / support-key steering / shamの比較である。人手確認が得られない場合は今回の研究を、接地・faithfulness改善を未達と明記した再現性・診断設計の卒研としてまとめる。
