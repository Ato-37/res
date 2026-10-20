# 内部検討メモ: faithfulness確認とデータ適合性の結果

> **内部検討用の統合メモ。最終原稿ではなく、faithfulness実証の成功を主張しない。**
> 正解根拠フレームを隠してスコアが下がること自体は予想される反応であり、それ単独に新規性はない。
> 以下は保存済み結果の透明な記録と、現状の主張限界を残す内部メモである。唯一それ以上を問う比較は
> 同数・時間位置対応control maskとの差だが、controlは視覚内容や識別情報量まで一致していない。

## 1. 事前登録したhead-faithfulness独立確認

NExT-GQAのheld-out動画79件で、gradient×activation attribution上位4 head（全28層から選択）を
最終prompt位置で`alpha=0.5`に減衰させ、head数・層分布・活性値規模を対応させた5個のcontrolと比較した。
測定値は介入前top choiceを固定したgold-vs-foil log-probability marginであり、動画ごとの
`d_i = target margin drop - mean(control margin drop)`の平均を主estimandとした。

- 完了79/79、失敗0。全79件を解析し、介入前の正誤で除外していない。
- target平均margin drop `−0.0601 nat`、control `+0.0155 nat`、selective差`−0.0756 nat`（SD `0.3667`）。
- 95% t区間`[−0.1578,+0.0065] nat`。事前成功基準はCI下限`>+0.25 nat`だったが、上端も`+0.25`を下回った。
- これは事前登録した確認セットにおける、指定したQwen2.5-VL-7B、4-bit、top-4、半減衰、5-control、16-frame、選択肢marginでの負の結果である。一般的なfaithfulnessの不存在は示さない。
- cohort identity監査は同じsample-manifest hashについて79個の一意ID・path・完全一致file、重複0を記録している。ただしresultとrun manifestのcrosshash binding、および共有scene/campaignを含む統計的独立性は未確立である。厳密な来歴制約は[確認結果と監査メモ](head_faithfulness_confirmation_results.md)を参照。

したがって、このhead選択法の同一primary仮説は閉じる。head/layer/alphaや後付け部分群を変えた追加探索で救済しない。定義・統計の詳細は
[`head_faithfulness_confirmation_results.md`](head_faithfulness_confirmation_results.md)、元データとmanifestは
[`head_faithfulness_confirm_20261009.json`](../results/head_faithfulness_confirm_20261009.json)を参照。

## 2. Grounding-mask choice-margin sensitivity probe

### 実験と観測値

NExT-GQAのheld-out動画60件に対し、Qwen2.5-VL-7B-Instruct（4-bit量子化、bfloat16）へ
16フレームを入力し、grounding区間に重なるフレームをマスクした場合の5択gold-vs-best-foil
log-probability marginの変化を測定した。同数の非-groundingフレームを時間位置が近くなるよう
選んでマスクした条件をcontrolとした。black fillとGaussian blur fill（radius 12 px）は同じ60動画で
実施され、両条件とも60件が完了し、実行時failureはなかった。介入前にgold choiceが正しかったのは
37件、誤っていたのは23件だった。

下表のevidence/control列は記録済みレコードの単純平均（記述統計）。selective列は事前primary
estimandである evidence-mask margin drop minus control-mask margin drop。

| cohort / n | fill | evidence drop平均 | control drop平均 | selective差（95% pointwise CI） | sign-flip p |
|---|---|---:|---:|---:|---:|
| 全件 / 60 | black | +0.208 | -0.088 | +0.296 [-0.046, +0.615] | .0823 |
| 全件 / 60 | blur | +0.196 | -0.165 | +0.360 [-0.035, +0.721] | .0595 |
| 介入前正答 / 37（primary） | black | +0.304 | -0.030 | +0.334 [+0.111, +0.581] | .0053 |
| 介入前正答 / 37（primary） | blur | +0.257 | -0.199 | +0.456 [+0.172, +0.760] | .0039 |
| 介入前誤答 / 23（secondary） | black | +0.054 | -0.179 | +0.234 [-0.549, +0.946] | .5711 |
| 介入前誤答 / 23（secondary） | blur | +0.098 | -0.109 | +0.207 [-0.663, +0.967] | .6592 |

追加の事後的component分解（各component対0のpointwise percentile-bootstrap 95% CIとsample sign-flip test、
black/blur×3 cohort×2 componentの12比較、multiplicity correctionなし）では、evidence dropの全件平均は
black +0.208 (CI [-0.096,+0.494], p=.168)、blur +0.196 (CI [-0.127,+0.490], p=.225)で、区間は0を含んだ。
baseline-correct群ではblack +0.304 (CI [+0.101,+0.544], p=.0032)、blur +0.257
(CI [+0.041,+0.507], p=.0349)。control dropは同群でblack -0.030 (CI [-0.203,+0.149], p=.767)、
blur -0.199 (CI [-0.389,-0.037], p=.0332)だった。これらは後付けcomponent分析であり、検定結果を
確証的主張に用いない。特にblurではcontrol側のmargin上昇もselective差に寄与しうる。

二つのfill間の対応差（blur minus black）は+0.122（SD=0.442、n=37、pointwise 95% CI
[-0.007, +0.277]、両側sample-level sign-flip p=0.0995）であり、fill条件間に明確な差があるとは
結論できなかった。全60件での平均はblack +0.296（95% CI [-0.046, +0.615]、p=0.0823）、blur
+0.360（95% CI [-0.035, +0.721]、p=0.0595）だった。baseline-incorrect 23件ではblack +0.234
(95% CI [-0.549, +0.946]、p=0.571)、blur +0.207（95% CI [-0.663, +0.967]、p=0.659）で、
いずれの区間も0を含んだ。

### 成果としての評価

「ground-truth intervalをマスクしたら正解marginが低下した」という単純な観測は、視覚情報を除けば
予測根拠が弱まるという自明な可能性と区別できず、それ自体は卒論の主要な研究成果にならない。
本実験が追加で問うたのは、同数かつ時間位置を対応させた別区間のmaskよりground-truth区間maskが
強くmarginを低下させるか、である。しかしcontrolは時間位置のみを近づけ、フレーム内容、物体・動作、
視覚的顕著性、回答識別情報量を対応させていない。black/blurはOOD入力も作り得る。

selective contrastは事前primaryの正答群では正方向でpointwise CIが0を含まなかった一方、全60件では
両fillともCIが0を含み、正答状態による効果差も示されていない。primary群のsign-flip p値は未補正で、
効果の一般化や因果的説明忠実性を証明するものではない。現状の位置づけは「mask評価パイプラインの
予備的感度分析」であり、「忠実な説明を検証した」ではない。実験が完了したことと、研究仮説が支持された
ことを混同しない。

### 仮に結果を報告する場合の注記案

図X. Grounding区間maskに対するchoice-margin感度。点は平均selective necessity margin drop、横線は
pointwise percentile-bootstrap 95%信頼区間を示す。正の値は、evidence位置をマスクしたときのgold対
best-foil margin低下が、同数の時間位置対応controlをマスクした場合より大きいことを表す。Black maskと
Gaussian blur（radius 12 px）は同一の60動画で比較した。baseline-correct cohort（事前指定primary）は
n=37、baseline-incorrect cohort（secondary）はn=23。区間はpointwiseで、多重比較補正を行っていない。
blur−blackの対応比較は図中に重ねず、同じ動画に対する差として別途報告する。

### 残る限界と判断

ここで測定したのは、指定promptにおけるgold-vs-best-foil next-token log-probability marginの
介入変化であって、自由生成の品質や説明の正当性ではない。sign-flip検定はsample差の対称帰無を仮定し、
区間・p値はpointwise/unadjustedである。

Frame maskingはblack画像または強いblur画像を入力するため、自然動画から外れた入力を生み得る。
従って現在の分析から、一般的な説明faithfulness、自由生成の正確性、特定attention headの因果性を
主張しない。一枚フレーム置換のfresh confirmationの非再現も、介入とestimandが異なるため、このmask
分析で覆されたとは扱わない。追加mask fillや同じ指標でのGPU探索は停止する。次の研究段階へ進むなら、
説明手法が選んだ根拠と、内容・識別情報量をより厳密に対応させたcontrol、または因果的counterfactualを
比較する別の事前固定研究質問が必要である。その設計と卒論スコープの採否は追加GPU計算の前に決める。

### 再現元

- Black: `results/necessity_choice_margin_replication_heldout.json`
- Blur: `results/necessity_choice_margin_replication_blur_heldout.json`
- 対応fill比較: `results/necessity_choice_margin_replication_mask_fill_comparison.json`
- 図・CSV・生成metadata: `results/necessity_choice_margin_thesis_summary_final_*`
- 解析コード: `scripts/plot_necessity_choice_results.py`

## 3. CLEVRER counterfactual pairのデータ適合性

公式train/validation QAの全数CPU監査で、counterfactual型は27,975問。choice-level programはすべて非空だが、同一sceneのcounterfactual質問がちょうど2問ある12,975組のうち、異なるchoice-program署名を持つのは12,974組だった。そのうち演算列まで一致したのは158組で、質問文token edit距離が2–3だったのは12組、4以上は146組だった。距離は表層文面の粗い記述量であり、12組を意味的に妥当な介入pairとは認定できない。

このため、CLEVRERのペアを使ったfaithfulness実験のデータ適合性は確立できず、このpair探索は停止した。GPU推論、個別例の事後選別、距離閾値の追加調整は行わない。根拠は[方向性レビュー](../experiments/2026-10-08_post_confirmation_direction_review.md)とcounts-only結果（[`choice-level`](../results/clevrer_program_choicelevel_corrected_audit_20261008.json)、[`operator trace`](../results/clevrer_program_operator_trace_corrected_20261008.json)、[`question token distance`](../results/clevrer_pair_question_token_distance_20261008.json)）。

## 総合的な主張範囲

現時点で直接報告できるのは、固定したhead-faithfulness estimandが事前の実質効果基準を満たさなかったこと、grounding-mask probeが一部の事前指定cohortで時間位置対応controlとの差を示した一方、全60件では区間が0を含み、controlの意味情報量やmask OODも未統制であること、そしてCLEVRER構造pairが意味的に妥当と確認できなかったことである。

従って「モデルの説明が忠実だと実証した」「faithfulness一般がない」「grounding区間maskの反応が因果的faithfulnessを証明する」とは書かない。卒論の主題を確定する前に、指導教員と相談し、負の独立確認と評価手法の限界を主軸にした結果報告として成立するか判断する。新規実験を提案する場合は、別のestimand・妥当性基準・データ成立性を結果閲覧前に定義する。
