# 旧研究計画書（履歴）：動画VLMのheadベース説明は予測に忠実か

> **状態：旧head介入計画。** 事前登録した独立確認は成功基準を満たさず、この計画のhead/layer/alpha探索は終了した。実験手順を再開しない。数値・手順は過去の設計と結果の来歴を保つために残す。現行の提案は[内部介入によるgrounding/faithfulness研究計画](internal_intervention_grounding_faithfulness_plan.md)を参照する。

## 1. 研究の目的

本研究の最重要目的は、動画Vision-Language Model (VLM) が出す説明が、そのモデル自身の予測を実際に支えているか（説明の忠実性、faithfulness）を検証することである。CLEVRER-Humansの因果edge QAを研究目的へ置き換えない。v6/v7監査で分かった候補数はデータ適合性の情報にすぎず、faithfulnessの証拠ではない。

### 研究質問

> Qwen2.5-VLが自分の回答に重要だと評価したattention head群を介入で弱めると、同じ層構成・同じ規模・近い活性値を持つ対照head群を弱めた場合より、その回答への確信が大きく低下するか。

ここで評価する説明は、自然言語の理由文ではなく、「この質問への回答で重要なheadはどれか」というhead群の説明である。NExT-GQAの時間区間ラベルは、head群が人間の根拠時間に注意を向けているかという別の妥当性確認に使う。人手区間へのマスク結果だけで、モデル説明のfaithfulnessを結論しない。

## 2. 用語と数値の定義

### 2.1 Faithfulnessと妥当性は別の問い

- **Faithfulness（忠実性）**：説明器が重要と示したheadを介入で弱めたとき、モデルの元の回答が、対応する対照headへの介入より大きく影響を受けるか。
- **時間的妥当性（grounding validity）**：説明対象headの注意が、人手で付けられた根拠時間区間に向いているか。
- **回答正解率**：モデルの回答がデータセットの正解と一致する割合。モデルの説明が忠実かどうかとは別に報告する。

回答が誤っていても、そのモデルが実際に使った情報を説明できていれば、説明はモデルの判断に対して忠実であり得る。そのため、primary分析を「介入前に正解した例だけ」に限定しない。

### 2.2 回答スコア

各質問には選択肢A〜Eを使う。各選択肢の最初の回答tokenの対数確率を`l_j = log p(choice j | video, question)`とする。選択肢tokenが一意に対応しない質問は、推論前のCPU preflightで除外し、件数と理由を報告する。

介入前にモデルが選んだ回答を`ŷ = argmax_j l_j`とし、回答marginを次で定義する。

`m(x) = l_ŷ(x) - max_{j != ŷ} l_j(x)`

marginが大きいほど、モデルは元の回答`ŷ`を他の選択肢より強く支持している。すべての介入条件で`ŷ`は介入前に固定し、介入後に変わった回答へ差し替えない。

### 2.3 head説明と介入

質問ごとに、回答marginへの勾配とhead出力の積で各headの重要度を計算する。head`h`の回答位置における出力を`z_h`として、

`a_h = | z_h · (∂m / ∂z_h) |`

と定義する。値が大きいheadほど、その出力を少し変えたときmarginが変化しやすいという局所的な目安である。これは単独では因果効果ではないため、後述のhead介入で検証する。

- 説明群`S_E`：`a_h`が最大の4 head。K=4は、全784 headのうち0.51%を示す疎な説明として、確認データを見る前に固定する。
- 対照群`S_C,r`：各説明headと同じlayerから、説明群に含まれないheadを選ぶ。head数とlayerごとの個数を一致させ、各head出力の大きさも近いものを選ぶ。乱数seed固定で5組を作る。
- 介入：回答を予測する最終prompt位置におけるhead出力を`z'_h = α z_h`へ変更する。primary doseは`α=0.5`（半分に弱める）。`α=0`（hard ablation）と`α=0.75`はsecondaryのdose-response分析に限る。

介入は動画フレームを黒塗りせず、モデル内部のhead出力に適用する。これにより画像マスク由来のOOD（学習時と大きく異なる入力）問題を避ける。ただしhead出力を変えること自体も内部状態を通常分布からずらし得るため、層・head数・活性値を対応させたcontrolが必要である。

### 2.4 Primary評価値

head集合`S`を弱めたときのmargin低下を

`D_i(S) = m_i(full) - m_i(intervene on S)`

とする。サンプル`i`ごとのfaithfulness差を

`d_i = D_i(S_E) - mean_r D_i(S_C,r)`

と定義する。`d_i > 0`なら、説明器が選んだhead群の介入は、対応する対照head群の平均より元の回答marginを大きく下げたことを示す。全動画を統計単位として平均`θ = mean_i(d_i)`を推定する。複数QAが同じ動画に含まれる場合も、primaryでは動画ごとに1 QAだけを使い、動画内の重複を独立標本として数えない。

事前の実質差目安を`δ = 0.25` nat（自然対数単位）とする。margin差0.25は、回答と最有力対立回答の相対確率比が介入前後で約`exp(0.25) = 1.28`倍変わる規模に相当する。これは普遍的な境界ではなく、この研究で「小さすぎて実用上意味が薄い差」と区別するために事前固定する設計値である。

**Primary成功基準**：実質差の境界`δ=0.25`を帰無仮説側に置き、`H0: θ ≤ 0.25`に対して、両側95% t信頼区間の下限が`+0.25`を上回ることを要求する。この判定なら点推定も必ず`+0.25`を超える。単に0を上回るだけでは成功としない。点推定、95%区間、全サンプルの`d_i`分布、正負割合を必ず併記する。baseline-correct群だけの結果でprimary判定を置き換えない。

### 2.5 時間的妥当性とhead間相互作用（secondary）

- **時間的妥当性（secondary）**：回答を出すquery位置から各video frame tokenへのattentionを、各head内でvideo token全体に対して合計1となるよう正規化する。時間対応表を使い、根拠区間内のframe tokenへの重み合計をheadごとに求め、選択head `S_E`内で平均した値を`C_i(E)`とする。同じ層構成・head数のmatched control群ごとにも同じ値を求め、その平均を`C_i(C)`とする。`g_i=C_i(E)-C_i(C)`、`φ=mean_i(g_i)`を報告し、`φ`の95%区間下限が0を上回るかをsecondary判定に使う。均一注意の値は参考値として示す。intervalはhead選択に使わない。この指標は注意の時間的位置の一致であり、それ単独では説明のfaithfulnessを示さない。
- **head間の非加算性**：少数サンプルで、群介入効果と個別head効果の和の差`I_i = D_i(S_E) - Σ_{h in S_E} D_i({h})`を計算する。`I_i > 0`はこのスコア尺度上で群介入が個別効果の和を上回ること、`I_i < 0`は下回ることを表す。ただし、非線形モデルにおける非加算性を、そのまま生物学的・機構的な「協調」やsynergyの証明とは呼ばない。

時間的妥当性とprimary faithfulnessの両方が支持された場合に限り、「このモデル・データ・head説明法の範囲で、時間的にも妥当で、予測に因果的に関係する説明だった」と限定的に述べる。片方だけなら、もう片方まで主張しない。

## 3. 既存結果が示すこと・示さないこと

1. NExT-GQAのblack/blur probeでは、baseline-correct primary群`n=37`のselective margin差がblack `+0.3345`（95% pointwise CI `[+0.1115,+0.5811]`、sign-flip `p=.0053`）、blur `+0.4561`（`[+0.1723,+0.7601]`、`p=.0039`）だった。一方、全60件ではblack `+0.2958`（`[-0.0458,+0.6146]`、`p=.0823`）、blur `+0.3604`（`[-0.0354,+0.7208]`、`p=.0595`）で区間は0を含む。加えて、このprobeはモデル説明が選んだhead/時間区間ではなく人手区間をmaskしており、controlも時間位置のみの対応だった。よって「人手区間maskへの限定的感度」であり、説明faithfulnessの実証ではない。
2. head singleton/group/activation patchingの探索では、独立確認後も特定headまたはgroupが安定して説明回路として選ばれたとは言えない。`layer=6/head=16`は確立した結論として再利用しない。
3. CLEVRER-Humans v5で明示的なframe locatorは見つからず、v6/v7で分かったpositive/negative edge候補の数は因果QAの可用性に限られる。時間区間GTがないため、現計画のtemporal validity評価用データを置き換えない。

このため、既存データを見た後にhead・layer・subgroup・mask条件を選び直して有意差を追わない。新しい説明スコアと介入を実装し、開発用データで設定を固定してから、未使用動画で一度確認する。

## 4. 実験手順

### Stage 0：プロトコル・manifestの固定（CPU）

- NExT-GQAの`val.csv`/`gsub_val.json`を開発用、testを確認用に分ける。動画IDで分離し、過去の全sample manifestで使用した動画IDを除く。公式データ説明ではvalidation CSVと対応する時間区間JSONが提供されている。validationがremoteにない場合はtestで代用せず、データ配置を整える。
- 各動画からQAを1問だけ、question typeを記録して固定seedで抽出する。正解・選択肢・grounding intervalを使って説明headを選ばない。
- 確認manifestは推論前に作り、動画存在、tokenization、option token一意性、16-frame sampling、head-output位置へのhook、各説明群に対する5組のcontrol作成可否を検査する。有限なbaseline marginは3動画smokeで確認する。条件不足の件数と理由を保存し、スコアを見てから補充しない。

### Stage 1：実装・smoke test（GPU、3動画）

- Qwen2.5-VL-7B-Instruct、4-bit、bfloat16、RTX 5070 Tiを予定条件とする。実機のtorch/transformers/bitsandbytes/CUDA versionとGPU名をmanifestに記録する。
- 16-frame samplingは維持し、Qwen video processorへ`cap_pixels_per_frame=True`を渡す。適用後の処理条件をrun manifestに記録する。これは未完了の初回smokeをメモリ制約下で再試行するため、今回の実装・以降の確認条件として固定する。
- 勾配計算では重みをfreezeし、Transformersのnon-reentrant gradient checkpointingを有効化する。これは推定対象（入力活性への勾配）を変えず中間活性を再計算するメモリ対策として用いるが、GPU smokeでhook勾配の成立を確認するまでは有効性を仮定しない。
- `o_proj`入力のhead活性をhookするため、attention重み自体を必要としない。全attention行列を明示的に保持する`eager`ではなく`sdpa` backendを使い、全条件で固定・記録する。数値差とhook互換性はGPU smokeで確認する。
- 3つの新規動画で、full margin、gradient×activation attribution、top-4選択、α=0.5の介入、5 controlが有限値で動くか確認する。frame-token mappingの検証は時間的妥当性secondaryの実行前に別途行い、primary-only pilotの開始条件にはしない。
- smokeは配線確認専用で、primary統計には含めない。異なる選択肢token、動画token位置ずれ、baseline再現性不良があればheld-outへ進まない。
- 説明器のsanity checkとして、独立確認に入る前に開発用の少数例でモデル重みを層単位にランダム化したときのhead重要度順位を保存し、学習済みモデル時の順位とのSpearman相関を報告する。説明がほぼ不変なら説明器が学習済みモデルの情報を反映していない疑いとして停止・調査する。この確認はprimary効果検定ではない。検証pilotはGPU時間・実行安定性・分散の校正だけに使用し、faithfulnessの証拠や方法選択の成績として扱わないため、このsanity checkの完了をpilot開始の前提とはしない。pilot後に行ったA-E出力readout行だけの限定診断では、3動画×5乱数化で平均head-rank Spearman ρ=0.466、top-4重複率=0.067、回答選択は15回中12回変化した。少数の探索的結果であり、全層は未ランダム化、また乱数行のスケール差も残るため、モデル重み依存性や忠実性の結論には使わない。次のsanity checkとして、4-bit層重みを再初期化・再量子化し、最終層から入力側へ累積ランダム化する実装を用意した。

**実施状況（2026-10-08）:** 出力層と累積全28層のrandomization sanity checkはGitHub上のartifact/manifestで完了を確認した。全層チェックは3/3 validation-development動画で、平均Spearman ρは深さ1の0.627から深さ28の−0.286、top-4重複率は0.167から0、回答選択は3/3で変化した。これは学習済みdecoderを再初期化する強い破壊的摂動への反応であり、primary faithfulness estimandの効果ではない。3例の記述統計に留め、独立確認へ進む根拠とはしない。次は追加GPU実験ではなく、[再設計プロトコル](research_redesign_protocol.md)に従うCPU設計レビューとする。

### Stage 2：開発pilot（validation、30動画）

- GPU時間、activation/marginの安定性、選択headのlayer分布、matched controlの作成率、gradient attributionの再現性を確認する。
- primary仮説検定には使わない。K=4、α=0.5、control 5組、score、failure判定はpilot後に確認データを見ずに固定する。pilot結果が弱いことを理由にKや閾値を変更して同じhypothesisを追わない。
- paired差`d_i`の標準偏差`s`をサンプルサイズ設計に使う。pilotの平均効果を確証データの効果として引用しない。

### Stage 3：独立確認（test、新規動画）

- 標本数は`H0: θ≤0.25`に対し、真の効果が`θ=0.50` nat（境界から0.25 nat上）、両側95%信頼区間下限が0.25を上回る確率を80%にする設計を起点にする。paired平均の近似式は
  `n ≈ ((1.96 + 0.84) × s / (0.50 - 0.25))^2`
  である。pilotで`d_i`の標準偏差`s=.75`なら`n≈70.6`、切り上げて71動画となる。最終数はpaired t-testの正確なpower計算またはシミュレーションで確定する。この計算例は説明head介入の実測値ではなく、black/blur probeの標準偏差を流用しない。80% powerは、効果が真に0.50 natならこの成功基準を満たす確率を指す。
- 有効な確認標本の目標は71動画（pilot`s=.75`の計算例）。事前preflight不適格分を考慮し、確認manifestには10%増しの79動画を用意する。ただし推論後のfailureに対する都合のよい差し替えは禁止。最終有効数が71を下回った場合はpower設計未達として報告し、同じ確認データに後付け追加しない。
- 全適格動画をprimaryに含め、介入前の正誤でprimary cohortを選ばない。gold accuracy、baseline-correct/incorrect別、question type別はsecondaryとして報告し、secondary family内はHolm補正する。
- 一つの主primary contrastについて、動画単位の`d_i`から平均`θ`の両側95% t区間を計算し、下限が0.25を超えるかで判定する。これは境界`θ=0.25`に対する両側`α=.05`の検定と対応する。paired bootstrap CIと、境界0.25を差し引いた値に対するsign-flip検定を頑健性分析として併記し、検定仮定が異なることを明記する。

## 5. 判定基準と停止条件

- **faithfulnessを限定的に支持**：primary基準（95% CI下限>0.25 nat）を満たす。結論は「選択されたhead群への内部介入が、対応するcontrol群への介入より元の回答marginを実質的に大きく下げた」という範囲に限る。さらに時間的妥当性のsecondary基準（`φ`の95% CI下限>0）も通った場合にだけ、根拠時間との一致も併記する。いずれもQwen2.5-VL-7B、NExT-GQA、指定prompt・frame sampling・head説明法に限定する。
- **内部寄与のみ**：head介入primaryは通るが時間的妥当性が通らない場合。説明headが予測へ寄与したとは言えるが、根拠時間を忠実に説明したとは言わない。
- **時間的一致のみ**：根拠区間とのattention一致は見られるがprimary介入差が通らない場合。attention位置の妥当性だけでfaithfulnessを主張しない。
- **不支持/不確定**：primary CI下限が`0.25`以下、必要有効数に届かない、またはmatched control/実装が成立しない場合。不確実性が残る結果を「効果なし」と断定せず、区間と有効数を示す。不支持・不確定なら追加のhead・K・α探索で同じ確認データを再利用しない。
- group nonadditivityはsecondary/exploratory扱い。多重性を補正した独立確認なしに「head間synergy」と結論しない。

## 6. 再現性・安全性・実行コスト

- 各結果にGit commit、seed、入力manifest全体のSHA-256、匿名sample key、split、16-frame sampling設定、model revision、quantization/dtype、GPU/software versions、head set、control sets、α、除外件数を記録する。公開artifactにQID・動画ID・質問文・選択肢・絶対パスは含めず、生manifestはローカルに保管する。
- 生動画、annotation pickle、質問文/選択肢本文、絶対パスはGitHubへ公開しない。公開するJSONは統計要約と匿名sample keyだけにする。
- 本確認80動画の場合、各動画でfull/gradient forward、説明群1条件、control 5条件だけでも数百forward相当となる。group-vs-single interactionを加えるとさらに増える。GPU所要時間は未測定なので、3動画smokeでwall timeを測ってからpilot/fullの見積りを出す。長時間runは事前に期待時間を提示し、smoke成功後にのみ依頼する。
- smoke/ pilot/ confirm JSONは別ファイルとし、publisher wrapperで統計要約だけを個別にcommit/pushする。結果を確認後、失敗数・有効数・条件・区間・多重性を実験ログへ追記する。

## 7. 現時点での結論

primary head-attribution smokeはtest側3件で初回OOM後、設定調整により3/3完了し、平均+0.125 nat（95% t区間[−0.282,+0.532]、境界0.25に対するp=.317）だった。別に、公式validation上の新規smokeも3/3完了（平均+0.108、95% t区間[−0.266,+0.483]、p=.245）した。続くvalidation development pilotは30/30完了・失敗0、`d_i`平均−0.0225 nat、SD 0.3774、95% t区間[−0.1634,+0.1184]、境界0.25に対する両側t検定p=.00045で、CI下限は事前の成功境界を下回り、上限も0.25未満だった。pilotは非確認的なので、このp値を独立確認の代わりに使わない。A-E出力readoutだけをランダム化した3動画診断（各5乱数化）は全件完了し、head-rank Spearman平均0.466、top-4重複率0.067、回答選択変化12/15だった。続く累積全28層randomizationも3/3で完了し、平均rank Spearmanは深さ1の0.627から全層置換後の−0.286、top-4重複率は0、回答選択は3/3で変化した。いずれも破壊的摂動に対するsanity checkでありfaithfulness効果ではない。K・alpha・control・指標をpilot結果に合わせて変更せず、独立確認にも進まない。次は追加GPU実験ではなく、primary estimandとmatched control、独立確認条件をCPU設計レビューで再固定する。時間的妥当性分析はframe-token mapping確認後に限る。詳細と再計算コードは[実験ログ](../experiments/2026-10-08_head_faithfulness_implementation.md)。

### 独立確認後の更新 (2026-10-08)

上記の「独立確認にも進まない」は、当時のpilot時点の記述である。その後、事前計画どおり別動画79件の固定protocol確認を実施し、79/79件成功・失敗0だった。主estimandは`−0.0756 nat`、SD `0.3667`、95% t区間`[−0.1578,+0.0065]`で、成功境界`+0.25 nat`を支持しなかった。したがって、当該head-faithfulness仮説はこのモデル・設定の範囲で不支持として閉じ、同一確認データでのhead/layer/alpha再探索を行わない。解釈、数値定義、限界は[独立確認結果](head_faithfulness_confirmation_results.md)を参照。

## 参考文献・データ仕様

- Xiao et al., “Can I Trust Your Answer? Visually Grounded Video Question Answering,” CVPR 2024. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.html). NExT-GQAの時間区間ラベルは根拠時間への妥当性確認に使うが、モデル説明の忠実性そのものとは同一視しない。
- Adebayo et al., “Sanity Checks for Saliency Maps,” NeurIPS 2018. [NeurIPS proceedings](https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). 説明手法が学習済みモデルの情報を反映するかを検査するsanity-checkの動機として参照する。本研究への適用可能性は実装時に別途検証する。
- CLEVRER-Humans official [data loader](https://github.com/xyang23/CLEVRER-Humans1.0/blob/main/description-gen/dataloader.py). 因果edge候補のデータ仕様確認用であり、本計画の時間grounding評価の代替ではない。
