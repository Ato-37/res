# 研究計画書（提案）：内部介入による動画groundingとfaithfulnessの改善

作成日：2026-10-09
状態：提案段階。データ適合性ゲート前であり、実験承認・実験結果ではない。
日程：10月後半〜11月前半を成立性確認に充てる。提出日・GPU枠は未確認のため、以後は相対目標とする。

## 1. 研究目的と問い

Qwen2.5-VLの推論時内部介入により、回答claimを支える時間的視覚証拠の局在を改善し、そのclaimが該当証拠に選択的に依存しているかを検証する。

次の二つを別の成果として扱う。

- **時間的grounding精度**：モデルが回答根拠として示す動画時間区間が、NExT-GQAの対応するgold intervalとどれだけ一致するか。
- **Faithfulness（証拠依存）**：そのclaimに関係する内部時間chunkを介入したとき、同じ規模のmatched control chunkを介入したときより、元の回答・claimスコアが選択的に低下するか。

正答率・hallucination benchmark・attention一致の単独改善をfaithfulnessと呼ばない。卒論で両方の向上を主張するには、両評価の事前規定条件を満たすことを要求する。

> 主研究質問：質問に応じて選ばれた時間visual evidenceを中間層で強調すると、時間局在精度は改善するか。その改善は、根拠claimの選択的な内部証拠依存を伴うか。

対象は長い自由形式CoT全体ではなく、一つの検証可能な回答claimと時間区間に限定する。これは旧head群faithfulness実験とは異なる新estimandである。

### 2026-10-17：卒研の中心対象を明確化

卒研の中心対象は、**モデルの説明が指す動画証拠に、説明対象の予測が選択的に依存するか**とする。主指標候補は、その選択的依存が**事前に固定した内部介入によって向上するか**である。したがって、単なるQA回答marginの動画入力感度やgrounding精度は主指標ではなく、補助的な測定となる。

結果閲覧前に固定する測定候補を次のように置く。`g`はbaselineまたは内部介入、`m_{g,i}(z)`は動画iの固定した元予測と固定foilのteacher-forced log-margin、`z`はfull、claim-evidence internal ablation、matched-control internal ablationのいずれかとする。

`F_{g,i} = [m_{g,i}(full) - m_{g,i}(evidence-ablation)] - [m_{g,i}(full) - m_{g,i}(control-ablation)]`

`ΔF_i = F_{intervention,i} - F_{baseline,i}`

正の`ΔF`は、指定モデル・内部状態の介入条件下で、説明が指すレビュー済み時間証拠の選択的依存がmatched controlに対して増えたことを表す操作的指標候補である。これは説明の真偽、完全な内部推論、自由生成説明全体の忠実性を証明しない。claim-to-timeの独立レビュー、時間区間とprocessor visual-tokenの検証、同じ回答/foilの固定、介入とablation双方のsham/control、layer/strengthの事前固定が必要である。式、介入、token score、成功基準、sample sizeはまだ凍結していない。

比較対象のbaseline生成回答・claim・時間区間は、全`g`条件で固定する。介入条件下でclaimを再生成すると主estimandの対象が変わるため、それを「同じclaimの証拠依存改善」と混ぜない。介入後の自由生成claimの品質やgroundingを測る場合は、別の副次評価群として報告する。また、もしKFP系処置を採るなら、key frameのquestion-conditioned選択は一度、通常のfull-video入力で行い、同じ選択をbaseline/treatmentと各ablation条件へ固定する必要がある。ablationごとに再選択すると、証拠除去とselectorの変化が混ざる。

**介入の循環性チェック（2026-10-17）。** 説明が指したtoken残差そのものを増幅し、同じtokenのablationで`ΔF`を評価する案は、選択対象と評価対象が一致して正の結果を操作的に作り込むおそれがある。この案は主介入候補から外し、残す場合も実装のpositive controlに限る。主介入はclaim spanやgold intervalを介入先に使わず、質問依存の内部信号だけで時間chunkを選ぶ候補を検討する。claim-to-time対応は評価専用に保つ。選択信号・層・強度を固定する根拠と、全体的なlogit/confidence上昇では説明できない対照が定まるまで処置は凍結しない。

既存のv2 black-mask studyはこの内部介入estimandを測らないため、別の準備用feasibility branchに留める。広い「動画VLM初のactivation engineering」はTA-AE (NeurIPS 2025)等と重なるため新規性として主張しない。LLM向けActivation Patching/Causal Faithfulness研究は方法論上近いが、同じclaim/time-mapped visual tokenへの依存向上を動画VLMで測る設定とは区別し、本文・実装の差分を確認する。

## 2. 根拠と引き継ぐ制約

- 旧head-faithfulness独立確認は79/79件完了、推定値−0.0756 nat、95% t CI [−0.1578,+0.0065]で、事前境界+0.25 natを支持しなかった。gradient×activation top-4 head仮説は閉じる。head/layer/alpha探索を再開しない。
- 既存NExT-GQA mask結果は人手区間への感度であり、モデル選択証拠や説明faithfulnessの確認ではない。mask OODとcontrol情報量不一致も残る。
- CLEVRER監査は意味的に妥当なvideo counterfactualを確立していない。単純maskをcounterfactualと呼ばない。
- 先行研究の一次資料を追加確認する。TA-AE（NeurIPS 2025）はQwen2.5-VLでactivation engineeringによる動画hallucination低減を報告し、CRA（CVPR 2025）はVideoQGに明示的因果介入を含む。したがって内部介入や動画grounding改善自体は新規性として主張しない。NExT-GQAのgold temporal IoUとclaim/explanation faithfulnessを同一の推論時介入で検証する問いは候補に留め、文献差分・注釈妥当性を確認するまで介入を固定しない。詳細は[追加監査ログ](../experiments/2026-10-09_round2_time_sampling_and_prior_art.md)。
- Related-work update (2026-10-15): Menzat et al.'s visual CT/CCT study (reviewed arXiv v1; its record reports acceptance to Findings of EMNLP 2026) is a close image-VLM precedent: it relates mentions of edited objects to candidate-normalized answer-distribution shifts and includes an image-reconstruction control. Cao et al. (arXiv v2) provide a particularly close video-temporal control precedent: they hold a user-provided order claim fixed and compare removal of event frames against removal of the same number of other frames at fixed budget. Javadov et al. test modality self-explanations against executed ablation/restoration; EDCT/EDCT-Bench, INFACT, FIFA, EG-VQA, Evidence-Backed Video QA, CausalPhys, and AAAI 2026 temporal-pathway work cover other adjacent estimands. The possible distinction here is limited to a model-generated, reviewer-mapped temporal claim interval and a rank-matched mask-control contrast scored with condition-specific gold-versus-best-foil margin; the foil may differ by condition. These differences alone do not establish novelty or priority. Preserve the frozen estimand as an operational post-hoc input-dependence measure only; it cannot prove internal-reasoning faithfulness, claim truth, or mechanism. See [Menzat et al.](https://arxiv.org/abs/2609.06704v1), [Cao et al.](https://arxiv.org/abs/2608.03160v2), and the [2026-10-15 PDCA follow-up](../experiments/2026-10-15_pdca_5h_followup.md).

## 3. 介入と条件（候補）

**主介入候補：KFP-inspired temporal evidence propagation。** 生成stepの質問依存な中間層信号からkey temporal chunkを選び、中間層のvisual-token表現を限定された時間近傍内で再重み付けする。まず論文で使われた層範囲を候補にし、Qwen2.5-VLの実層数・video token/chunk構造と対応するかCPU監査する。成立しなければGPUへ進まない。

ただし、[VERHallu/KFP論文](https://arxiv.org/html/2601.10010)の公開記述は層8–15、Gaussian幅`σ=1`、近傍幅`m=5`、混合係数`β=0.6`を示す一方、key-frameの`top-k`値は確認できず、パラメータablationは別モデルを含む。[著者の公開コードリポジトリ](https://github.com/zefanZhang-cn/VERHallu)も2026-10-17時点ではREADME/LICENSEのみで、再現用実装が見当たらない。このため、KFPは根拠のある設計候補ではあるが、そのまま再現できる固定treatmentではない。`top-k`などを本データ上で探索して埋めることはせず、先に仕様・実装の再現可能性を解決する。

- Baseline：介入なし。
- Treatment：論文に沿う単一の固定設定。確認結果を見てtop-k・層・時間幅・強度を探索しない。
- Matched control：同じ層、同数token、同じ強度・演算量で、質問非依存に選ぶ時間chunkへ介入。
- 必要な時間構造control（例：一様強調または順序摂動）はdev段階で条件を定義し、確認前に固定する。
- TA-AEは追加比較候補だが、初回確認には必須としない。複数手法の結果を見て最良手法を選ばない。
- Attention/TAMは介入routing信号候補に留め、因果的寄与や説明忠実性の証拠とは呼ばない。
- 隣接frameへの広い伝播が区間境界をぼかす可能性を、grounding primaryで検出する。

## 4. 評価プロトコル案

### 4.1 cohort

動画単位で評価し、primaryでは1動画1 QAとする。過去のscreening・pilot・confirmation・mask probe全てのmanifestからvideo/vidor IDを除外し、動画単位で開発と独立確認を分ける。確認manifestは推論前に固定・hash記録する。除外後に必要数を確保できなければ、同一データを都合よく再利用せず再設計する。

NExT-GQAのgsub intervalが対象質問・claimに対応するかを先に監査する。human intervalは評価専用で、推論時のchunk選択や閾値調整に使わない。

### 4.2 Grounding primary候補

固定promptでモデルに (1) 選択肢回答、(2) 一つの短く検証可能な根拠claim、(3) 根拠開始・終了時刻を出させる。baselineと介入条件は同一promptを使う。既存の5択margin結果と別タスクとして扱う。

- 主候補：出力区間と対応gold intervalのtemporal IoUを動画単位で平均する。
- Secondary候補：Recall@IoU 0.3/0.5、端点誤差、区間長、回答accuracy。
- 複数gold intervalの集約、timestamp解析失敗・空interval・claim不一致の扱いをCPU監査後に固定する。都合の悪い出力を除外しない。parser失敗は除外せず低スコア扱いにする案を事前検討する。
- gold intervalを介入選択・強度決定へ漏らさない。

このpromptが元データの問いとannotationに適合することが確認できない場合、モデルへ時刻を要求すること自体を主要評価にしない。

### 4.3 Faithfulness estimand候補

baselineで生成した回答claim文字列を固定し、teacher-forced claim/answerスコアをSとする。内部時間chunkに同一強度の介入を行い、model-selected evidence chunkとmatched control chunkの差を比較する。

D(E) = S(full) − S(intervene on selected evidence chunk)
D(C) = S(full) − S(intervene on matched control chunk)
d(i) = D(E) − D(C)

正のdは、このスコア尺度上で、モデル選択chunkの介入がcontrolより元claimを強く弱めたことを示す。これは操作的な選択的証拠依存であり、claimの真実性や機構全体の説明ではない。内部activationのOOD、chunk間依存、prompt依存も限界として扱う。

対象token（answer/claim）、score・符号、matching規則、SESOI（実質最小効果）は独立確認前に固定する。旧+0.25 natを根拠なく移植しない。pilot平均や有意性を見て指標・閾値を選び直さない。

### 4.4 統計

- 同一動画内の条件を対応比較し、動画単位の効果量、95%区間、要求数・有効数・失敗数を報告する。
- SESOIは区間annotation粒度、出力再現性、測定上解釈可能な差から確認データを見ずに定める。
- pilotの動画単位分散からsample size/powerを算定し、条件を事前記録する。pilotは計測安定性・分散・実行時間に限り、確認証拠としない。
- G/F両方をprimaryにする場合、両成功条件を定義する。secondary familyは多重性を調整する。
- correctnessや質問型サブグループを見てprimary cohortを作り直さない。

## 5. 中長期ロードマップ

日付は提案目標であり、提出期限・GPU予約に合わせて調整する。Go/No-goの記録を各段階の成果物に含める。

| 時期 | 目標・成果物 | Go/No-go条件 |
|---|---|---|
| **10/09–10/15：CPUデータ監査** | gsubとQAのjoin率、interval形式・単位・複数区間、質問claimとの対応、既使用video ID、sampling/chunk時間解像度を匿名集計 | annotationが対象claimに使えない、重複除外後にcohortが組めない、時間対応が定義不能ならNExT-GQA primaryを保留 |
| **10/16–10/23：estimand固定** | prompt、claim形式、G/F primary、介入/control、除外、SESOI、power、解析コード仕様をversioned protocolに固定 | claim/spanが安定出力されない、matched internal controlが成立しないなら実験承認しない |
| **10/26–11/06：実装・smoke** | CPU synthetic/unit tests、Qwen video-token/time mapping、middle-layer hook、timestamp parserを検証。ゲート通過後、remote GPU最大3動画smoke | NaN/OOM/mapping不整合/範囲外timestampは停止。smokeは効果検定に含めない |
| **11/09–11/15：開発pilot・審査** | pilotで失敗率・分散・時間・介入動作を確認し、介入一つと独立confirmation標本数を固定。Go/No-goを記録 | pilot効果の向き・有意性で指標や手法を選び直さない。曖昧ならNo-goまたは問いを狭める |
| **11月後半–2027年1月：独立確認（条件付き）** | 一回のheld-out確認。匿名summary、manifest、解析コード、dated logをpublish | 標本不足・失敗偏り・primary未達をそのまま報告し、追加探索でtestを再利用しない |
| **2027年1月以降：再現・執筆** | 図表再生成、方法・結果・限界の記述、可能なら独立再現 | 再現できない効果は確認済み範囲に限定。締切に間に合わない未検証主張を加えない |

**10月〜11月前半の到達目標は卒研完了ではない。** 新しい問いを実行可能な形で固定し、測定・対照・標本設計が成立するかを判断するところまでである。

## 6. 再現性・安全性・停止条件

- Qwen2.5-VLはbfloat16を使用し、4-bit時もcompute dtypeをbfloat16とする。model revision、seed、GPU/software versions、sampling、条件、requested/effective n、失敗をmanifestに残す。
- GPU長時間run前にsmokeを実施し、推定時間・VRAM・対象範囲を提示する。runごとにunique results pathを使い、結果と必要manifestだけをstageする。
- 動画・元annotation・質問/選択肢本文・識別子・絶対path・credential・weightsはGitHubへ出さない。ユーザー指示（2026-10-15）によりローカル限定の保護データ/モデル処理は権利確認ゲートなしで実施可能とする。権利状態は未確認のまま記録し、外部共有・再配布は行わない。
- 次のどれかなら確認実験を停止し、設計を見直す：gold intervalとclaim不一致、chunk-to-time mapping不確定、matched control不成立、smoke不安定、必要標本確保不能。
- 独立confirmationが成功しても、結論は指定Qwen2.5-VL版・NExT-GQA適格質問・prompt・sampling・介入・指標に限定する。他モデル・自然動画一般・長いCoTへの一般化は別検証を要する。

## 7. 次に行う作業

1. **完了（2026-10-09）**：counts-only CPU監査を実装・実行。validationは期間内interval 3,355件/3,358件、既使用動画を除いて567動画。testは構造的に有効なinterval 5,517件/5,553件（不正値36件）、既使用動画を除いて575動画。詳細は[監査ログ](../experiments/2026-10-09_temporal_grounding_data_audit.md)と公開済みsummaryを参照。
2. **完了（2026-10-09）**：全CSV video IDでgsub/frame2time/video map keyが一致。duration/構造上有効なgold intervalの格子内包含率はvalidation 98.17% (3854/3926)、test 98.77% (6479/6560)。Qwen2.5-VL processorはsynthetic 8-frame CPU smokeでFPS・4 temporal grid steps・token数の整合を確認。詳細は[time-grid/Qwen記録](../experiments/2026-10-09_time_grid_and_qwen_mapping.md)と公開summaryを参照。実データ/Qwen token対応やgrounding効果の確認ではない。
3. **完了（2026-10-09）**：公式Qwen2.5-VL-7B-Instructをrevision cc594898137f460bfe9f0759e9844b3ce807cfb5で取得し、合成4-frame・1-videoの4-bit attention forwardを完了。requested/effective 4/4、1/1サンプル、ピーク割当6.446 GiB。合成overlap値は破棄し、grounding結果とは扱わない。詳細は[forward smoke記録](../experiments/2026-10-09_round3_synthetic_qwen_forward.md)。
4. **ローカルpilot完了（2026-10-15）**：ユーザーが権利ゲートをローカル実験の前提から外し、保護対象のローカルデータ読み取りと推論を許可。NExT-GQA実動画で16/32フレーム・黒塗り/ぼかし対照の探索的QA-marginを実行。効果符号はコホート間で反転し、claim-level faithfulness効果は0件。全85候補のPTS監査では32-frame interval hit 98.6%、64-frame 100%だが、幾何学的hit率でありfaithfulnessではない。権利状態は未確認、すべてローカル、公開なし。次はモデル生成claimの対象区間レビューとclaim-score/control estimandを固定してから実験する。hit-rateをfaithfulnessと呼ばない。詳細は[2026-10-15実験ログ](../experiments/2026-10-15_pdca_5h_followup.md)。
5. Added literature gate (2026-10-09): CRA/TA-AE overlap with the broad goal; ICLR 2026 image-VLM intervention and the STEAR video-intervention preprint further narrow novelty. No intervention-method novelty claim is currently supported. Check annotation/claim fit, and consider a replication/generalization study with advisor input. See experiments/2026-10-09_round4_proxy_metric_audit.md.
