# 研究方針・設計の概要

このファイルは、コーディングエージェント(Claude Code, Codex等)がこのリポジトリに初めて触れる際に読む前提の文書。実験の進捗・個別の結果は[../experiments/](../experiments/)配下の各ファイル(日付順)を参照。このファイル自体は「なぜこの設計なのか」という、コードや個々の実験ログからは読み取れない背景を記録する。

## 研究テーマ

動画VLM(Qwen2.5-VL)における**忠実性(faithfulness)の内部介入による診断と補正**。熊本大学の卒業研究(卒研)。

対象はドメイン非依存(動画の内容が事故/医療/スポーツ等であることは本質ではない)。人間被験者アンケートやLLM-as-judgeによる「納得感」評価は、確証バイアスや動画を見ずに評価できてしまう問題があるため不採用とし、忠実性の内部的な検証一本に絞っている。

### 現行の提案方針（2026-10-09）

2026-10-18 QTA-Bias方向の更新: 卒研の主張は、説明が示す動画証拠への固定予測依存を内部介入で高められるかに置く。対象候補はbaselineのp/qと説明claimを独立に結び付け、QTA-Bias下のvisual-key evidence ablationとrank-distance control ablationの差をbaselineとの差分`ΔF`で測る草案である。動画claim reviewの先行受領書hash、各行の`supports_claim`、reviewed-interval hashをprediction-link packet/receiptに束縛し、stage 2を別担当者による条件付きテキスト識別判定と明確化した。またreview player時刻からdecoder PTSへの較正式と逆変換、1 ms区間制約、guard内で根拠group集合が変わる行を落とす厳密有理数CPU経路を整備し、processor-batch adapterにguard省略拒否を通した。新しい三行pre-score joinとreview/processor/clock row adaptersは独立ピン済みreview payload・動画・batch・sampling・較正payload・質問/選択肢/claim hashes、anchor ID/時間形式、temporal group範囲とevidence/control割当を照合するが、構造検査だけで採点を許可しない。stage-1統合監査を追加し、worksheet・lineage・custody・両HTML・3本のmediaの同一バイト結合、release/reviewer署名の検査を合成fixtureで確認した。独立監査後、時計/frame snapshotのTOCTOU・重複画像・資源上限とstage-1 bounded-read分岐を補強し、全scripts suite **1,549件**を再検証した。さらにrow binder envelopeを固定する非認可wrapperを追加し、不整合なsuccess summaryを拒否するテストを含め全scripts suite **1,553件**を通過した。このwrapperは入力来歴を検証せず、stage-1、clock joint-freeze、prediction-linkの本人性・真正性検査を束ねていない。stage-1は依然非認可で、trust-root/releaseと時計を含むcohort-wide production gateおよびclaim-mapped実データscorerは未完成。直接の実データ進捗は動画claim review **0/3**、prediction-link adjudication **0/3**、採点**0/3**、player-clock/decoder対応**0/15**、実`ΔF`/CI/testなし。詳細は[2026-10-18 QTA audit log](../experiments/2026-10-18_qta_checkpoint_hash_and_construct_link_audit.md)と[protocol draft](qwen_question_attention_faithfulness_protocol_v1_draft.md)。

2026-10-18 追記: 固定予測scorerを9-forward synthetic QTA runnerへ接続し、同じmultimodal batchとbaseline-full固定p/q、rendered prompt由来の回答token ID、全条件共通の最終score位置を採点前に束縛した。A–E各ラベルを同一promptに連結したprocessor batchでprefix全体が保持されることも実行時に検査する。独立監査後のGPU smokeはsynthetic 1件・9/9 forward、51.046秒、peak allocated/reserved 10.755/12.439 GiBで完了したが、synthetic `ΔF=+1.19×10⁻⁷ nat`は数値確認のみで`faithfulness_result=false`。最新scripts全体テストは**1,559/1,559**。安全性検査後の公開可能な[synthetic GPU summary](../results/synthetic_qta_bound_scorer_gpu_20261018_0713.json)を保存した。実動画のclaim review・prediction-link・採点・player-clock/decoder対応はそれぞれ**0/3・0/3・0/3・0/15**のままで、root pin、独立役割の本人対応と立会handoffがないためreview packageは未開封、実データ推論は未許可。詳細は[QTA scorer integration log](../experiments/2026-10-18_qta_bound_scorer_runner_integration.md)と[human-review runbook](nextgqa_v6_human_review_runbook.md)。

2026-10-18 treatment design: reviewed evidence区間を直接biasする処置は、同じ区間のablation比較を機械的に押し上げうるため主処置候補から除外した。質問token attentionから独立に選ぶQTA-Biasを、次のclaim-linked feasibility runの唯一のprimary-treatment候補として選定し、layers 5–16、`ceil(0.10*T)` complete temporal groups、`ln(2)` biasを固定した。selectorとreviewed evidence/controlの重なりは全行で記述するが、除外・subgroup・再調整には使わない。これは処置仕様の選択であり、実験プロトコル全体の登録・実データ実行許可・効果検証ではない。詳細は[QTA treatment selection log](../experiments/2026-10-18_qta_treatment_selection.md)と[protocol draft](qwen_question_attention_faithfulness_protocol_v1_draft.md)。

2026-10-17 構成概念監査: 現行v2は生成claimをQA採点promptに含めず、claimで選んだ動画frameをmaskしてQA marginを測る設計である。よって結果が示せるのは3件有限cohortでの「claimに沿って選ばれた視覚入力への回答依存」であり、説明文そのもののfaithfulnessではない。直接generated-explanation faithfulnessは未測定。v2を変更せずこの限定的なfeasibility指標として扱うか、出力を見ない段階で説明内容を直接対象にする別protocolを設計する必要がある。詳細は[構成概念監査](../experiments/2026-10-17_claim_mask_construct_validity.md)。

2026-10-17最新状況: hash-pinned v6 claim-review handoffは現在の3件コホートで準備済みだが、人手レビュー・採点許可は未取得。レビュー開始を採点プロトコルの完全freezeから切り離し、v2の固定済みestimandとレビュー資料だけを固定する「review-only release freeze」と、全レビュー・時計証拠・実採点入力を結ぶ後段のscore freezeに分けた。前者は採点・モデル実行を許可しない。独立した担当者/鍵のtrust rootとreceiptがまだ無く、packageは開封不可のままである。現行候補計画に対し、動画を一度ずつデコードして予定15 anchorのRGB PNGをローカル出力し、独立監査でplan/provenance/PTS/hash結合、ファイル数、0600/0700権限を確認した。Windows capture helperの順序バグを修正し、5%・25%・50%・75%・95%の順、fit/heldout分離、token重複を検証する。Windows ValidateOnlyと非撮影PermissionSmokeOnlyは成功。全画面captureから15個の矩形を切り出すprivate cropperも準備し、合成テスト9/9と独立監査を完了したが、実画面capture・crop・decoderとの人手照合は未実施で、captureはEdge上の人間操作が必要。cropperは矩形の動画のみ純度を判定しない。実動画Qwen CUDA smokeは4フレームと32フレームで成功し、32フレーム時はpeak allocated 10.50 GiB、処理後に14.85 GiB freeを確認した。これはfaithfulness測定ではない。直接claim-faithfulnessは0/3人手レビュー、0/3採点、効果量・CI・検定なし。production atomic gateと人間/時計証拠が未達のため9-forward claim-faithfulness実験はまだ実行できない。詳細は[human-review runbook](nextgqa_v6_human_review_runbook.md)、[real-video GPU smoke](../experiments/2026-10-17_real_video_gpu_path_smoke.md)、[decoder-anchor export](../experiments/2026-10-17_clock_anchor_decoder_export.md)、[screenshot cropper](nextgqa_clock_screenshot_cropper.md)、[v6 export contract](nextgqa_claim_review_export_contract_v6.md)、[clock/frame evidence reader](nextgqa_clock_frame_evidence_reader_v1.md)。

2026-10-17 続報: v6 review exportの構造監査を通過したworksheet/lineage/review/indexの正確なbyte列へ、任意のEd25519 detached attestation v2を付加・検証するlocal-only CLIを追加した。root pin付き署名rosterからclaim-reviewer鍵を導出してkey-to-roleを確認するが、本人性、実際の視聴、媒体custody、claimの意味的妥当性は証明せず、呼び出し側が渡すreceipt hash自体も検証しない。score authorizationは付与しない。詳細は[attestation experiment log](../experiments/2026-10-17_completed_review_attestation.md)と[human-review runbook](nextgqa_v6_human_review_runbook.md)。

同日続報: review-onlyのintegrity/release rosterとwitness handoff署名を作るローカルCLI `scripts/nextgqa_review_ceremony_signing.py` を追加し、source inventoryに登録した。これは署名鍵の本人確認や実際の儀式を証明しない。freeze owner/custodianのkey IDと公開鍵aliasの曖昧な組合せをvalidatorで拒否する回帰テストを加えた。全テスト1,358/1,358、候補inventoryは63項目・欠落0だが、review/freeze/score authorizationは未成立。直接explanation faithfulnessの進捗は人手レビュー0/3、採点0/3、効果推定なし。詳細は[attestation experiment log](../experiments/2026-10-17_completed_review_attestation.md)。

2026-10-17 続報: レビューexportは同じpayload/indexの再ダウンロードと非同期生成中の二重出力防止に対応し、失敗時も画面上で再試行できる。HTMLの内容・構文検査、独立監査、全scripts suite 1141 testsが成功。clock plannerはhash-bound校正recordをopaque review token経由でworksheet/videoへ接続でき、identity形式の混在、token欠落、誤ったtokenからのmedia hash不一致をfail-closedで拒否する。これはデータ結合の修復であり、ブラウザ画面とdecoder frame/PTSの対応証拠や校正認証を追加しない。generated-claim faithfulnessは引き続き `未実施`（レビュー0/3、採点0/3、効果量なし）で、実レビュー、独立したframe correspondence audit、凍結済み入力、production atomic preflightが未完了。GPU採点は未実施。

2026-10-17 続報: all-nine atomic batch freezerのhash経路をtyped streaming SHA-256へ更新し、BF16を含むtensor値をPython list/JSONへ展開しないようにした。全tensor同時finite maskによるピーク増加もchunk scanで抑えた。192 MiB合成BF16入力のhashは0.186秒、測定上のhash中ピーク増分10.3 MiB。2 MB BF16 synthetic CPU→RTX→CPU hash一致も確認。これはbatch integrity工程の工学検証で、実動画・model forward・faithfulness測定ではない。直接faithfulnessはレビュー0/3・採点0/3、効果量なし。詳細は[atomic tensor hash log](../experiments/2026-10-17_atomic_tensor_hash.md)。

2026-10-17 続報: 固定Qwen2.5-VL checkpointで合成3行×3条件のGPU scoring pathを9/9 forward完了（30.16秒、peak allocated/reserved 10.74/12.47 GiB、最小free 1.92 GiB）。これは実データではなくcapacity/integration smokeで、faithfulness結果ではない。review-only release順序については、元の非承認integrity verifierを維持したまま、独立root署名のcustodian/witness rosterとdetached witness statementを検証する手順ゲートを追加した。結果は手続き上のレビュー開封許可に限り、実 chronology/人間identity/一回性は証明せず、score/model実行は不可。実root pin・実役割・witness ceremony・人手レビュー・15組のframe対応と時計校正は依然未完了。直接generated-claim faithfulnessはレビュー0/3・採点0/3、効果量/区間/検定なし。次は実レビューを開始できる信頼根と独立担当者の確定、その後にproduction all-three-row atomic preflightを完成させる。詳細は[release-order gate log](../experiments/2026-10-17_review_release_order_witness_gate.md)、[synthetic GPU smoke log](../experiments/2026-10-17_synthetic_nine_forward_gpu_capacity.md)、[human-review runbook](nextgqa_v6_human_review_runbook.md)。

2026-10-16更新: NExT-GQA `sentence_exploratory` の3件について、ローカルCSV・claim lineage・匿名レビューパッケージ・動画バイトの対応を3/3確認した。CPUでQA-margin runnerの未マスク動画prompt/processor経路を16/32フレームで監査し、A–Eラベルのdistinct single-token条件も確認した。人手レビューは保留中である。既存v5 rendererのレビューJSON exportは凍結v5方針の「tokenと判定/区間のみ」という記述を超える項目を含むため、レビュー開始前に全出力項目を開示する新規ページが必要である（[export開示記録](nextgqa_claim_review_export_contract_v5_renderer_v1.md)）。worksheetは`review_completed=false`、実行許可なし。直接のclaim-faithfulnessスコアは0/3、効果量・区間・検定なし。HTML5時計UI/schema v2変換器、観測→校正の合成テスト、上流hashを校正記録へ伝搬する未検証wrapperは準備したが、実ブラウザ観測・PTS校正・表示フレームとdecoderフレームの対応監査は未実施である。wrapperはplanner入力にも未接続であり、レビュー画面からcustody/観測までの完全なprovenance束縛も未完成。新たな先行曝露監査で候補3件全てが118件の過去使用候補除外listと重なることが分かったが、listの来歴は未解決のため、この3件をscale-up用新規標本として扱わない。補助16/32-frame QA-gold区間実験の各10件平均は再現できるが、保存artifactからレコード別の対応は独立に再構成できず、paired差は未検証の探索値である。したがって、現作業は実データの来歴・入力経路の準備診断であり、faithfulnessの実験結果ではない。n=3は有限記述pilotに限り、規模拡大や母集団推論の根拠にはしない。規模拡大の事前登録に必要な未決事項は[scale-up設計worksheet](nextgqa_scaleup_preregistration_skeleton.md)に整理した。詳細・次アクションは[2026-10-16 PDCAログ](../experiments/2026-10-16_pdca_6h_restart.md)。

2026-10-12追加: [8時間PDCAの記録](../experiments/2026-10-12_pdca_8h.md)で、実NOAA動画1件の7B BF16/4bitキャッシュ付き生成を3条件・6 forward検証し、native/no-op完全一致、有限出力、decode位置とcache長を確認した。別の公開CSV監査では62件すべてが点注釈で、固定5分区間には1件だけ含まれた。これは実行条件と注釈構造の確認であり、QAや説明faithfulnessの結果ではない。人間レビューはユーザーから未完了と確認され、注釈の最終QCも未検証のため、対応する評価ゲートは保留する。[キャッシュ継続の実行条件](qwen_temporal_cache_contract.md)を参照。

2026-10-12更新: 使用条件を確認したNOAA動画1件と固定合成入力で、GPUによる時刻位置の動作検証を実施した。QA成績やfaithfulnessの改善は測定していない。FPS metadataの欠落、時間座標の丸め、抽出点と区間spanの違いを分けて監査した。NOAAの3動画・6 QAは人間による確認が未完了のため`no_go`を維持する。[時刻・サンプリングの結果と限界](temporal_runtime_and_sampling_limits.md)、[1時間PDCA実行記録](../experiments/2026-10-12_pdca_1h.md)を現行の実行結果として参照する。閉じたhead探索は再開しない。

旧head-faithfulness確認は事前基準を満たさず、追加のhead/layer/alpha探索は停止した。研究を完了する段階ではなく、新しい評価対象へ進むための計画案として、**中間層の時間visual evidence介入によるgrounding改善と、claimの選択的証拠依存の検証**を定めた。[新研究計画書](internal_intervention_grounding_faithfulness_plan.md)が次の作業計画であり、まずCPUでNExT-GQAのinterval/claim適合性・重複ID・token-to-time mappingを監査する。データゲートを通るまで実データのQA・grounding・faithfulness評価を行わない。使用条件を別途確認した入力による動作検証は評価と区別して記録する。これは旧結果を覆すものでも、まだ承認済みの実験でもない。

2026-10-17に卒研の中心対象を「説明が指す動画証拠への予測の選択的依存」と明確化し、その依存が事前固定の内部介入後に増えるかを主指標候補とした。式・介入設定・標本設計・成功基準はまだ凍結されておらず、v2のQA-margin black-mask結果をこの評価の代用にはしない。

旧nearest-frame necessity、single-frame evidence-control、head scan/patching、head randomization、head ablation/steering、unsupervised head selection、task-validity QA scoring、旧grounding proxyの実データCLIは、この現行計画の実行経路ではない。閉じたhead仮説は再探索せず、旧single-frame/nearest-frame介入は現行claim-mask/time-mapping契約に合わないため、関連CLIとpublisherは停止する。旧validation pilot orchestratorも入力manifest作成前に停止する。合成診断や過去結果の再解析は引き続き別用途として扱う。動画を使うlabel-free runtime smokeはQA・grounding・faithfulnessを測らず、実行互換性確認として別扱いにする。汎用attention wiring smokeを使う場合も、ローカル処理が許可された素材だけを使用する。将来の実データ推論には、認証され、固定cohort/モデル/介入条件に結び付いた新しい実行ゲートが必要である。

下記の三段構成・その後に記載されたhead中心の計画は、旧設計と実施済み記録を残すための履歴であり、現行の次アクションとして実行しない。旧数値・負の確認結果は保持し、新計画とは別estimandとして扱う。

## 三段構成

1. **診断**: head ablation(attention knockout)で、動画への時間的groundingに因果的に効いているattention headを特定する。ベースの着想はLocalizationHeads(CVPR2025 Highlight, `github.com/seilk/LocalizationHeads`)を画像から動画/時系列に拡張したもの。
2. **補正**: 診断で特定したheadへのactivation steering。CRA-GQA(CVPR2025 Highlight, `github.com/WissingChen/CRA-GQA`)のfront-door介入(視覚側)/back-door介入(言語側)という因果推論の枠組みに位置づける構想。TA-AE(NeurIPS2025, `github.com/cai-jianfeng/TA-AE`)の対照ペアoffset vector注入方式との比較アブレーションも計画している(現状の実装は素朴な定数倍スケーリングのみで、TA-AE方式への切り替えは未着手)。
3. **検証（未成立）**: MECD/MECD+のEvent Granger Testに着想を得たnecessity probeを実装したが、NExT-GQA/Qwenに対する外部妥当性基準として確立したわけではない。根拠区間maskでスコアが下がること自体は予想される反応であり、それだけでは説明忠実性の成果にならない。`necessity_choice_margin_test.py`はstrict grounding frame maskと等数・時間位置対応control maskを5択gold-vs-best-foil marginで比較するが、controlは画面内容・識別情報量を対応させておらず、masking OODも残る。独立動画60件の結果は、事前primaryのbaseline-correct群n=37でselective margin dropがblack +0.334 (95% pointwise CI [+0.111,+0.581], sign-flip p=.0053)、blur +0.456 (CI [+0.172,+0.760], p=.0039)。一方、全60件のCIは両fillとも0を含み、correctness interactionも明確でない。従って現状はパイプラインの予備的感度分析であり、一般的faithfulness、モデル固有の因果説明、卒論の中心成果を支持していない。

土台データセットはNExT-GQA(CVPR2024, `github.com/doc-doc/NExT-GQA`)。QAごとに根拠時間区間ラベルが付いており、推論ベースで再学習不要という要件を満たす。

**2026-10-08時点の進捗:** 事前に固定したhead-faithfulness独立確認は79/79件成功したが、主効果は`−0.0756 nat`、95% t区間`[−0.1578,+0.0065]`であり、事前実質境界`+0.25 nat`を支持しなかった。したがって、このgradient×activation top-4・alpha=0.5・matched-control設定でのfaithfulness仮説を閉じる。追加のhead/layer/alpha探索はしない。これはこのモデル・評価・介入に限定した負の確認結果であり、一般的なfaithfulnessの否定ではない。独立再計算と卒論向けの定義・解釈は[確認結果記録](head_faithfulness_confirmation_results.md)、過去のpilot/sanity checkは[実験記録](../experiments/2026-10-08_head_faithfulness_implementation.md)、元の設計判断は[primary estimand review](../experiments/2026-10-08_primary_estimand_decision.md)を参照。

モデル説明を動画counterfactualで検証する案もデータ適合性ゲートを通らなかった。CLEVRERの同一scene候補はchoice program・演算列・質問文token距離の集計まで完了したが、意味上妥当なmatched pairは確立していない（詳細は[方向性レビュー](../experiments/2026-10-08_post_confirmation_direction_review.md)）。したがって現在、GPU実験へ進める確立済みfaithfulness仮説はない。これは卒研を完了・結果整理する段階という意味ではない。まず実質差境界0.25 natの根拠と関連する画像・動画VLM XAI研究を再点検し、妥当な新estimandとデータ妥当性ゲートを作る。[2026-10-09の閾値・文献レビュー](../experiments/2026-10-09_faithfulness_threshold_and_literature_review.md)に現時点の比較と次のCPU調査を記録した。新しい実験は、説明対象・介入・control・意味のある効果または推定目標・データpair成立性を事前に定義してから検討する。

公式CLEVRER train/validation質問JSONのCPU監査では、合計228,940問中27,975問がcounterfactual型だった。
修正版の[choice-level監査](../results/clevrer_program_choicelevel_corrected_audit_20261008.json)では、全27,975問で非空のchoice programを確認した。
同じsceneにcounterfactual質問がちょうど2問ある12,975 sceneのうち、choice-programの完全署名は12,974 sceneで異なり、1 sceneでは同一だった。
ただしpayloadは文字列token列であり、署名差だけでは動画上の介入が一要因だけ異なることも、counterfactualとして意味的に妥当であることも示さない。

先行する「operator trace全件 unavailable」という結果は、辞書ノード形式のみを探す抽出器の誤りだった。公式CLEVRER executorはchoice-programとquestion-level token列を連結し、登録済みmodule名を演算、それ以外を引数値として扱う（[公式run_mc.py](https://github.com/chuangg/CLEVRER/blob/master/executor/run_mc.py)、[executor.py](https://github.com/chuangg/CLEVRER/blob/master/executor/executor.py)）。
token列対応の[修正監査](../results/clevrer_program_operator_trace_corrected_20261008.json)では、異なるchoice-program署名を持つ12,974ペアのうち、演算列の一致は158（1.22%）、不一致は12,816（98.78%）だった。source hashは前回と同一で、train 104/8,641、validation 54/4,333が一致候補である。これは全件集計であり、統計検定は行わない。意味的妥当性を確認するまではGPU実験を行わない。

続く[質問文距離監査](../results/clevrer_pair_question_token_distance_20261008.json)では、演算列一致158ペア中、token edit距離2–3は12（7.6%）、4以上は146（92.4%）で、距離0–1や計算不能はなかった。これは表層文字列の記述統計にすぎず、近い12ペアも介入が一要因だけ違うとは確認されていない。したがってCLEVRERの同一sceneペアを使うfaithfulness実験のデータ適合性は立証されていないと判断し、この構造pair探索を停止する。GPU実験や距離閾値の追加探索は行わない。再開には、事前定義した意味的ペア基準と、個別例を結果を見ずに判定する手順が必要。

## 重要: 3つの土台論文はコード流用ではなく着想の借用

2026-09-25の調査で判明した実装コストの前提:

- **LocalizationHeads**: 画像LVLM(LLaVA-1.5-7B)×RefCOCO(物体位置特定)が対象。動画への拡張(空間→時間)は一から設計が必要。
- **CRA-GQA**: 公開コードのバックボーンはTempCLIP(CLIP特徴ベースの非LLM groundingモデル)。front-door/back-door因果定式化はCLIP特徴に対して定義されており、Video-LLMのattention head活性化への定式化ではない。
- **MECD**: Event Granger TestはMECD独自データセット(ActivityNetベース)上で専用モデルを学習・評価するもので、NExT-GQAには非対応。

**つまりこのプロジェクトは「コードを繋ぎ合わせる」のではなく、「elbow法によるhead選択」「front-door/back-door因果の考え方」「Granger式necessityテストの考え方」という着想だけを借りて、Qwen2.5-VL×NExT-GQAという新しい組み合わせに合わせて再設計・再実装するプロジェクトである**。この前提で実装コストやスケジュールを見積もること。

各論文の詳細(手法・数式・実験結果・この卒研との関係)は、Claude Docsアーティファクトとして要約済み:

- [LocalizationHeads](https://claude.ai/artifact/YM7pTtwhH3kZBQuAvh4Qd1) — head選定基準(value-based elbow + spatial entropy、教師なし)の詳細
- [CRA-GQA](https://claude.ai/artifact/GJhqtN48FyQ9jVzgo7gz7Q) — front-door/back-door介入の数式
- [MECD](https://claude.ai/artifact/J3Bz62ToeWMhTP8gzTyN1h) — Event Granger Testの定式化
- [TA-AE](https://claude.ai/artifact/Nw6ScFxEJjzfF8zrRPw2eQ) — Qwen2.5-VL向け公式実装あり、対照ペアのoffset vector注入というsteering手法
- [Faithfulness Metrics Don't Measure Faithfulness](https://claude.ai/artifact/UzpUVU53iCWmXzSuUExSng) — 忠実性指標そのものへの批判。この卒研の問題意識の裏付け

(これらはClaude Docsのアーティファクトであり、Claude以外のエージェントがこのリンクを開けるとは限らない。中身が必要な場合はユーザーに内容を共有してもらうか、同等の要約を別途作成すること。)

## モデル・技術選定

- **本命モデル**: Qwen2.5-VL-7B-Instruct。動画ネイティブ対応で、画像grounding(bbox出力)に明示的に訓練されているため、良いlocalization headが自然に育っている見込みが高いと判断。
- **比較対象として検討したが未着手**: VideoLLaMA2-7B(MECD等の先行研究で使用実績あり)。
- **診断と補正でattention実装を分離する設計**: 診断(head ablationでどのheadが効くか特定)はattention weight自体を見る必要があるため`attn_implementation="eager"`が必須。補正(該当headの出力をforward hookで改変)はattention確率を覗く必要がないため、本来は`flash_attention_2`で高速に回せる設計を想定している(Qwen2.5-VLのGitHub Issue #1271で「eager/flash混在で生成が破綻した」という報告があるため、役割を完全に分離する)。**ただし2026-10-01時点の実装はまだすべてeagerで統一されており、flash_attention_2への切り替えは未着手**。
- **Qwen2.5-VLは必ずbfloat16を使う**。`torch_dtype`と`bnb_4bit_compute_dtype`の両方。float16は7Bモデルで`torch.multinomial`がNaN/InfによりCUDA device-side assertを起こす(詳細は[experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md](../experiments/2026-09-29_01_environment_and_diagnosis_pipeline.md))。

## スケジュール目安(2026-08-24起点、全6ヶ月)

1-2ヶ月目=環境構築・スパイクテスト・診断パイプライン、2-4ヶ月目=steering補正実装、4-5ヶ月目=本評価・アブレーション、5-6ヶ月目=執筆。2026-10-01時点は1-2ヶ月目の終盤〜2ヶ月目(診断パイプラインの検証・補正の初期試作)に相当する。

## スコープ: このリポジトリの対象範囲

このリポジトリ(`res`)は**動画VLM忠実性研究(卒研本体)のみ**を対象とする。ユーザーには別トラックとして「DIP×SVARM-IQ(特徴量重要度の高次分解)」プロジェクトが`~/test/dip_svarmiq_poc`に存在するが、これは院試・研究計画書など別目的のものであり、このリポジトリとは完全に別物。混同しないこと。

## 検討して保留した代替方向性

2026-09-02に「修士まで見据えてもっと新規性を狙いたい」という要望で、以下を含む代替方向性(A〜I、6つの構造軸)を検討したが、2026-09-25に本人が確認の上「確定方針(上記三段構成)のまま進める」と決定し、以下は不採用ではなく未選定のまま保留されている。

- A. 推論トレースの忠実性(video CoT faithfulness)
- B. 必要性・十分性の反実仮想検証(NExT-GQAの「Pointing≠Causing」の定量化)
- C. 言語prior vs 視覚根拠の因果的分離(NDE/NIE)
- D. 多段階因果チェーンの忠実性
- E. 時間順序の反実仮想(反転・シャッフル、低コスト)
- G. SAE(Sparse Autoencoder)による因果概念発見・steering — 現行案との連続性が最も高い
- I. パラフレーズ一貫性を忠実性の必要条件として使う
- 構造軸: 説明手法の種類/摂動・介入の種類/necessity・sufficiency・comprehensivenessの動画版一般化/真の因果構造が既知の合成データ検証(CLEVRER型)/モデルスケール・アーキテクチャ/時間粒度

ドメイン特化の方向性(事故ドメイン監査、事故ドメインデータセット構築)は評価が低く、ドメイン差し替え自体を新規性の軸として提案しないという方針も確定している。

**将来「もっと新規性が欲しい」という話が出たら、まずここに立ち返ってよい。ただしユーザーから明示的に「見直したい」という指示がない限り、これらを主軸として提案しない。**

## 現在の到達点（2026-10-15、詳細は experiments/ 参照）

2026-10-15時点では、事前基準未達のhead-faithfulness確認枝は停止しており、head/layer/alpha探索を再開しない。主な次候補のclaim-linked evidence-frame maskは、対象母集団・分布仮定・SESOIの決定、動画ごとの権利確認、人手claim-to-frameレビュー、runner統合が未完了のため、confirmatory scoringとGPU推論は`no_go`である。schema v13 validatorと合成テストはプロトコル整備であり、実データfaithfulness測定ではない。この研究作業窓での新規faithfulness標本・効果推定は0件。現在の継続作業は[2026-10-15 5時間PDCA follow-upログ](../experiments/2026-10-15_pdca_5h_followup.md)、その前の作業記録は[2026-10-15 PDCAログ](../experiments/2026-10-15_pdca_5h.md)を参照。

claim-maskの事前primaryは、各条件で別々に選ぶbest foilとのgold-vs-foil log-marginのcontrol-minus-claim-mask差である。foilは条件間で切り替わり得るため、将来の正の差はgold log-probabilityの低下、best-foil log-probabilityの上昇、または両方を含み得る。これはgold回答確率だけの低下を意味しない。解釈の定義と合成算術例は[protocol contract](claim_mask_protocol_contract.md)と[5時間PDCAログ](../experiments/2026-10-15_pdca_5h_followup.md)に記録する。

同日の合成GPU engineering checkの負荷比較では、4-frameと16-frameを各1ケース・3 forward、RTX 5070 Tiで実施した。video token数は512/2,048、最大peak allocatedは6.04/7.13 GiBだった。別の16-frame監視付き反復では、全実行窓（モデルロード・前処理・forward・公開処理を含む）の200 ms間隔測定がutilization平均17.34%、p95 90%、最大100%だった。これはforward単独の利用率ではなく、詳細な集計と制約はPDCAログと[monitor summary](../results/qwen_temporal_position_gpu_monitor_summary_20261015_01.json)に記録した。3時間PDCAでは合成条件を32-frame、続いて40-frameへ拡張し、各1ケース・3 forwardが完了した（4,096/5,120 video tokens、最大peak allocated 10.64/13.23 GiB）。4/16/32/40-frameの単回比率は[再現用分析器](../scripts/analyze_qwen_temporal_position_scaling.py)と[集計artifact](../results/qwen_temporal_position_scaling_summary_20261015_04.json)から再生成できる。40-frameでは32-frame比の最大allocated memoryが1.243倍、条件別forward時間が1.897/2.286/2.206倍だったが、各条件は一度だけで入力内容・warmup・実行順が統制されておらず、一般的なscaling lawではない。別プロセス3回で同一40-frame固定fixtureを反復した結果、native/no-op/legacyのforward時間中央値は7.383/5.962/5.832秒、範囲は1.189/0.667/0.209秒だった。peak allocatedは各条件で反復間同値（13.2259/13.2250/13.2250 GiB）。このn=3は同一synthetic fixtureの工程再現性確認であり、条件差の検定や一般化可能な分散推定ではない。[反復集計器](../scripts/analyze_qwen_temporal_position_repeatability.py)と[集計artifact](../results/qwen_temporal_position_repeatability_20261015_04.json)で再現できる。条件別測定と限界は[3時間PDCAログ](../experiments/2026-10-15_pdca_3h.md)に記録した。これらは時刻位置実装の動作・資源診断であり、faithfulnessの標本・効果推定には数えない。個別summaryは[4-frame](../results/qwen_temporal_position_synthetic_4frame_20261015_02.json)、[16-frame](../results/qwen_temporal_position_synthetic_16frame_20261015_02.json)、[32-frame](../results/qwen_temporal_position_synthetic_32frame_20261015_01.json)、[40-frame](../results/qwen_temporal_position_synthetic_40frame_20261015_01.json)、[監視付き16-frame](../results/qwen_temporal_position_synthetic_16frame_monitor_20261015_01.json)。

動画間のsource cluster依存が未監査であるため、独立動画を仮定する予定のStudent-t区間を、N=80・36条件・各50,000反復の合成Gaussian random-intercept感度分析で確認した。たとえばICC=.20では、balanced K=10の名目95%区間coverageは79.25%（Monte Carlo Wilson区間78.89–79.60%）、固定不均衡K=10 layoutでは77.05%（76.68–77.41%）だった。[結果artifact](../results/claim_mask_cluster_coverage_sensitivity_20261015_01.json)と[生成コード](../scripts/simulate_claim_mask_cluster_coverage.py)は再現可能な集計値とsimulation errorを含む。これは仮定した相関モデル下での方法感度にすぎず、実データにcluster依存があること、実際のICC、妥当な推論法、SESOI、faithfulness効果を示さない。実データの対象・cluster lineageのoutcome-blind監査が先行する。

追加のoutcome-blind tail感度分析では、動画間ICC、cluster random-effect分布、動画ごとのresidual分布を分離して288合成条件を比較した（N=80、各20,000反復）。Gaussian random-interceptの主要な例は先行50,000反復gridとMonte Carlo誤差内で再現され、Student-t(3/5)成分では仮定により区間coverageが変わった。[来歴hashを含む最新版artifact](../results/claim_mask_tail_coverage_sensitivity_20261015_02.json)と[生成コード](../scripts/simulate_claim_mask_tail_coverage.py)を参照。先行`_01`は数値結果は同じですが、直接importする依存ファイルhashが欠けるため来歴の完全なartifactとしては使いません。分布の妥当性や実データの依存構造、別の信頼区間の優位性を示すものではなく、推論法の選定は引き続き保留する。詳細は[2026-10-15 PDCAログ](../experiments/2026-10-15_pdca_3h.md)。

公開一次資料から、NExT-QA動画はVidOR由来、VidOR動画はYFCC100M由来で、YFCC元メディアのCC条件は投稿者ごとに異なり各item metadataに記録されることを確認した。これは個別動画の利用許可を確定しない。対象動画からYFCC recordへの対応、条件、取得経路に適用されるagreementは未確認であり、権利ゲートを維持する。

### 2026-10-06時点の履歴

- 診断パイプライン(head ablation、attention抽出)は複数マシンで動作確認済み
- 主要候補: `layer=6, head=16`。ただし根拠の信頼度は指標によって異なる(attentionベースのhit_rateは位置バイアスの疑いがあり、steeringのcorrect_logprobベースの統計検定の方が頑健)。詳細は[experiments/2026-09-30_fixed_chunk_baseline_confound.md](../experiments/2026-09-30_fixed_chunk_baseline_confound.md)
- head選定の教師なし化(LocalizationHeads本来の基準)を試みたが、現状の実装ではlayer=6/head=16を再発見できていない
- 補正(steering)は初期試作段階、TA-AE方式への切り替えは未着手
- 2026-10-07の再評価: 正解根拠区間を隠したときスコアが落ちるという観測そのものは自明で、卒論の主要成果とはみなさない。非自明な問いはmatched controlより根拠区間maskの影響が特異的に大きいかだが、現在のcontrolは時間位置のみ対応し、映像内容や回答識別情報量を統制していない。baseline-correct subgroupのpointwise結果だけを根拠に説明忠実性を主張しない。保存済み結果と限定事項は[内部検討メモ](thesis_results_draft.md)および[実験ログ](../experiments/2026-10-01_zzz_patch_scope_comparison.md)に記録する。次のGPU実験は保留し、説明の選択根拠と意味的に妥当なcounterfactual/controlを比較する別プロトコルを先に設計する。

- 2026-10-07の研究質問再設計案: 追加mask/head探索を行わず、人手因果イベント注釈を基準とする「因果edge判断への内部介入の選択性」を候補質問として定義した。CLEVRER-Humansを制御環境の第一候補とするが、元動画は合成であり自然動画一般化には使えない。イベントを時間区間に結び付けられるか、配布物・利用条件が適切かをCPU監査するまで実装/GPUへ進まない。詳細と停止条件は[再設計プロトコル](research_redesign_protocol.md)と[日付ログ](../experiments/2026-10-07_research_question_redesign.md)を参照。

- 2026-10-08のv6/v7監査: 同一targetにpositive/negative incoming-edge候補を持つsceneはtrain 529/574、valid 464/525。各1本ずつに限るとvalidは28/525 scenes、複数候補から選べば最大464 scenes。これは注釈付き候補の可用性であり、faithfulnessの測定ではない。**卒論の最重要目的は動画VLMのexplanation faithfulnessであり、因果edge QAへ研究目的を置き換えない。** CLEVRERは、モデルが選んだ説明・内部機構への介入が予測へ与える影響を検証できる設計を作れる場合に限り、faithfulness研究の評価環境候補として使う。時間区間注釈がないため、時間局在faithfulnessの根拠にはならない。詳細は[再設計プロトコル](research_redesign_protocol.md)と[日付ログ](../experiments/2026-10-07_research_question_redesign.md)。

## 方針更新: soft ablationを中心にした忠実性評価(2026-10-01)

具体的な研究質問・数値定義・標本設計・判定基準は[動画VLM explanation faithfulness研究計画書](faithfulness_research_plan.md)に記載する。validation 3件smokeは3/3完了（平均+0.108 nat、95% t区間[−0.266,+0.483]）。続く30件pilotは30/30完了、平均selective margin差−0.0225 nat（SD 0.3774、95% t区間[−0.1634,+0.1184]）で、事前成功境界+0.25 natを満たさず、区間上限も境界未満だった。出力層・累積全層randomizationもsanity checkとして完了したが、いずれもfaithfulness効果ではない。結果に合わせたK/alpha/control変更はせず、追加GPU探索を止めてCPU設計レビューへ移る。時間attentionのframe対応は時間妥当性分析前に別途検証する。再計算用コードと解釈は[2026-10-08実験ログ](../experiments/2026-10-08_head_faithfulness_implementation.md)。

これまでの「grounding headを発見してsteeringで補正する」という表現は、
attention hit rateの時間的位置バイアスと、完全なhead zero化による内部表現の
崩壊を十分に区別できないため、研究の主軸を次のように更新する。

> attentionの相関だけでなく、attention head出力を連続的に減衰させる内部介入で、
> 動画VLMの忠実性を検証する。

対象headの出力は `h' = alpha * h` とし、`alpha=1`(介入なし)から
`alpha=0`(完全ablation)までのdose-response curveを測定する。主分析は
`alpha=0`だけではなく、`0.75, 0.5, 0.25, 0.1, 0.05, 0.01`等のsoft
ablationで行い、完全zero化は端点・補助分析として扱う。

主な評価値は正解トークンのlogprob変化であり、control token、全体entropy、
モデルのtop token、対象head以外のcontrol layer/headも併記する。これにより、
選択的な予測変化と、単なる出力分布の崩壊を区別する。`layer=6/head=16`は
確定したgrounding headではなく、過去のscreeningで得られた検証対象候補とする。

今後は (1) attention hit rateとfixed-chunk baselineの比較、(2) soft ablationの
独立データ検証、(3) 根拠区間maskによるnecessityテストとの対応、の順に進める。
精度改善や新しいheadの探索は、これらの最小構成が成立した後に行う。根拠区間
maskにはOOD入力混入の限界があるため、結果は使用したモデル・データ・介入・
評価指標の範囲に限定して報告する。

2026-10-05のfresh one-frame evidence-vs-placebo choice-margin confirmationは
baseline-correct primary cohortで非有意かつ先行探索効果を再現しなかった
(`experiments/2026-10-01_zzz_patch_scope_comparison.md`)。同じ一枚置換仮説の再探索は止め、
元々のverification段階であるgrounding evidence necessityへ戻る。ただし新しいblack-mask
choice-margin studyも別介入・別estimandであり、smokeと独立sample確認前は探索的とする。

2026-10-05のfresh necessity-choice runと独立video replicationはbaseline-correct primaryで
似た正方向効果を示し、同じreplication videosでのblur-fill感度分析でも同方向だった。
追加mask-fill探索は停止し、これらを限定的な介入感度として整理する。black/blur双方のOODと
prompt依存marginの限界は残る。

直近の未解決タスクと優先順位は、最新の`experiments/*.md`ファイルの「次のアクション」セクションを参照すること。

## 2026-10-12時点の数値再現性・実行 provenance

Qwen2.5-VLの系列長診断は、固定した1クリップ・4フレーム・5 forward条件で、通常のCycle13実行、ロード直前shard hash認証実行、ロード前後hash一致実行の3回を比較した。19 logit比較、9 hidden比較、10 output projectionの固定スカラー値は3回で一致した。最後の実行では5 shard（合計16,584,414,560 bytes）をモデルロード直前・直後にhashし、全ファイルのサイズとSHA256が一致した。この診断は数値経路の再現性を示すが、因果マスクの正しさ、QA精度、説明faithfulnessを示すものではない。

head/layer/alpha探索は事前閾値未達のため再開しない。NOAA QA枝は3/3グループ・6/6結合・ラベル漏れ0だが、人間レビュー署名がないため`no_go`を維持する。BNB行数診断も孤立線形演算の実装差を示すだけで、Qwen hidden driftの原因帰属には使わない。

卒論用のblack/blur対応比較図・CSVは
[`results/necessity_choice_margin_thesis_summary_final_forest.svg`](../results/necessity_choice_margin_thesis_summary_final_forest.svg)
および同prefixの`_estimates.csv`、`_metadata.json`。`scripts/plot_necessity_choice_results.py`
で保存済みJSONからCPU再生成できる。本文・図注の転記用ドラフトは
[`docs/thesis_results_draft.md`](thesis_results_draft.md)。統計と適用範囲は最新実験ログを参照。
