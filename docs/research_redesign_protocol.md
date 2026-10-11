# 研究質問の再設計案: 因果イベント判断に対する内部介入

更新日: 2026-10-07

> **2026-10-08追記:** この文書中の「未使用test動画で一度だけ固定protocolのconfirmationを行う」という作業は完了した。79/79件の独立確認で主効果は`−0.0756 nat` (95% t CI `[−0.1578,+0.0065]`)、実質境界`+0.25 nat`を満たさなかった。この結果を受け、同じhead-faithfulness仮説の追加GPU探索は停止する。詳しい定義と解釈は[確認結果記録](head_faithfulness_confirmation_results.md)。以下はデータ再設計時の履歴・ゲートであり、現時点でCLEVRER実験の実施承認を意味しない。

## 結論

現時点では「根拠フレームを隠すと回答スコアが下がるか」を追加で測らない。
その効果は視覚情報を消せば起こり得る自明な入力感度であり、現在の時間位置対応controlも
画面内容・回答識別情報を統制できていない。既存mask結果は予備的感度分析として閉じる。

次に問う候補は、より限定して次のとおりとする。

> 人手で注釈された因果イベント関係について、動画VLMの内部表現・head群への介入は、
> 同じ動画内の非因果イベント関係に対する介入より、因果判断の正誤マージンを選択的に変えるか。

これは「モデルの説明が忠実」と直ちに結論する問いではない。まず、因果ラベルに対する
内部介入の選択性をテストする診断研究である。特定headを最初から決めず、単head・複数head・
複数層の相互作用を候補として扱う。現在までのlayer/headスクリーニング結果から候補を
確定したとはみなさない。

## なぜこの変更か

直近のNExT-GQA結果では、black/blurを通じたevidence-minus-control margin差は、全60件で
black +0.296 (95% pointwise CI [-0.046,+0.615], p=.082)、blur +0.360
(CI [-0.035,+0.721], p=.059)で、いずれも区間が0を含む。baseline-correct n=37では
正方向だが、これは入力摂動の感度であり、モデル自身が提示した説明の忠実性を直接検証しない。
さらにbaseline-correctで条件づけた部分群結果は別集合で再現していない。したがって、同じmask、
同じmargin、同じsubgroupを追加探索しても、中心的な不確実性は解消しない。

## データ候補と採否

### 第一候補: CLEVRER-Humans

CLEVRER-HumansはCLEVRER動画に対し、イベント記述と人手による因果イベントグラフ(CEG)を付した
データセットである。論文は1,108動画、8,581イベント記述、21,167関係注釈、1,076 QAを報告し、
因果関係は人手判断でラベル化されている。一方、映像は合成物理シーンであり、イベントタイプや
視覚分布が自然動画と異なる。従って採用する場合も「人手因果注釈のある制御環境での機序検証」と
位置付け、NExT-GQAや自然動画一般へ外挿しない。

Ubuntuでの初回ファイル探索では、CLEVRER-Humansの質問/CEGファイルは見つからなかった。
CLEVRER公式READMEによれば元データは20,000本、validation splitだけでも5,000本で、各動画は5秒・
128フレーム・480×320。CLEVRER動画データ自体はCC0と記載されている。ユーザーが確認したvalidation
archiveのContent-Lengthは6,206,059,321 bytes、home filesystemは897G空き。ただしCLEVRER-Humansの
1,108動画サブセットとvalidation全体は同じとは限らず、5,000本すべてを取得する必要は未確認。

先にDryadの約3.05MBの注釈package（質問JSON/CEG pickle等）だけを取得する。標準ライブラリの
`scripts/download_clevrer_humans_annotations.py`は必須ファイル、ZIP内パスの安全性、50 MiB上限を検査し、
既存出力を上書きせず展開する。pickleはスクリプト内でロードしない。注釈側から動画ファイル名を確認して
必要subsetを絞るまで、6.2GB archiveは取得しない。

UbuntuからDryadの一括APIを呼ぶとHTTP 401となる場合がある。これはアーカイブ検査の失敗ではなく、配布APIの
認証拒否である。APIトークンをリポジトリやコマンドへ埋め込まず、公式[Dryad記録](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp7c)
の「Download full dataset」からブラウザで取得したZIPを`--archive-file`で検査・展開する。取得したZIPをGitへ
追加しない。

### 比較候補: ACQUIRED

ACQUIRED論文は、物理・社会・時間的反実仮想を扱う約3.9Kの実世界動画QAを報告している。
自然動画との適合は魅力だが、現在確認できた公開論文情報だけでは、今回必要なイベント単位の
時間区間・因果グラフ注釈、動画の取得可能性、実験利用条件を確認できていない。これらを満たすと
確認できるまでは主実験データに選ばない。

### 実験コードを書く前のデータ採否ゲート

1. 公式配布元から動画・質問・CEGラベルを取得でき、再配布せずローカル実験で使える条件を確認する。
2. CEGのevent nodeが動画内の時間区間またはフレームへ再現可能に対応付けられるかを確認する。
   対応付け情報がなければ、時間局在介入を主張する設計には使わず、別の評価単位に再設計する。
3. 正・負の因果エッジが同一動画内で対になり、質問文・イベント記述の長さや選択肢形式を
   対応させられるサンプル数を数える。
4. Qwen2.5-VL-7Bの無介入で、対象タスクに最低限の回答可能性があるかCPUでなくGPU smokeを
   1–3動画だけ使って確認する。正解率がchance付近、または入力形式が破綻する場合は本実験へ進まない。
5. 上記を通過したときだけ、別動画IDで探索/確認splitを固定し、事前登録相当の解析計画を保存する。

このゲートを通過しない場合、CLEVRER-Humansへの移行を既成事実化せず、NExT-GQAの既存結果を
限界付き予備評価として閉じる。必要なデータ取得や長時間GPU実行は、ゲート確認後にコストを示して行う。

## 暫定評価プロトコル(ゲート通過後のみ)

### 評価単位・estimand

- 基本単位はvideo × target event × candidate cause-eventの因果関係判断。
- 同一動画内の人手positive edgeと、イベントの出現機会・時間間隔・記述形式を対応させたnegative
  edgeをペアにする。ラベルを見てから都合のよい例だけを除外しない。
- 主スコアは正解ラベル対foilのlog-probability margin。回答正誤も別に報告する。
- 主estimandは、内部介入によるpositive-edge margin変化とmatched-negative-edge margin変化の
  sample-level差の平均。符号、CI、検定、除外、失敗を事前に固定する。

### 内部介入の選び方

- どのlayer/head/groupを介入対象にするかは探索splitだけで選ぶ。confirmation splitを候補選びに使わない。
- 探索では全層のsingletonと、少数headを組み合わせたgroupを別々に評価する。協調性は
  pair/group介入がsingleton効果の和を超えるかという事前定義した交互作用で見る。
- 非加算性だけからhead間の機序・方向性を断定しない。具体的な媒介経路を主張するには、追加の
  層別patch/相互介入と独立確認が必要。
- 全head全組み合わせの総当たりはしない。探索family、候補数、選択基準、多重性処理、計算量を
  smoke前に明記する。候補を結果を見ながら増やす場合は探索として区別する。

### 対照・妥当性

- 同じ動画・同じ質問・同じgold/foilでの介入前後を対応させる。
- matched-negative edgeに加え、無関係なinternal groupまたは層への介入を陰性対照にする。
- 入力動画の黒塗り/blurを主介入にしない。動画側の操作を補助的に行う場合も、物理的に自然な
  counterfactual動画と単純maskを区別し、単純maskの効果を因果的説明の証拠にしない。
- self-checkは実装整合性のみを確認し、因果妥当性の証明とは扱わない。

### 統計・報告

- split単位は動画ID。探索・確認の動画重複を禁止する。
- primary contrastを1つ固定し、両側95% CIと対応検定を報告する。探索familyにはHolm等の補正を適用し、
  unadjustedの個別結果は探索値と明記する。
- requested/effective n、失敗・除外理由、モデル/quant/dtype、GPU、seed、frame sampling、prompt、
  tokenizer/library version、介入hook位置をrun manifestに保存する。
- baseline-correct例だけをprimaryにする場合は、独立splitで介入前正誤を条件定義し、全例結果と
  correctness interactionも併記する。観測後に部分群を作り直さない。

## 研究上の成功条件と停止条件

成功は「特定層で有意差が出ること」ではない。最低限、次を満たす必要がある。

1. causal-positive対matched-negativeの事前primary contrastが独立動画splitで同方向に再現する。
2. 選択した内部groupが、探索splitでの選択後もconfirmation splitで効果を保つ。
3. 効果が無介入精度の破壊、全質問共通のlogit崩壊、または介入量だけの単調な劣化で説明されない。
4. 複数headの協調を主張する場合、singleton和との差を独立splitで評価し、探索familyを補正する。

データ注釈とモデル入力が対応しない、無介入性能がchance水準、primary contrastがconfirmationで再現しない、
または主要効果が一般的な分布崩壊と区別できない場合、この内部因果説明仮説を停止する。head候補を変えて
有意差を追わない。

## 参照文献・データ情報

- CLEVRER: Yi et al., *CLEVRER: Collision Events for Video Representation and Reasoning*, ICLR 2020.
  [論文](https://jiajunwu.com/papers/clevrer_iclr.pdf),
  [公式README/データ仕様](https://data.csail.mit.edu/clevrer/README.txt)
- CLEVRER-Humans: Mao et al., *Describing Physical and Causal Events the Human Way*, NeurIPS 2022 Datasets and Benchmarks.
  [論文](https://papers.nips.cc/paper_files/paper/2022/file/32fea358f4811ec6d703e8c17028ce1d-Paper-Datasets_and_Benchmarks.pdf),
  [公式プロジェクト・配布情報](https://sites.google.com/stanford.edu/clevrer-humans),
  [Dryadデータ記録](https://doi.org/10.5061/dryad.5tb2rbp7c)
  (最新版package 3.05 MB),
  [Dryad download API仕様](https://github.com/datadryad/dryad-app/blob/main/documentation/apis/README.md),
  [公式dataloader](https://github.com/xyang23/CLEVRER-Humans1.0/blob/main/description-gen/dataloader.py),
  [公式event生成コード](https://github.com/xyang23/CLEVRER-Humans1.0/blob/main/description-gen/dataset_pipeline.py)
- ACQUIRED: Wu et al., *A Dataset for Answering Counterfactual Questions In Real-Life Videos*, EMNLP 2023.
  [論文](https://arxiv.org/abs/2311.01620)

## 現段階の次の作業

GPU実験や追加head探索はしない。Dryad注釈ZIP取得とv5監査は完了。README記載のCEG 1,099 videosと実測scene
1,099は一致する。train/valid question sceneは全件graphに一致したが、causal-cloze sceneは100件中31件のみ
graphと一致し、69件の不一致の意味を配布仕様と照合する必要がある。CEG node/edge合計はREADMEとの差179/290が
残る。v4ではgraph node属性が空、edge属性は整数`weight`のみで、named time keyは0件だった。v5ではtrain/validの
各CEG graph fieldに2,769/2,590個のstring node labelがあり、frame番号・時計表記・秒数パターンは0件だった。
これは明示的locatorの未検出であり、自然言語の時間意味がないことの証明ではない。ただし公式annotation説明にframe
interval annotationの記載がなく、時間区間ground truthとして確認できないため、時間局在faithfulnessは評価対象にしない。

v6 CPU-only監査では同一targetに`CEG_pos`/`CEG_neg`双方のincoming edge候補があるsceneをtrain 529/574、valid
464/525確認した。合計8,940 edge-pair combinationsは同一scene/target内で重なるため、独立標本数ではない。負辺は
20,018本、正辺は2,724本で頻度差も大きい。公式loaderは正辺のpredecessorを正答候補、負辺のpredecessorを誤答候補として
同種の「responsible for target」QAに使う。従って候補データがあることは確認できたが、モデル性能・内部介入の選択性・
faithfulnessの証拠ではない。

v7では正負incoming edgeが各1本のtargetはtrain 62 targets/57 scenes、valid 33 targets/28 scenesだった。strictな
一候補ずつのvalid群は28 videos (5.3%)と小さい。複数候補を許して各targetから正辺・負辺を1本ずつ選べば最大464 valid
scenesを利用できるが、これは候補選択規則を新たに固定する必要がある。**この候補数はfaithfulnessの証拠ではなく、因果edge QAへの目的変更を正当化しない。** 卒論の目的は引き続き動画VLM explanation faithfulnessである。CLEVRERを使う場合も、モデル自身の説明または説明手法が選択した内部機構・入力根拠を明示し、その対象への介入効果を意味的・情報量的に妥当な対照介入と比較する必要がある。CEGは説明の外的妥当性ラベル候補にはなり得るが、時間intervalがないため時間局在評価には使えない。

faithfulness estimandのCPU設計レビューを完了した。既承認のhead-group estimand（top-4 gradient×activation、最終prompt位置で`alpha=0.5`、層数/活性値対応control 5組、動画単位のtarget-minus-control margin drop、成功基準CI下限`>0.25` nat、独立test split）を変更せず維持する。validation pilotは実質境界を支持しなかったが独立confirmationではないため、追加探索ではなく、未使用test動画で一度だけ固定protocolのconfirmationを行う。手順と停止条件は[primary estimand decision log](../experiments/2026-10-08_primary_estimand_decision.md)。GPU実行前に新規3動画smokeとfresh manifestの重複除外を行い、smoke動画はconfirmatory cohortに入れない。

段階実行・重複除外・公開を制御するrunner `scripts/run_head_faithfulness_confirmation.py` とCPU testsを追加した。runnerは過去test-smoke、全repo sample manifests、今回smokeを除外し、3動画smoke成功時にのみ79動画confirmationへ進む。最低有効数71未満なら失敗として扱い、結果を見た後の補充はしない。

**2026-10-08更新:** 既に計画されていた累積全層ランダム化sanity checkは、RTX 5070 Ti上でsmoke 1/1、全層3/3の完了がGitHub上の結果・manifestで確認された。3例の平均rank Spearmanは最終深度で−0.286、top-4重複は0/3、回答選択は3/3で変化した。ただしこれは学習済みネットワークを初期化重みに置き換えるsanity checkであり、faithfulnessの介入効果ではない。この結果を理由とした追加GPU探索は行わず、上記CPU設計レビューへ進む。詳しい数値・定義・限界は[実験記録](../experiments/2026-10-08_head_faithfulness_implementation.md)を参照。
