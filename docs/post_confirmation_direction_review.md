# 次の研究方向のCPU設計レビュー

更新日: 2026-10-08

## 1. 今回閉じる仮説

79動画の独立確認では、gradient×activation上位4 headを`alpha=0.5`にしたmargin低下がmatched controlより大きいという主効果は得られなかった。`theta=−0.0756 nat`、95% t CI `[−0.1578,+0.0065]`で、成功基準CI下限`>+0.25 nat`を満たさない。よって、同じhead選択法・同じデータでhead/layer/alphaを変える探索は停止する。

既存のgrounding区間mask分析も、maskによりスコアが落ちることだけは忠実性の成果でない。black/blur、時間対応control、baseline-correct部分群を含む既存分析は、自然動画から外れたmask入力とcontrol情報量の不一致が残る予備的感度評価として扱う。追加fill探索はしない。

## 2. 先行研究の更新と意味

- Balasubramanian et al.のFindings of EMNLP 2025は、大規模vision-language modelのchain-of-thoughtについて、テキスト由来・画像由来のbiasが回答と説明内のbias言及へどう影響するかを調べている。従って「VLMの説明がbiasを正直に報告するか」全般を新規性として掲げるのは避ける。[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.723/)
- Ding et al.のEDCTは、画像VLM自身の説明から視覚概念を取り出し、生成画像編集でその概念を変えて回答と説明の一貫性を試す枠組みを提案している。現在確認できたものは2025年arXiv preprintで、査読済み一般原理としてではなく隣接提案として扱う。画像編集とLLM judgeが必要なため、そのまま動画へ移すと介入品質・評価器依存が大きい。[arXiv:2510.00047](https://arxiv.org/abs/2510.00047)
- Süssの2026年Diplomarbeitは、VLM自然言語説明の視覚反実仮想テストを画像データSNLI-VEで扱っている。画像VLM説明のcounterfactual faithfulness自体も既に検討されているため、単なる「VLM説明へ反実仮想を適用」では差別化できない。[TU Wien record](https://repositum.tuwien.at/handle/20.500.12708/226904)
- FaithLM (EACL 2026)は説明内容に反するヒントを与え、予測変化で説明の介入的faithfulnessを測る。これは主にLLMの説明研究であり、動画の時間情報へ直接対応するものではない。[ACL Anthology](https://aclanthology.org/2026.eacl-long.177/)
- より基礎的なNLE faithfulness研究は、説明が主張する理由を操作するテストを提案している。したがって本研究で必要なのは、新しい名称のscoreを作ることではなく、動画に固有の、妥当な反実仮想・control・時間単位を構成できるかである。[Atanasova et al., ACL 2023](https://aclanthology.org/2023.acl-short.25/)、[Siegel et al., ACL 2024](https://aclanthology.org/2024.acl-short.49/)
- 追加でCLEVRER公式仕様を確認したところ、train/validationのcounterfactual QAには各選択肢のfunctional programと正誤ラベルがあり、全データには物体軌跡・衝突event annotationもある。データセットにはcounterfactual questionが37,253件あると原論文が報告する。[公式README](https://data.csail.mit.edu/clevrer/README.txt)、[CLEVRER原論文](https://arxiv.org/abs/1910.01442)

## 3. 推奨する次の問い（まだ実験承認ではない）

> 動画VLMが回答後に述べる「回答を決めた動画上の出来事」は、同じ動画の妥当な時間的反実仮想でその出来事を変えたとき、回答分布が予測可能な方向に変わるか。また、単なる同時間長・同位置のsham編集より選択的な変化か。

従来のhead群説明ではなく、モデルが言語で明示した動画イベントclaimを評価対象にする。研究目的は引き続き動画VLMのexplanation faithfulnessだが、対象は自然言語の理由文であり、過去のA–E margin実験とは別estimandである。

### 定義案A: claimに対応する動画counterfactual

各動画QAで、まず回答`A`を記録し、その後に短い理由文`E`を別promptで生成する（predict-then-explain）。理由文から検証可能な動画イベントclaim`c`を人手または固定ルールで特定する。反実仮想動画`V_c`はclaimのイベント特徴だけを変え、その他を保つ。対照`V_sham`は同等の長さ・編集量で、claimと無関係な区間を変える。

`Δ_c = score(A | V, Q) - score(A | V_c, Q)`

`Δ_sham = score(A | V, Q) - score(A | V_sham, Q)`

`d_i = Δ_c - Δ_sham`

ここで`score(A|.)`は元の回答を固定した回答log-probability margin。正の`d_i`は、説明claimに対応した反実仮想がshamより元回答を弱めることを表す。ただし、この定義だけで説明全体の忠実性やモデル内部の唯一の理由を証明するわけではない。動画編集が対象イベント以外も変えた場合は因果解釈できない。

### 定義案B: CLEVRERの既存programを使う低コスト候補

CLEVRER train/validationの既存counterfactual questionsは、各回答選択肢にprogramと正誤ラベルがある。これを使い、モデルが生成したイベント説明claimと、選択肢programが表す対象イベント/物体除去の関係を照合する案がある。A案のように動画自体を編集せずに始められる可能性があり、既存annotationで候補QA数・question pairing・claim-to-program対応をCPU監査できる。

ただしprogramと説明が一致し、回答も正解することは、説明がモデル内部の実際の判断過程を反映した証拠ではない。これはまず「counterfactual rationaleの正確さ/構造整合性」であり、faithfulnessを名乗るには説明claimに対する介入比較（例: claimを反転・除去したときの固定回答marginの変化とmatched textual-sham）が別途必要である。質問文自体を変える操作と動画内容を変える操作も同一視しない。

### GPU前に通す設計ゲート

1. **既存の構造的counterfactualを先に監査する。** 公式CLEVRER train/validation QA JSON、answer/program、object trajectory/collision annotationがローカルで利用可能か、question種別・scene対応・複数選択肢のprogram形式をCPUで数える。利用可能なら、まずprogramと生成説明の対応可能性を小さく監査する。test splitはanswer/programが省かれるため、確認用の正解根拠として使えない。
2. **動画を変える案の供給源も区別する。** NExT-GQAの既存動画・時間区間注釈だけでは、現実的にイベント内容を変えたcounterfactualは得られない。black/blurや単純なframe deletionをcounterfactualと呼ばない。CLEVRERのscene/causal graphから新しい映像を再レンダリングできるか、または人手検証済みの自然動画対があるかを調べる。
3. **説明claimを検証可能にする。** timestamp付き説明を要求するだけでは時間正解にならない。CLEVRERではevent descriptionとprogramの対応を確認し、曖昧/未対応claimを事前規則で扱う。NExT-GQAならhuman segmentとの一致と介入可能性を別に記録する。
4. **faithfulnessと正確さを分ける。** programと説明claimの一致率だけをfaithfulnessと呼ばない。説明claimへの介入とmatched shamが実現できるか、あるいは測れるのが構造的説明正確さまでかを事前に決める。
5. **比較条件と統計を固定する。** 元回答を全条件で固定し、回答marginとanswer flipを別に報告する。主estimand、実質差、95% CI、除外、複数claimの動画単位集約、複数検定補正をsmoke前に決める。
6. **最初はCPU監査と少数手動例だけ。** data source・annotation・license・入力適合性を通過しない限りGPU推論コードを書かない。GPU smokeは設計ゲート通過後だけ。

## 5. 現時点の判断

この方向は、過去結果を使ったpost-hoc rescueではなく、評価対象（head attributionから自然言語video-event claimへ）そのものを変える新しい研究課題である。画像VLM・LLMには直接隣接研究があるため、現時点で新規性や実行可能性を確定しない。最大のボトルネックはGPUではなく、対象claimだけを変えた動画反実仮想と、それに見合うsham対照を用意できるかである。

公式CLEVRER仕様では、train/validation QAの各選択肢にfunctional programと正誤ラベルがあり、物体軌跡・衝突event annotationにもアクセスできる。一方testには回答/programがない。CLEVRER-Humans側のCEGは人間が書いた因果説明だが、既存監査では明示的な時間区間ラベルは確認されていない。従って既知のHuman CEGファイルを公式QA/programの代用品として扱わない。

今回、質問文・回答・programを表示せず、counterfactual question候補、answer/programの存在率、choice構造を数えるCPU監査と、件数だけをGitHubへ公開するrunnerを実装した。監査結果は
[`results/clevrer_program_schema_audit_20261008.json`](../results/clevrer_program_schema_audit_20261008.json)
に公開済みで、train/validation合計228,940問すべてでanswer/program-key/scene-linkの存在を確認し、counterfactual型は27,975問だった。
選択肢212,073件でもanswer/program-keyの存在を確認した。これはキーの存在数で、program payloadが非空・実行可能であることを保証しない。質問と生成説明の対応やfaithfulnessも測っていない。

最初のUbuntu検索は`*/questions/train.json`に限定していたため見つからなかったが、質問JSON自体は後続の監査で読み込まれ、匿名化された集計がGitHubに公開された。schema auditではtrain 152,572問・validation 76,368問を確認し、counterfactual型は計27,975問だった。続く同一scene監査では、train 8,642 scene・validation 4,333 sceneでcounterfactual質問が2問あり、そのすべてで完全program署名が異なった。各splitでcounterfactual質問を含むsceneは全sceneの10,000/5,000、sceneあたりの最大質問数は2だった。出典はcounts-onlyの[scene監査結果](../results/clevrer_program_pairability_audit_20261008.json)。train/validationの入力ハッシュは先行schema監査と一致する。

この結果は同じ動画内の質問候補があることまでは示すが、以前の「全ペアでprogram署名が異なる」という解釈は撤回する。署名計算はchoice-levelではなくquestion-levelのprogram欄を使っていた。field監査ではquestion-level payloadは全counterfactual行でlist型、dictノード0件だったため、choice-level program構造は未検査である。[pairability結果](../results/clevrer_program_pairability_audit_20261008.json)と[field監査](../results/clevrer_program_field_schema_audit_20261008.json)はこの制約付きで読む。operator trace全件不可という結果も、抽出対象の誤りで説明できるためプログラム構造に関する証拠ではない。

そこでCPU監査を修正し、非空のchoice-level programを優先し、question-level欄はfallbackにする。新しい集計では、フィールドの存在とpayloadが空でないことを区別し、question-level/choice-levelの型・キー構造を別々に報告する。raw program値、質問、回答、scene IDは出さない。修正版のテスト後に一度再集計し、そのchoice programからoperator traceを比較する。修正後もfaithfulnessへ直行せず、動画一定・対象イベントのみ差分という意味上の条件が成立するかをゲートにする。ゲートを通過しなければmask/head探索を再開せず、卒論を「忠実性を検証する方法の限界と負の確認結果」の範囲でまとめる。

この計画は研究方向の推奨案であり、counterfactual動画作成・外部編集API・長時間GPU実行の承認ではない。

## 6. 後続データ適合性監査の結論（2026-10-08追記）

上記のCLEVRER案について、question JSONの探索・choice-program署名・公式executorに基づく演算列・質問文token距離のCPU監査を実施した。最終結果と判断は[実験ログ](../experiments/2026-10-08_post_confirmation_direction_review.md)に記録した。異なるchoice-program署名を持つ12,974ペア中、演算列一致は158件。さらに質問文token距離が2–3のものは12件のみで、4以上が146件だった。これらの表層・構造指標は意味上妥当な反実仮想pairを立証しない。

よって、本レビューで提案したCLEVRER-based faithfulness実験はデータ適合性が確認できず、現時点で停止する。GPU実験、個別ケースのpost-hoc選別、token距離閾値の追加調整は行わない。今後再開する場合は、別個に定義したestimand、意味的pair基準、個別例を結果に盲検化して評価する手順を先に作る。本節が本レビューの最新判断であり、上記の当初案より優先する。

## 7. 視覚説明faithfulnessの追加先行研究監査（2026-10-15追記）

- Menzat et al., [Counterfactual Tests for Measuring Chain-of-Thought Faithfulness in Visual Language Models](https://arxiv.org/html/2609.06704)（arXiv:2609.06704、Findings of EMNLP 2026採択）は、A-OKVQAとSNLI-VEの画像ペアに対してvCT/vCCTを適用する。元画像と単一物体を編集除去した画像でVLMの選択肢分布を比べ、説明が編集対象に言及するかを評価し、再構成controlも使う。画像に対する同一生成器の説明・回答変化整合性は既に扱われている。動画フレーム、claim-to-time mapping、等量の時間control mask、gold-vs-best-foil log-margin差の組合せとは異なる。
- Liu et al., [On the Faithfulness of Visual Thinking](https://arxiv.org/html/2510.23482)（arXiv:2510.23482）は画像のzoom-in/tool-use型multimodal CoTで、視覚cropをnoiseに置き換える介入とテキストCoTの誤り注入を行い、答えaccuracyを比較する。視覚要素の介入で答えへの寄与を調べる枠組みは近いが、元動画の時間frame・外部にレビューされた自然言語claimとのmapping・matched temporal controlは扱っていない。
- Moll et al., [Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations](https://arxiv.org/html/2510.11196)（arXiv:2510.11196）は、胸部X線VQAの画像・prompt介入と、放射線科医の知識ベースに基づく説明評価を組み合わせ、4名の放射線科医による読影評価も報告する。画像VLM説明faithfulnessと専門家評価の先行例だが、動画時間情報や本計画のframe-mask margin estimandとは異なる。

この更新は「視覚説明faithfulnessへの介入評価」一般を新規性として主張できないことを明確にする。残る候補差分は、**動画上で事前固定・独立レビューされた自然言語回答claimの時間区間をmaskし、同数の最近接時間control maskと比べ、固定gold-vs-best-foil log-margin差を測る**というプロトコルの組合せに限られる。これも本PDCAで確認した限定検索に対する候補差分で、firstnessや全世界的な新規性を立証しない。研究価値の判断では、NExT-GQA適合性、人手mapping、権利、SESOIの未解決ゲートを通過した後も、この動画領域への適用が卒論として十分な独自性を持つかを再評価する。
