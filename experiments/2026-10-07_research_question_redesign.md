# 研究質問の再設計: 因果イベント判断に対する内部介入

## 目的

ground-truth intervalをblack/blur maskしてmarginが変わるという自明な入力感度を、研究の中心成果として
扱わない。人手の因果イベント注釈を評価基準にした内部介入の選択性という、次の候補質問と採否条件を固定する。

## 根拠として確認した既存結果

NExT-GQAの対応black/blur heldoutは各60件、失敗0。保存済みCSV/JSONと直近ログによれば、全件のselective
margin dropはblack +0.296 (95% pointwise CI [-0.046,+0.615], sign-flip p=.0823)、blur +0.360
(CI [-0.035,+0.721], p=.0595)。baseline-correct n=37はblack +0.3345 (CI [+0.1115,+0.5811],
p=.0053)、blur +0.4561 (CI [+0.1723,+0.7601], p=.0039)。ただし各p/CIはpointwiseで未補正。
baseline-correct subgroupは先行探索の独立confirmationで再現せず、controlの時間位置対応は内容や識別情報を
揃えていない。従って、mask効果から説明faithfulnessや内部headの因果性を結論しない。

## 今回の変更

`docs/research_redesign_protocol.md`を追加し、CLEVRER-Humansを制御環境での第一候補、ACQUIREDを実世界動画の
代替候補として整理した。因果edge margin、探索/確認動画split、内部singleton/group介入、対応陰性対照、
多重性、停止基準を提案し、特定headやlayerを既定の候補にしない。CLEVRER-Humansは動画が合成環境由来の
ため自然動画一般化に使えない。ACQUIREDについては今回確認した論文概要だけでは、必要なイベント時間区間・
因果graphの利用可能性と動画アクセス条件を確定できないため、採用を保留した。

## 解釈と限界

これは実験結果ではなく、次の実験前の研究計画案である。因果ラベルが人手であることだけで動画上の時刻が
分かるとは限らず、CLEVRER-Humansの各eventをフレームへ結び付けられるかは未監査。内部介入もモデルの
説明faithfulnessを直接示すわけではなく、まず因果QA判断への内部経路の選択的影響を評価する案である。
データ構造・利用条件・Qwenの無介入可否を通過するまでコード実装やGPU実験を行わない。

## Ubuntu archive確認とannotation取得器

ユーザーがUbuntuで報告した値はhome filesystem 1007G total/897G available、CLEVRER validation video archive
HTTP 200、`Content-Length=6206059321`、`Accept-Ranges: bytes`。容量は十分だが、CLEVRER-Humansの全1,108動画が
このvalidation split内にあるとは未確認なので、全5,000動画を先に取得しない。

Dryad公開記録は最新版annotation packageを3.05MBとし、`train/valid_ceg_data.p`、question JSON等を列挙する。
公式CLEVRER-Humans dataloaderはgraph entryの`scene_index`と`video_filename`を利用する。公式生成コードには
trajectoryから一部物理イベントのstart/end frameを得る処理もあるが、それだけで全human CEG nodeの時間区間が
あるとは言えない。注釈packageから動画名とCEG schemaを先に確認し、必要subsetを決める方針へ改めた。

Dryad packageの安全な取得用に標準ライブラリだけの`download_clevrer_humans_annotations.py`を追加。50 MiB上限で
stream取得し、期待6ファイル・ZIP traversal/symlink拒否・出力上書き拒否を検証し、pickleはロードしない。
Ubuntuで自動取得を試したところDryad download APIがHTTP 401を返した。これは認証拒否でZIP検証前の失敗。
そこで公式Dryadページからブラウザ取得したZIPを`--archive-file`で同じ検査・展開に通す経路を追加し、401時は
この方法を明示するエラーへ変更。認証トークンを保存・埋め込みしない。ローカル合成ZIP、既存出力保護、
path traversal拒否、401案内を含むCPU unit tests 6件成功。macOSの既定pycache先は書込不可だったため、
構文検査は一時ディレクトリを指定して再実施する。

## 検証

- AGENTS.mdと参照先claude.mdを確認。
- 研究概要、最新の2026-10-07結果ログ、内部検討メモを確認。
- 公開論文/データ配布の一次情報を確認。CLEVRER-Humans論文はCEGに人手の因果関係注釈を持つ一方、元動画は
  CLEVRERの合成物理シーンである。ACQUIRED論文概要は実世界動画とphysical/social/temporal counterfactual QAを
  報告するが、今回必要な時間区間注釈と配布条件は確認できなかった。
- ユーザー実行ログでDryad ZIPの取得・検査・展開成功を確認: 3,048,224 bytes、SHA-256
  `ceea34c48a1c50c70b1b7c158b5e69b395e8bffe847cdcb9563d2ebef48cc6e7`。6ファイルすべて抽出された。
  これはアーカイブ展開の確認であり、pickleの中身や時刻対応は未確認。GPU実験も未実施。
- ユーザー提供の初回監査JSON: train CEG 574 scenes / 2,769 nodes、valid CEG 525 scenes / 2,590 nodes。
  `CEG_pos` edgeはtrain 1,318・valid 1,406、`CEG_neg`は10,198・9,820で、各splitの`CEG_full` edge数は
  pos+negと一致した。合計1,099 scenesで、Dryad paperの1,108 videosとの差9 sceneの理由は未確認。
  question JSONはtrain 535・valid 183 top-level records、causal clozeは100 records。これはrecord数であり、nested QA
  総数ではない。初回監査は質問sceneとの交差を計算していない。
- 初回監査でCEG graph node/edge attribute内の既知の時間keyは両splitとも0件。ただしevent description textは走査
  しておらず、この値だけで自然言語eventからの時間対応可能性を否定はできない。明示時刻付き介入へ直行する根拠はない。
- 公式dataloaderはpickleを`pickle.load`し、sceneごとの`video_filename`とCEG graphを扱う。公式生成コードには
  `start`/`end`フレームを持つ一部自動抽出イベントがあるが、人手CEGノードの時刻根拠とは区別が必要。
- 監査CLIとunit testsを追加。JSON件数・キー、scene/video名、graph node/edge数、時間関連キーを要約する。
  pickle読込は任意の明示フラグに限定し、入力が公式Dryad配布物であることの確認を要求する。
- ユーザー添付のREADMEとv2監査: package READMEはCEG graph coverageを1,099 videosと明記し、実測の
  train 574 + valid 525 = 1,099 sceneと一致する。論文紹介の1,108はこのREADME上のCEG subset数ではなく、
  したがって差9は今回のpackage内の欠損とは扱わない。question JSONのscene recordsはtrain 535 / valid 183、nested questionは
  1,197 / 216で合計1,413。causal-clozeは100 scenesに431 nested questions。README記載の1,413
  question-answer pairsとは数値一致するが単位定義は要確認。1,724 event descriptionsが431問×4 choicesに対応する
  かはv3 choice-countで検証する。
- READMEはCEG 1,099 videos・5,538 events・23,032 relationship annotationsと記載。実データのv2監査は
  train+valid 1,099 scenesでscene数は一致する一方、CEG_fullは5,359 nodes・22,742 edgesでREADMEより179 nodes・
  290 edges少ない。この差は現段階では説明できず、データ版・集計定義の差として再確認が必要。
- v2のtrain question-to-graph照合0件は、question JSONのscene keyが非ゼロ埋め（例`3`）、pickle側が5桁
  （`00003`）であることによる監査実装の誤り。validは183/183一致。監査コードで5桁scene ID正規化を追加。
  cloze keyにも同じ正規化を適用した。訂正後coverageは未確認。
- v2の時刻キー0件はgraph node/edge attributes内の指定keyだけを検索した結果。description textには適用せず、
  temporal grounding不存在の証明とはしない。

## 次のアクション

監査runnerを追加して、訂正版のmetadata-only JSONを`results/`へ生成し、指定JSONだけをcommit/pushする。
これ以降はユーザーに端末出力の添付を求めず、GitHub上のartifactを確認する。scene keyを5桁に正規化した
train/valid/cloze coverage、nested question/choice数、およびREADMEとの差（179 nodes・290 edges）を再計算・照合する。原本pickle/質問文は
公開せず、監査JSONにも絶対パスを含めない。時間根拠と集計差が解消するまでは動画取得・GPU実験を保留する。

## v3監査結果とスキーマ監査への更新（2026-10-07）

Ubuntuで`--trust-official-pickle`を明示して実行されたv3監査をGitHub上で確認した。結果JSONは
`results/clevrer_humans_annotation_audit_v3_20261007.json`、公開コミットは
`4a51b1d9ffbbe4fdd36bba4cc134cbecac06b6e7`。source pickleや質問文本文、絶対パスは結果に含まれていない。

- trainはquestion scene 535件すべてがgraphに一致し、graphのみ39件。validはquestion scene 183件すべてがgraphに一致し、graphのみ342件。
- causal-clozeは100 scene中31件だけがtrain/valid graphと一致し、69件は不一致。これは不整合の可能性を示すが、配布READMEまたは公式loaderのsplit/ID定義を突合するまでは欠損と断定しない。
- nested questionはtrain 1,197、valid 216、causal-cloze 431。choice配列要素はそれぞれ1,197、432、1,724。QA-pairとchoice要素の意味的な対応はREADME等で定義確認が必要。
- CEGはscene 574+525=1,099でREADME記載のscene数と一致。node 2,769+2,590=5,359、edge 11,516+11,226=22,742で、READMEの5,538 events / 23,032 relationsとの差179/290は残る。現時点で集計単位やpackage版の差を説明できていない。
- graph node/edge属性に対する指定時間キーは0件。ただしv3報告では属性キー一覧自体を集約しておらず、自然言語descriptionやnode IDに埋め込まれた時間情報も未検査。従って「時間根拠なし」とは結論しない。

v4監査をGitHub上で確認した。`graph_attribute_key_occurrences`はtrain/validともnode側が空、edge側は`weight`のみ。edge attributeの値型は全て`int`で、node/edge attribute中の指定時間keyは0だった。edge weight値の分布はまだ集計しておらず、CEG field間に同一関係が重複して含まれるため、キー出現数をrelation件数と解釈しない。

公式dataloaderでは、`CEG_pos.nodes`のnode `n`を`"Which of the following is responsible for " + n + "?"`として質問文に用いる（公式[`dataloader.py`](https://github.com/xyang23/CLEVRER-Humans1.0/blob/main/description-gen/dataloader.py)の`convert_ceg_new`）。公式[annotation説明](https://sites.google.com/stanford.edu/clevrer-humans/labeling-interface)はイベント記述の正誤・因果関係と、イベントを探しやすくするbounding boxを説明するが、frame interval annotationは記載していない。ゆえにnode labelはevent-description文字列とみられる一方、動画時刻に直接対応するannotationは確認できていない。

v5監査をGitHub上で確認した。train/validの各CEG graph fieldにはそれぞれ2,769/2,590個のstring node labelがあり、指定したframe番号・時計表記・秒数パターンは0件、graph属性内のnamed time keyも0件だった。これは明示的時間locatorを検出できなかったという結果であり、自然言語上の時間意味を完全否定するものではない。一方、frame interval ground truthがない以上、時間区間削除によるfaithfulness主張はこのデータで行わない。

次はv6 CPU監査で`CEG_pos`と`CEG_neg`の辺を結果ラベルを出さず集計し、同一target eventにpositive/negative incoming edge候補があるscene数と組合せ数を確認する。これはwhole-video causal-edge QAの実行可能性を見るだけで、動画上の時間局在や説明faithfulnessの証拠ではない。ペアが十分あっても、実験に進む前に問いの構成妥当性・答えの定義・動画入力条件を別途確認する。まずv6 schema/集計をsmoke-testし、動画の大量取得やGPU実験はその結果を見て判断する。

## v6監査結果（GitHub公開を確認、2026-10-08）

`results/clevrer_humans_annotation_audit_v6_20261008.json`をremote mainのcommit `9c7f52f`で確認した。trainは574 scene、validは525 sceneを解析。`CEG_pos`/`CEG_neg`の双方に辺があるsceneはtrain 535/574、valid 472/525。同一targetに両方のincoming edge候補があるsceneはtrain 529/574 (92.2%)、valid 464/525 (88.4%)。両split合計で共通target 2,372件、候補の組合せ8,940件だった。正辺2,724本、負辺20,018本で負辺が約7.35倍あり、候補選択時のクラス頻度・候補難度の偏りを無視できない。

公式loaderの`convert_ceg_new`は、target `n`について`CEG_pos.predecessors(n)`を正答候補、`CEG_neg.predecessors(n)`を誤答候補として扱い、「Which ... is responsible for [target]」形式のQAを構成している（[公式loader](https://github.com/xyang23/CLEVRER-Humans1.0/blob/main/description-gen/dataloader.py)）。したがってv6はwhole-video二択causal-edge QA候補の存在を支持するが、辺組合せは同じscene/target内で重複し、8,940を独立標本数として扱えない。またモデル性能・内部介入の選択性・faithfulnessを示す結果ではない。

次にv7 CPU監査で、正負それぞれのincoming edgeがちょうど1本のtarget/scene数を追加集計する。これを二択プロンプトの最も単純な適格群の規模確認に使う。そこから先のGPU実験はまだ開始せず、候補数と論文上のestimandを再評価してから決める。

## v7監査結果と判断点（GitHub公開を確認、2026-10-08）

`results/clevrer_humans_annotation_audit_v7_20261008.json`をremote mainのcommit `eba2b97`で確認した。targetにpositive/negative predecessorが各1本だけある候補はtrain 62 targets/57 scenes、valid 33 targets/28 scenesだった。単純な「各targetで正1・負1のみ」という厳格条件では、held-out validは28 videos (28/525=5.3%)に限られる。

一方、どちらかの種類に複数候補を許し、そのtargetからpositive predecessorを1つ、negative predecessorを1つ選べば、validでは最大464 scenesが候補になる（v6）。sceneを統計単位として一つのtargetだけを事前seedで選ぶ設計なら、edge組合せの多い動画が分析を支配する問題を避けられる。ただし候補の選び方・難易度の分布は新たな実験設計事項であり、まだモデル実験は行っていない。

**研究目的に関する訂正（2026-10-08）:** 卒論の最重要目的は動画VLMのexplanation faithfulnessであり、因果edge QAへ置き換えない。上記候補数は注釈データ上の二択問題の可用性にすぎず、説明器の選択がモデル予測を忠実に反映するかを測定していない。CLEVRERは、モデル自身が選んだ説明対象への介入と、意味・情報量を揃えたmatched controlとの差を定義できる場合に限って、faithfulnessを検証する環境候補とする。時間intervalがない以上、時間groundingの外的正解にはならない。

次のアクションは追加GPU実験ではなく、faithfulness estimandの設計レビューである。説明対象、介入、matched control、必要性/十分性または選択性の指標、独立確認splitを明示できるまでは新しい実験を作らない。従来のNExT-GQA black/blur結果は人手ground-truth区間へのmask感度であり、説明器が選んだ対象のfaithfulnessを直接検証していない。head/group patchも頑健な説明回路の証拠を確立していないため、追加探索は停止する。
