# QTA–NExT-GQA 区間整合性とレビューartifact（2026-10-19）

## 目的

固定済みQTA question-swap selectorが選んだ時間群と、NExT-GQAの質問関連区間 `gsub` が粗い時間解像度でどの程度重なるかを探索的に調べた。これは生成説明の証拠レビューでも、予測への依存でも、faithfulness測定でもない。並行して、レビュアーが判定入力から限定的な構造監査票の作成まで同じ日本語HTML作業画面で行えるようにした。

## データ時刻の扱い

公式READMEは `gsub` をQA質問に対応する時間区間、`frame2time` をframe IDから秒への対応、`upbd` を動画サンプリング時刻として説明している（[NExT-GQA README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)）。公式 `getspan.py` は `frame2time` を時間群生成に使う（[getspan.py](https://github.com/doc-doc/NExT-GQA/blob/main/code/TempGQA/tools/getspan.py)）。このため `frame2time` を主時計としてデコードPTSと照合し、`upbd` は対応関係の仕様が確認できない補助診断に限定した。各質問区間は切り詰めず、映像PTS範囲と選択群のprocessor-grid対応を満たさない動画ペアは除外し、置換しなかった。

## 実行と由来

固定cohortの80動画・160質問を処理した。80動画は過去のselector機構解析に用いたtest-associated cohortであり、新規独立cohortではない。出力は `data/private_nextgqa/qta_question_swap_gsub_alignment_local_20261019_0200.json`、mode `0600`、SHA-256 `c9c4d714b63f924020a734995e673a7a85496f68975f4f928e111322d3151631`。全15個のコード・データ・固定計画・selector結果のhashは現物と一致し、記録された両repository commitは現HEADと一致することを別監査で確認した。実行コードには未コミット変更が含まれるため、再現時はcommitのみでなくartifact記録のhashを使う。rights statusは未確認、入力・派生出力ともローカル限定。

最初の集計出力は `started_at_utc` に終了近傍の時刻を誤記していた。時刻欄を関数開始時に取得するよう修正し、同じ固定cohortを再実行した。修正版では開始時刻が `2026-10-18T16:33:05Z` と記録され、同じ集計値を再現した。

## 観測結果

- 完了: 80/80動画、160/160質問の結合、138質問の注釈区間検査、130質問の時間群対応。
- 評価可能: 60/80動画ペア。除外20件はsource clock不整合11、注釈区間が動画PTS範囲外5、processor gridの群対応差3、全群で非局在1。置換はなし。
- 有効な質問行でQTA A/Bのmatched群IoU平均はそれぞれ0.0941/0.1145、cross-question群IoUは0.1164/0.1000。120個のA/B matched-minus-cross値の平均は **−0.00389**、中央値0。
- 2群ランダム基準IoU平均はA 0.0984、B 0.1007。注釈時間の被覆率はmatched群でA 0.1310、B 0.1461。
- 仮説検定、信頼区間、母集団推論は実施していない。

独立監査は、80=60+20、注釈区間カウント、群数分布、A/B平均からのmatched-minus-cross再計算を確認した。数値は記述比較にすぎず、group選択が質問の証拠を正しく特定したとは言えない。NExT-GQA区間は質問関連ラベルであって、モデル説明が指す証拠区間ではない。32フレームを16時間群へ集約する粗さ、test-associated cohortの再利用、20動画の除外、ブラウザ時刻とデコーダPTSの独立人手照合未完了が限界である。

## 日本語レビューartifactの変更

`docs/nextgqa_human_review_workspace_ja.html` は説明用ガイドから実作業ワークスペースへ変更した。承認済みのオフラインclaim/clock、15組frame-pair、prediction-link画面をsandbox iframe内で開き、判定出力、画像対package manifest、custody identity mapを同じタブへ読み戻す。入力hash、件数、review token、pair ID、clock anchor IDを相互照合し、区間値・時計時刻・判定内訳を含む構造監査票を保存できる。監査票には生のtokenやanchor IDを含めない。担当割当候補票も同画面で作成できる。frame-pair生成画面にはnetwork-deny CSPと日本語文言を追加した。

利用者の要望を受け、artifactの目的を「手順の説明」ではなく「人手監査の実施・照合・結果票作成」と明示する画面文言に整えた。説明レビュー、時計/画像対確認、予測リンク判定と、それらの記録の構造監査はartifact内で行う。真正なroot鍵・署名・引渡し順の確認はレビュー開始前の信頼準備であり、このHTML内の自己申告チェックでは代替しない。

artifact内の判定は**形式・件数の限定監査**である。署名・公開鍵root・本人性・review order・元動画とdecoder画像の一致・clock校正・production score gateは検証せず、採点を許可しない。開封前のrelease-order真正性確認は引き続き人間の信頼情報を要する。独立静的監査後にclaim interval有限値/範囲、clockの3×5固有アンカーとsequence、pair固有ID、予測レビューtoken重複などを修正した。

検査: 関連Pythonテスト群とQTA alignment testsは合計123/123成功。Python compile、HTML parser/unique-ID/link確認、Node JS syntax、6入力をtoken/hash/anchorで結んだvalidator fixture成功、不正区間・重複pair ID・古いclock timingの拒否、`git diff --check`を確認した。独立監査が初期validatorの抜けに加え、anchor IDが動画ごとに反復する設計を指摘したため、`(review_token, anchor_id)`で照合するよう修正した。別の監査で、前回票の残留とidentity mapの余分なキー受け入れを指摘された。新しい入力時に古い票を即時消去し、生成元のidentity schemaを完全一致で検証するよう修正した。Windows Chromeをheadless起動し、6種全入力の読込、時計/画像対の整合、sandbox iframe、監査票生成、token非掲載を合成データで確認した。正常時のヘッダーは `3/3, 15/15, 3/3, 0/3`。さらに成功票を生成した同一ページで認可フィールド付きの不正identityを再読込し、当該入力の拒否・古い票の消去・出力ボタン無効化を確認した。実動画の人手レビュー、実レビュー結果、署名真正性は未検証。

## 解釈と次の行動

この探索では質問注釈への局在に一貫した強い一致を確認できなかった。これは予測依存やfaithfulnessに関する結論ではない。結果を見てselectorや閾値を調整しない。次は真正な役割名簿・root pin・レビュー専用受領書・未開封引渡しが整った後、レビュー担当者がartifact上で実動画・時計・フレーム・予測リンクを監査する。人手でレビューするclaim、時刻・フレーム照合、prediction-linkは現在それぞれ0/3、0/15、0/3で、直接スコアも0/3。レビュー完了前の新たなGPU score runは開始しない。

## artifact内での開封前監査実装（追記、2026-10-19）

利用者の要望を受け、開封前の暗号・byte監査を別CLIと自己申告checkboxに任せる構成を改めた。ワークスペースHTMLへWebCrypto Ed25519/SHA-256検証を追加し、別経路root fingerprint、root署名integrity/release roster、freeze-owner receipt、witness attestation、固定82 role inventory、inventory対象の全ファイルhashを同じ画面で検証する。fresh ceremony challengeは画面内で128-bit乱数から発行し、attestation内IDとの一致を必須にした。署名済みhashと一致する `standalone_review_html` または `review_package_html` だけが対応するレビュー画面で開く。検証入力の変更・再検証時は既存画面/第2段階チェックを消去する。合格は手続上のレビュー資料開封だけで、モデル読込・スコアを許可しない。

独立監査で見つかったinventory外HTMLの受入れ、最終await後の検証状態競合、古いattestation再利用を修正した。合成鍵で構築した82 roleの署名済みfixtureをNode WebCryptoからHTMLハンドラへ読み込み、正常な署名・全hashでgateを通過、root pin不一致・inventory file改ざん・receipt署名改ざん・fresh challenge不一致を拒否、gate再失効時に画面選択/第2段階状態を無効化することを確認した。inventoryに一致するCSP HTMLは画面に読み込み、内容が異なるCSP HTMLはhash不一致で拒否した。検証中にroot pinが変更されても古いgate結果が有効に戻らないことも確認した。これらは合成データでのブラウザAPI/DOM-harness検査であり、実担当者・実鍵・実引渡し・動画レビューの証拠ではない。

**残るartifact監査範囲:** root fingerprintそのものの外部信頼、実世界の本人性/時系列、0600権限、動画とデコーダ画像の視覚的同一性、時計実時間校正はartifact単体で自己証明できないため、統合監査票に未検証として明記する。第2段階のbaseline確認も現在は画面上のチェック入力であり、baselineの署名付きmanifest自体をartifact内で再検証する追加作業が残る。証跡を持つレビュー担当者はまだ割り当てられておらず、実データレビューは0/3、時計・フレーム15件は0/15、prediction-linkは0/3、直接faithfulness scoreは0/3のまま。直接faithfulnessの効果量・CI・検定に新規進捗はない。

## 監査状態の持越し防止（追記、2026-10-19）

独立監査で、root/inventoryを切り替えた後も以前の6件のaudit JSON stateが残り、新しい開封前gateと古いレビュー票を組み合わせうる問題を確認した。gate入力変更時と再検証開始時に、6 JSON state・file input・表示ラベル・監査結果票を全て消去し、非同期読み込み世代も無効化するよう修正した。引渡しattestationの検証成功時はchallengeとattestation file inputを消費し、次の検証には新しいchallengeと立会人署名が必要になる。第2段階の任意HTMLと3つの自己申告checkboxは無効化し、署名済みpost-stage1 freeze receiptをartifactが検証できるまで開かない表示へ変更した。

検査: `node --check`成功。合成82-file署名fixtureを使うNode WebCrypto+DOM harnessで、gate成功、inventory一致HTMLの開封、不一致HTMLの拒否、pin変更後のgate失効と監査JSON file state消去、誤pin/改ざんfile/古いchallenge/不正receipt signatureの拒否、成功時のchallengeとattestation消去、stage2常時disabled、検証中pin変更の競合防止を確認。関連Pythonテスト156/156成功、`git diff --check`成功。これはNode harnessと合成鍵による機械検査で、Chrome実行、レビュー実行、担当者/鍵の実在確認ではない。

この修正でartifactは開封前検証、レビュー用画面の表示、構造・相互整合性監査票作成を同じローカル画面にまとめた。ただしレビューJSON自体の署名付き前段階receiptはまだ検証しておらず、audit JSONを外部で作成してからartifactに読み込ませる手順は残る。root変更時には一旦全JSONが消えるため、取り違えは防ぐが、JSONが今の画面上で実際にレビューされて生成されたことまでは証明しない。stage2のbaseline/PQ固定証跡とhash検証も未接続である。人手claimレビュー0/3、frame-pair/clock 0/15、prediction-link 0/3、直接faithfulness score 0/3で、直接faithfulnessの新しい効果量・CI・検定はない。GPU実行を正当化するレビュー済みcohortもまだない。

## artifact内の第2段階レビューと署名（追記、2026-10-19）

利用者の要望に合わせ、prediction-linkの別HTMLを読み込む導線を外し、baselineとcohort sourceを読み込んで質問・選択肢・claim・固定P/Qをこの日本語artifact上に直接描画するよう変更した。レビュー担当者は画面上で3件の判定・理由を入力する。割当済みprediction reviewerのPKCS#8秘密鍵を一時読み込み、生成したreceiptの正確なbytesをEd25519署名し、roster公開鍵でその場で署名を再検証する。成功時はreceiptとexact-byte adoption envelopeを別JSONとして保存し、同じ画面で構造・token順・source field hash・表示内容hash・P/Q再計算・root/release roster署名・adoption署名を検証する。秘密鍵入力は成功/失敗時にクリアし、ネットワーク送信コードは置かない。artifact監査票はscore/モデル実行を許可しない。

独立監査で見つかったPython/browser受理差（`schema_version: 1.0`、存在しないUTC日付、長さ128超token、壊れたUTF-8、孤立surrogate）に対し、JSON整数フィールドの小数/指数表記拒否、厳格UTF-8、surrogate検査、Python codepoint長、実在日付検査を追加した。Node WebCrypto+DOM harnessでは、Python生成の合成stage2 fixtureでreceipt/baseline/cohort/release rosterの検証成功、署名改変拒否、`1.0` version拒否、2026-02-31拒否、invalid UTF-8拒否を確認した。前の82-role gate positive/tamper/race testも再実行した。関連Python仕様テスト174件は成功。これらは合成artifactの機械検査で、実ブラウザ上のフォーム操作、実レビュー、鍵担当の本人性確認ではない。WebCrypto/PKCS#8とPython canonical署名bytesの独立比較は別監査で代表payload一致・OpenSSL署名検証が報告された。Chrome実画面での新UI E2Eは独立確認中。

**残る固定化問題:** cohort sourceのhash自体はbaseline・receiptとの内部整合を検査するが、freeze ownerが別経路で固定したcohort hashとの照合は未接続。このため署名は「rostered reviewer keyが、このreceipt bytesを採用した」ことのみを示し、候補cohortの外部固定化・本人性・実際に画面を見て判断したこと・判定の意味的正しさは示さない。stage1 claim/clock/frameレビューの署名済みreceiptもWebCryptoでまだ完結検証していない。レビュー担当と信頼root資料が未提供なので、人手claimは0/3、clock/frameは0/15、prediction-linkは0/3、直接faithfulness scoreは0/3。直接faithfulnessの効果量・CI・検定に進捗はなく、GPU実行を正当化するレビュー済みcohortもない。

## 探索的な短時間レビュー画面（追記、2026-10-19）

利用者の要望に合わせ、予備チェック用の別フローを作成した。日本語ランチャーのボタンからローカルartifactを開き、その画面でニックネームを入力すると質問へ進む。質問画面はローカルの3動画をHTMLへ埋め込んでいるため、担当・鍵・JSON・動画ファイルを個別に選ぶ操作は不要。質問は1件ずつ表示し、「だいたい合っている／一部ちがう／動画だけではわからない」のいずれかを回答する。時刻とメモは任意。回答を保存すると探索的な印象記録JSONをダウンロードする。

生成コードは `scripts/build_nextgqa_quick_review_artifact.py`、生成物はgitignore対象の `data/private_nextgqa/quick_review_ja_20261019.html`（約19.1 MB）。調査で元素材の2本はH.264/AAC、1本はVP6F/MP3と確認した。VP6FはChromiumで読み込めず、該当動画をH.264/AACへローカル変換し、偶数寸法が必要なため500×375を500×376へ1 pixel padした。他の2本は元bytesを保持。埋め込み後の3本はffprobeでH.264確認し、ffmpegで全編デコードした。回答JSONには動画識別子やhashを含めない。artifactには動画・質問・claimの保護対象データが埋め込まれるため、生成物と回答JSONはローカル限定で再配布しない。rights statusは未確認のまま。

この簡易チェックは本人性・役割の独立性・全編視聴・回答の正しさを検証せず、完全な幻覚判定や署名済み人手レビューに相当しない。元のstage1/予測リンク手続やconfirmatory cohort、採点許可を置き換えず、探索的な反応としてのみ解釈する。曖昧な回答を許容し、直接faithfulnessスコアや効果量は算出しない。

独立監査では、nicknameが本人性・独立性を証明しないこと、埋め込みデータの再配布漏洩リスク、元の独立盲検reviewと簡易チェックを混同しないことを確認した。1件ずつの表示と用途制限をartifactへ反映した。初回Chrome smokeで2件目の`readyState=0`を発見し、ソース調査で非対応VP6F codecを確認してから上記変換を追加した。現行artifactは動画データをBlob URLから再生し、個別file inputやnetwork接続を持たない。独立Chrome smokeではnicknameをURLに載せず、3件全てで有効durationがロードされ、最終画面と保存ボタンを確認した。合成入力で保存ボタンを押し、隔離一時フォルダーにJSONが1個作られることを数えた後、ファイルを削除した。JSON本文は開いておらず、実レビューではない。

追加の独立静的監査で、埋込JSON中の`</script>`がscript要素を閉じ得る問題、任意出力先、生成HTMLの既定ファイル権限が未制限という点を指摘された。inline JSONでは`<`を`\u003c`へescapeし、実際にscript終了文字列を含む合成payloadがHTML parser境界を壊さずJSON復元可能なことをunit testで確認した。生成先は`data/private_nextgqa`配下に制限し、生成HTMLをmode 0600に設定する。検査では3件の質問・claimとH.264化済み全動画の出力を確認し、2件のartifact安全性テストが成功した。人手レビューは未実施。実レビュー件数は従来どおり0/3で、faithfulness研究として直接効果量・CI・検定に新規進捗はない。

次は担当者が必要に応じてランチャーを開き、ニックネーム入力後に短い探索的確認を行う。結果は元の独立レビュー記録から分離し、実施されるまでは完了数に加えない。

## 簡易チェックの映像条件を統一（追記、2026-10-19）

独立UI監査で、簡易チェック画面は動画の音声・字幕を再生できる一方、確認対象のclaimは映像を根拠に生成されており、判断条件がずれる問題を確認した。新しい生成物 `data/private_nextgqa/quick_review_ja_video_only_20261019.html` では、3本とも音声・字幕ストリームを含めず映像のみを埋め込む。古い生成物は上書きせず残し、案内ページのリンクを映像のみ版へ切り替えた。案内文にも、ニックネーム後に3件を続けて回答して最後に保存する手順と、実名を避けたローカル保管を追記した。

テスト: 追加した音声・字幕除去のFFmpeg呼出しテストを含む簡易artifact suite **3/3成功**、Python compile成功。新しいartifactを現地packageから作成し、HTML内の3動画を個別抽出してffprobeで全てH.264映像のみ（音声/字幕なし）を確認し、全編ffmpeg decode **3/3成功**。artifactはmode 0600、約30.6 MB。source/test/案内に`git diff --check`成功。Chromiumでの実動画UI再実行はこの追記では行っていない。保護対象データ・回答は外部公開していない。

この変更は簡易印象チェックの入力条件を揃えるだけで、claimの真偽や完全な幻覚性の判定にはならない。現行作業ツリーに探索的JSONは見つからず、実レビュー実施を確認できていない。直接faithfulnessは引き続き claim review **0/3**、clock/frame照合 **0/15**、prediction-link判定 **0/3**、直接score **0/3**。直接効果量、CI、検定は未取得。GPU側では既定proxy/full-cohort診断が実施済みで、これらを繰り返しても直接スコアには進まない。人間の回答と正式なレビュー/信頼ゲートがない現状で、新しい実動画score GPU forwardを正当化する入力はない。

次は案内ページの映像のみartifactをローカルで開き、探索的回答を完了した場合だけ、その回答を正式レビュー記録と分離して読み取る。回答を受け取れない間は、CPUでphase-gateと署名済みstage-1/外部cohort-pin検証の合成テストを独立監査し、実鍵・実担当者がない段階でexecution authorizationを与えない。

## 外部cohort pinをphase receiptへ接続（追記、2026-10-19）

独立監査で、phase receiptは候補内のartifact bytesを署名していたものの、別経路で凍結したcohort hashをphase-1/phase-2検証へ渡して照合する入力がなかった。phase-1 artifact roleに `external_cohort_pin` を追加し、ordered `expected_cohort_tokens` と各 `source_rows`（question/options/claim/video/interval digest、およびsupport decision）からcanonical identity digestを作る非認可candidate builderを追加した。phase-1検証は署名receiptが束縛するpin bytesのSHA-256を必須caller値 `expected_external_cohort_pin_sha256` と比較し、pin内digestを現在のsource rowsから再計算する。phase-2検証も同じcaller pinでphase-1を再帰検証する。

検査: 合成Ed25519 keyで、pin一致、pin hash不一致、古いpin bodyに対するsource差し替え拒否、差し替えcohortを新pin/receiptで再署名した後も古い外部pinでは拒否、phase-2経由の古いpin拒否を確認した。関連5 suiteは **60/60成功**、Python compileと対象ファイルの`git diff --check`成功。独立監査でtoken/source row順序結合、署名済みrole経由、stale-pinケース、`model_load_permitted=false`、`scoring_authorized=false`、`authorized_forwards=0`を確認した。全て合成データ・鍵で、実レビュー、実trust root、GPU、モデルロードは未実施。

**信頼上の限界:** verifierはcallerから渡されたpin digestとの一致を確認するだけで、その値を別の信頼経路から取得した事実までは認証せず、reportも`external_cohort_pin_origin_authenticated=false`とする。production runnerはroot/freeze-ownerから独立に配布されたhashをtrusted configとして読む必要がある。候補builderの出力を同じ呼出し側がそのままhashして渡すだけでは外部pinにならない。このCPU構造検査は署名済みstage-1レビューreceiptのブラウザ内完結検証やphase gate/runnerを実装したものではなく、実行許可も与えない。

直接faithfulnessは引き続きclaim review **0/3**、clock/frame照合 **0/15**、prediction-link判定 **0/3**、直接score **0/3**。直接効果量、CI、検定に進捗はない。探索的回答JSONもローカルprivate data rootでは未確認。実データの直接scoreを伴うGPU runは今回も実行していない。次は署名済みstage-1 receiptのartifact内検証と、外部pinを受け取るproduction caller contractを、同じ非認可のまま独立監査する。人手review・15箇所対応・実root/role資料なしにGPUスコアを開始しない。
