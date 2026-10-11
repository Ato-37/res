# Patch scope smoke結果と対応比較

## 目的・結果

ユーザー添付ログ（3件、5層、all/prompt/answer）で90 self-checkすべてPASS。
baselineは表示精度で3条件一致。sample1のanswer rescueは全層0でall=prompt。
単一回答tokenの場合の期待と整合するが、ログだけではtoken数を確認できない。
sample0だけがswapで悪化（-1.0977）、他2件は改善している。

sample0 layer6: all=0.3432, prompt=1.4797, answer=1.1663。
非加算項all-prompt-answer=-2.3028。layer12は-1.1740。
layer27は0.8253-0.1875-0.6378=0（表示精度）。最終層では位置ごとの
スコア寄与が分離する構造とも整合するが、実装構造の証明ではない。
3件から候補層や補正の有効性は確定しない。self-checkも因果妥当性の保証ではない。

## 改善実装

compare_activation_patch_scopes.pyで同一ID・同一入力配置・baseline・設定・
self-check成功・strict groundingを確認してから位置介入を対応比較。
欠落sampleの黙ったintersectionは禁止。非加算項とprompt-answer差、各scopeの
rescueを全件/悪化群に分けてbootstrapで記述する。CIは探索的・点ごとで未補正。
非加算項は位置介入間のscore上の相互作用で、head間の因果関係ではない。

## 検証・次の実行

標準ライブラリの合成データで非加算項、singleton、ID/入力/baseline不一致、
重複ID、self-check未完了/失敗をテスト。実GPU結果JSONはローカルにはない。
新規3件と既存9件、計12テストが成功。
まず既存smoke JSONをCPU比較（GPU再実行不要）。その後同じ5層・3scopeで
confirm heldout 25件をstrict grounding付きで実行する。既存25件は既に探索に
使用済みなので独立な確認試験とは呼ばない。スキップの一致を確認する。
条件間で成功集合が異なる場合は原因を調べてから比較する。
候補固定後に別集合で確認し、その後head群介入へ進む。

## 対応比較のユーザー実行結果・25件への移行

smoke比較が成功。悪化群は1件のみでCIはNA。
layer6非加算項の全件平均-1.039479、layer12 -0.652353。
layer27は全3件で非加算項0、prompt-answer平均-0.423498。
小標本bootstrapでCIが0を跨がなくても層の確定根拠にはしない。
候補層・フレーム数・介入・self-check許容値を変更せず25件へ拡張する。

run_patch_scope_audit.pyを追加。旧結果のsamples_fileからmanifestを解決し、
必要件数と動画存在を先に検査する。新規出力ディレクトリを要求して上書きを防止。
all/prompt/answerを順次実行して比較する。途中失敗なら停止し、既存結果は保持。
最終比較は有効集合の一致を要求する。同一条件で生じたskipはそのまま記録する。
manifestが旧実行後に変わっていないことまでは旧JSONから証明できない。
今回も探索であり、新規独立確認試験ではない。GPU実行はユーザー環境で行う。

## 2026-10-02 25件結果とhead群介入への移行

3条件の有効集合は20件で一致し、4つのswap構築失敗に加え、strict groundingで
sample6も除外された。全件でself-checkはPASS。比較結果は
`results/patch_scope_audit_25/comparison.json`。

allの平均rescueはlayer0=-0.173278、layer6=-0.018257、layer12=-0.163273、
layer24=+0.030628、layer27=+0.061388。点wise bootstrap CIを多重比較補正して
いない探索値であり、layer0/12のCIが0を跨がないことも確定的主張には使わない。
layer27のharmed subsetは+0.494188だがn=10、CI[-0.038665,+1.231081]。
従って単一layer全出力patchから候補を選ぶ根拠は弱い。

非加算性はallでlayer0=-0.031814、layer6=+0.113199、layer12=+0.023689、
layer24=+0.161160、layer27=0。smokeで見えた大きな非加算性は再現しなかった。
位置scopeの追加探索は止め、o_proj入力のhead連結表現を直接patchする
`head_group_activation_patching.py`へ移行する。まずsmall smokeでhead layout、
self-check、head singleton/all groupを検証し、合格後に固定条件でheldoutへ拡張する。
これはhead間因果の証明ではなく、head群介入を可能にする診断基盤である。

## 2026-10-02 head-group smoke出力の監査

ユーザー提供ログのhead-group smokeはfull/swappedとファイル書き出しまで完了した。
ただし標準出力にはhead groupごとのrescueとself-check件数がなく、3行だけでは
head単独/allの比較もself-check合否も判定できない。JSONには記録される設計だが、
運用上の可観測性が不足していたため、sampleごとのself_check=PASS件数を表示し、
解析スクリプトでlayer/group coverageとself-checkを機械検証する。
既存の3件ログから新しいhead効果を推定しない。修正版で同じsmokeを再実行する。
旧版JSONにself-check個別記録が残っている場合は、解析側で期待件数を
`2 * layer数 * group数`から後方互換的に推定する。

smoke解析ではlayer6全head=+0.334425、layer12全head=+0.196932で、
直前のlayer全出力patchの値と一致した。o_proj入力でのhead分解はこの範囲で整合。
ただし単独headの順位はn=3、harmed n=1で探索的。固定隣接ペアを追加し、
pair rescue - singleton A rescue - singleton B rescueを相互作用指標として記録する。
この指標はscore上の非加算性であり、headの独立因果性の証明ではない。

隣接ペアsmokeの有効n=3、harmed n=1。layer6のpair excessは多くが負で、
例えば17+18=-0.588328、19+20=-0.836806、23+24=-1.092899、
25+26=-1.021518。一方、layer12でも多くが負で、17+18=-0.765027、
13+14=-0.783266、15+16=-0.568073だった。layer6の最大正は13+14=+0.093815、
layer12は2+3=+0.200582だが、いずれも点wise未補正CIが広くn=3。
よって固定隣接ペアで安定した協調作用は示されなかった。

heldoutではn=20、harmed n=10。layer6のpair excessは19+20=+0.428586
(CI[+0.050810,+0.953598])、23+24=+0.300041
(CI[+0.006588,+0.597941])など正方向のペアがある一方、layer6全体の単独head
平均はほぼ負、layer12全headは-0.163273。layer12のpair excessには21+22=-0.443885
(CI[-0.864650,-0.078865])がある。54ペアの点wise未補正探索なので、これらは
選択された候補であり、協調headの確定結果ではない。

隣接ペア固有の効果とhead番号距離の影響を分けるため、gap=2のskip1ペアを
固定controlとして比較するスクリプトを追加する。sampleごとに全adjacent pair
excess平均と全gap2 pair excess平均を比較し、ペア数を疑似反復として扱わない。

### 次の実行

選択済みharmed sampleや上位ペアを使わず、layers=6,12、groups=singletons,
adjacent,all、scope=all、strict-grounding、self-checkを固定してconfirm heldout
25件へ拡張する。有効集合は過去結果と一致しない可能性があるため、失敗IDと
self-checkを記録する。全28 singletonと全27 adjacent pairを含むため長時間実験で、
実行前に小規模smokeで動作確認済み。結果後にpair excessを解析し、再現がなければ
head群補正へ進まず、介入診断の限界として整理する。

## 2026-10-02 gap=2距離control結果

gap=2 controlも有効n=20で全self-check成功。layer6はadjacent平均+0.098607、
gap2平均+0.132270、対応差-0.033663、CI[-0.147346,+0.038913]。
layer12はadjacent平均+0.035628、gap2平均+0.035018、対応差+0.000609、
CI[-0.042358,+0.038160]。したがって、隣接head固有の協調作用はこの評価範囲で
支持されない。pair rescueの正方向はhead番号距離に特異的でない可能性が高い。
ただし比較はlayer6,12だけで、全層のhead-group協調を否定するものではない。

次段階は同じ介入・strict grounding・self-checkで`--layers all --groups all`を
実行し、28層のlayer全体patchを同一プロトコルでscreenする。既存の28層soft
ablation結果とは介入が異なるため、単純比較しない。`--layers all`を追加し、
手入力によるlayer取りこぼしを防ぐ。

既に直接実行を開始した場合は`publish_existing_head_group_result.py`を終了後に使い、
GPU再実行なしで同じ解析・manifest・commit・push処理を行う。

## 実験結果の自動publish方針

`.gitignore`で`results/`は除外されているため、直接GPUスクリプトを実行しても
結果はGitに現れない。`run_all_layer_and_publish.py`を追加し、git identityを
GPU実験前に検査してから、all-layer実験、CPU解析、manifest生成、対象3ファイル
だけの`git add -f`・commit・pushを順に行う。既存の無関係な変更をcommit対象にしない。
現在実行中の直接コマンドには遡及適用されない。
Codexチャット自身はGitHub pushを受信して自発的に新規返信を開始できないため、
次のユーザー入力時にpullして解釈する。自動的な数値解析はwrapperで完了させる。

## 2026-10-02 publish wrapperのGit identity失敗

初回の`run_all_layer_and_publish.py`実行では、remote側に`user.name`/`user.email`が
未設定だったため、`subprocess.run(..., check=True)`がPython tracebackで終了した。
GPU実験自体はまだ開始されていない。wrapperと既存結果publisherの事前チェックを
local config限定・`check=False`へ変更し、設定不足時に設定コマンドを含む短い
エラーを返すようにした。これにより、設定確認前に長時間GPUジョブを開始しない。

## 2026-10-02 全28層 all-head patch screen

公開結果`head_group_all_layers_heldout.json`はconfirm heldout 25件のうち有効20件、
失敗5件で、28層すべてself-check PASS。各層の全28 headをまとめてpatchする
単一groupを評価した。平均rescueはlayer0=-0.1733 (pointwise bootstrap 95% CI
[-0.3520,-0.0046])、layer12=-0.1633 ([-0.3165,-0.0116])が0より負で、他26層は
CIが0を含んだ。28層の未補正screenなので、layer0/12も確証的差とは扱わない。
全headをまとめた介入ではhead内の正負効果が相殺され得るため、これだけで単headや
複数headの協調を否定できない。

次は`groups=singletons`で全28層×28 headの単head patch screenを行う。これは784
条件の探索であるため、結果から有望headを選んでも同じデータ内の順位やpointwise CIを
確証に使わず、候補と分析手順を固定して未使用データで再確認する。`run_all_layer_and_publish.py`
に`--groups`指定を追加し、全層singletonsの結果も解析・manifest・commit・pushまで
自動化した。既存のhead-group smokeがsingletonsとself-checkのコード経路を検証済み。

## エージェントの結果受領後手順

以後、ユーザーから実行結果を受け取ったら、その結果の検証・統計と限界の分析、
次の改善コード実装、dated experiment log更新、remoteで実行する完全なコマンド提示を
一つの応答内で行う。全く同じログの再掲だけなら短く返す。長時間実験では同じ経路の
smokeが未確認の場合に先にsmokeコマンドを提示し、候補選択に使ったデータを確認に
再利用しない。リポジトリのagent guidanceもこの手順に合わせて更新した。

## 2026-10-05 全層singleton screenと多重比較

公開singleton結果は28層×28 headの784条件、要求25件中有効20件、失敗5件。
最大の平均rescueはL9/H19=+0.2392だがpointwise CI[-0.1572,+0.7830]。
他の上位もCIは0を含み、正方向候補の同一データ上の選択は確証にならない。
784条件中、未補正bootstrap CIが0を除外したのは負方向81、正方向2で、
このまま正の少数候補を選ぶのは多重比較上不適切。

解析器に全層×singleton 784検定をまとめた両側paired sign-flip max-|t|補正を追加し、
10,000回のsample単位符号反転で既存JSONを再解析した。対称なsample-level帰無差を
仮定する。補正後p<=.05は5条件すべて負方向(L5/H26, L7/H5, L10/H16, L8/H26,
L16/H3)、正方向は0件。最大平均の正方向headも補正p=1.0だった。従って現在の
screenから回復候補headを確証する結果はない。負方向もactivation patchが正しい
動画情報の害を証明するものではなく、hybrid inputでのlogprob応答として解釈する。

次は正方向平均上位5 head (L9/H19, L7/H26, L5/H23, L3/H4, L12/H21) を固定候補とし、
未使用video IDの新規sample setで、候補を含む5層の全singleton (140 head条件)を再測定。
5候補だけでなく各層の残りheadも同じ介入で測り、候補の選択性を比較する。5層は
screening結果を見て選んでいるが、確認sampleを独立化し候補/層/解析familyを事前固定する。
wrapperはlayer subsetを受け、sample JSON・raw結果・補正解析・manifestを一括で
commit/pushする。結果はpointwise bootstrapとmax-Tの仮定に依存するため、成功しても
モデルとNExT-GQAのこの介入・指標範囲に限定して報告する。

## 2026-10-05 fresh sample生成時のarchive path

候補確認用sample生成はeligible candidate 4,823件を確保した後、ZIPを開く段階で失敗。
指定は`~/nextgqa_sample/NExTVideo.zip`だが、以前確認したarchive位置は`~/NExTVideo.zip`。
動画抽出やsample JSON出力は始まっておらず、GPU実験にも未着手。`build_samples.py`は
指定archiveが見つからない場合にホーム直下の同名archiveを探し、存在しなければ候補パスと
代替方法を含むparser errorを出すようにした。再実行では先にGit pullし、同じcommandで
自動的に`~/NExTVideo.zip`を使わせる。

## 2026-10-05 実験結果publish時のnon-fast-forward

候補確認runはローカルで結果commitまで作成したが、`git push origin main`が
`rejected (fetch first)`で失敗した。実験処理自体はpush段階まで到達している。
Codex側からremote GPU cloneのローカルcommitは読めないため、GPUを再実行せず、
remote上で`git fetch origin && git rebase origin/main && git push origin main`を行う。
今後のwrapperはpush失敗後にoriginをfetchし、worktreeがcleanなら未公開commitを
origin上へrebaseしてpushを再試行する。競合またはdirty worktreeの場合は停止し、
ローカル実験commitを保持したまま明確な手動復旧案内を出す。force pushは行わない。

## 2026-10-05 singleton候補の新規動画確認

添付実行ログでは新規manifest 35件を生成し、5層×28 singletonを測定。
strict grounding/self-checkに成功したのは28件（各sample 280 checks PASS）、
mask構築失敗は7件（ID 4,5,19,20,26,27,31）。固定候補はscreenで選んだ
L9/H19, L7/H26, L5/H23, L3/H4, L12/H21。confirmationでの平均rescueは順に
-0.070653, -0.086184, +0.039957, -0.011784, -0.039827で、各pointwise 95% CIは
すべて0を含む。140 singleton条件のmax-|t|補正後p値も候補5件すべて約1。
したがってscreen上の正方向順位は独立動画で再現せず、当初の正方向head候補を
支持する証拠は得られていない。7/35の除外とn=28の不確実性は残る。

次の解析はheadを新たに選び直すのでなく、事前固定5候補それぞれについて、同じ層の
他27 singleton headの平均rescueとの差をsample内で計算する候補選択性contrastとする。
`analyze_singleton_confirmation.py`を追加し、5 contrast間のpaired sign-flip max-|t|
補正とpointwise bootstrap CIを出力する。これは同層head平均に対する特異性の確認であり、
単一headの因果性や相互作用を証明しない。分析スクリプトは合成/実験JSONをまだ
実行していないため、GPU machineでpull後に生JSONを解析する必要がある。

## 2026-10-05 次段階: head探索を止め、入力介入の特異性を検証

singleton確認の候補対同層平均contrastは、L9/H19=-0.0226 (CI[-0.1933,+0.1490]),
L7/H26=-0.0717 ([-0.2046,+0.0491]), L5/H23=+0.0790 ([-0.0515,+0.2133]),
L3/H4=-0.0105 ([-0.1813,+0.1396]), L12/H21=-0.0122 ([-0.1742,+0.1443])。
max-|t|補正pはすべて0.80以上。head検索の継続は停止し、これはhead不在の証明ではなく、
事前候補の再現・選択性が確認できなかったと解釈する。

`evidence_control_probe.py`を追加し、内部head介入を外して入力レベルの特異性を測る。
各sampleで同じ枚数のフレームを同じ置換操作で1枚ずつ置き換える。根拠armはstrict
grounding内の位置、placebo armは時間的に近い非根拠位置をtargetとし、双方で別の
非根拠frameをdonorにする。複数のtarget位置で対応反復し、同一sample内で平均差を取る。
primary contrastは
`mean(full - evidence_replaced) - mean(full - placebo_replaced)`。正値は根拠位置の変更が
同じ1-frame操作の非根拠位置変更より正解回答logprobを強く下げたことを表す。sample単位
bootstrap CIと符号反転検定を出す。現在のsample manifestには選択肢/foilがないため正解系列
logprobを使用し、選択肢marginとは呼ばない。結果は一枚単位の入力レベル時間位置特異性に
限られ、複数フレーム同時介入、head因果、XAI一般の妥当性を主張しない。

control位置はevidenceと重ならず、targetとdonorを同一にしない。根拠frame数が多くても
少なくとも2つの非根拠frameがあれば1-frame比較可能。evidenceもしくは非根拠frameの不足は
明示的failureとして残す。短い動画でframe sampling indexが重複する場合はunique indexに
deduplicateし、3 distinct frames未満はfailureにする。
まず旧confirmation manifestの2件でsmokeし（既使用データなので推論には再利用しない）、
その後に過去全sample manifestを除外した新規60件を作って本測定する。wrapperは結果と
manifestを対象限定でcommitし、non-fast-forward時はclean worktreeを確認してrebase/push。
この実験・解析は未実行で、数値結果はまだない。

### 2026-10-05 入力介入probe結果

GPU結果はGitHub commit `c9b35d1`で公開済み（実行コード親commitは`8545110`）。
60件要求、56件有効、4件失敗。失敗はstrict groundingがsampling frameに0件のID 32/34と、
groundingが15/16 sampled framesを占めて非ground controlが1枚しかないID 30/57。
56件に重複sampled indexはなかった。1-frame replacementのevidence drop平均は+0.0037
(SD=0.5712)、nearby non-evidence placebo drop平均は+0.0769 (SD=0.4733)。primary paired
selective contrastは-0.0732 (SD=0.6788)、pointwise bootstrap 95% CI
[-0.2603,+0.0932]、両側paired sign-flip p=0.4541、正値割合44.6%。この試験では、根拠位置の
変更が近傍非根拠位置の同一置換より正解系列logprobを強く下げる証拠は得られなかった。
CIが0を跨ぎ、小さな正の特異性も残るため「効果なし」とは結論しない。回答選択肢marginは
測らず、モデルが当該動画質問に正解していたかで層別もしていない。

判断: 追加head探索・sample数拡張は現時点で停止する。次の作業は、NExT-GQA task accuracy、
質問形式・回答選択肢を含む評価の監査と、根拠区間アノテーション/一枚置換の妥当性を報告する
こと。特に全動画で正解系列logprobを採点しているので、誤答動画の根拠依存を測ってしまう
可能性がある。今の指標とデータで有意差を追って探索するのは避ける。

### 次の実装: 既存probeのmultiple-choice task-validity監査

`build_samples.py`はNExT-GQAのa0-a4候補、qid/video_id/type、正解choice indexを保存する。
既に公開済みsample JSONにchoicesがない場合でも、監査スクリプトがsample video path,
question, answer, groundingを元CSV/gsub/mapに一意joinして後方互換に処理する。
`audit_evidence_probe_task_validity.py`は既存probe各recordのfull videoを使い、選択肢付きpromptで
A-E各tokenのnext-token logprobを1 forwardで採点。MC accuracy、Wilson CI、正解choice対最良foil
margin、question type別accuracyを報告し、過去のfree-answer selective contrastをbaseline正誤で
記述的に層別する。層別値はpromptの異なる診断間の事後的関連で、因果contrastではない。
新たなevidence/placebo画像条件は再実行せず、既存probeの動画・frame indicesのみ再使用する。

このMC scoreは選択肢promptに依存するtask-validity監査で、公式の自由生成評価を置換しない。
最初に既存56件のうち2件でjoin・choice tokenization・推論をsmokeし、問題がなければ全56件を
一度だけ監査する。実行後の結果は以下。

### 2026-10-05 task-validity監査結果と次のchoice-margin介入

監査結果はGitHub commit `0174ff8`で公開済み。既存probeの56件すべてを処理し、失敗0。
選択式promptでの正答は42/56=75.0%、Wilson 95% CI [62.3%,84.5%]。
正解選択肢対最良foilの平均next-token logprob marginは+2.063、sample bootstrap CI
[+1.277,+2.855]。question-type別正答率はCH 5/9、CW 22/28、TC 7/10、TN 8/9。
タイプ別は小標本の記述統計。

既存free-answer probe contrastを監査時の正誤で事後層別すると、正解群n=42で平均
+0.051 (CI[-0.087,+0.198])、誤答群n=14で-0.446 (CI[-1.004,-0.020])だった。
これは異なる回答形式の後付け層別であり、因果的なeffect modificationとは解釈しない。
誤答層のCIが0を外れる見かけも、正誤層別を含む探索、多重性未補正、少数例のため
確認的証拠ではない。監査はchoice-letter next-token prompt依存であり、公式自由生成精度
ではない。

次は元の56件で同じframe sampleと保存済みの同一evidence/placebo置換pairを使い、
介入後もNExT-GQAの5択正解margin (= gold A-E logprob - best foil A-E logprob)を測る
`evidence_control_choice_probe.py`を追加した。primary contrastはmarginについての
`(full-evidence replacement drop) - (full-placebo replacement drop)`。これは前回の自由回答
logprobより質問の正解選択を直接測るが、既に元probe結果を見た同一標本上の後付けoutcome
なので探索扱い。置換は依然hybrid/OOD入力であり、XAI説明の一般的妥当性や内部head因果を
証明しない。全件の方向が一貫し、効果が再現候補となる場合のみ、解析と候補/仮説を固定して
未使用video IDの新規confirmation setを作る。まず2件smoke、その後56件のchoice-margin測定。

### 2026-10-05 choice-margin介入probe結果

smokeとheldout結果はGitHub commits `06f1e7f` / `7a96880`で公開済み。smoke 2/2、
heldout 56/56で、いずれも失敗0。heldout全体のprimary-like探索量（全標本）は平均
selective margin drop +0.107 (SD=0.473)、pointwise bootstrap 95% CI
[-0.013,+0.231]、positive fraction 55.4%、sign-flip p=0.1003。従って全56件では
証拠位置介入の選択肢marginへの選択的効果は確証されない。

監査時点でunperturbed選択が正解だった事後群42件は+0.213 (SD=0.427、pointwise CI
[+0.089,+0.344]、sign-flip p=0.0015、positive 61.9%)。不正解群14件は-0.209
(SD=0.476、CI[-0.441,+0.036]、p=0.128、positive 35.7%)。ただしこの群分けは前の
free-answer probe結果を見てから注目した探索層で、同じ56 video上の追加outcomeである。
従ってcorrect群の未補正p/CIを独立した確認結果のように扱わず、独立確認用の仮説生成とする。

次段階では仮説を「介入前5択予測が正解の動画では、根拠位置置換が、同一doseの近傍
非根拠位置置換よりgold-vs-best-foil marginを低下させる」と固定する。新規unique videoを
60件作成し、baseline-correct群をprimary cohort、全件・baseline-incorrect群・correctness
interactionはsecondaryとして同時に出力する単一GPU scriptを追加した。2–3件smokeの後、
未使用video IDで60件を実行する。freshnessが確認されない限りconfirmatoryとは呼ばない。
choice-letter prompt依存、frame replacementのhybrid/OOD性、baseline-correct subgroupへの
条件付け、NExT-GQA/Qwenモデル範囲という制約を残す。head causalityや一般的XAI faithfulness
の証拠とは区別する。

#### Fresh confirmation sample作成の停止と修正

2026-10-05のsmoke sample生成は、CSVのqid=4でanswer textがa3/a4の両方に一致し停止した。
これはデータ破損ではなくラベル情報の制約である。公式NExT-GQA READMEはCSVが元の
correct answer_idを対応するテキストanswerに置換した形式と説明しており、qidは質問IDで
選択肢indexを表さない。従って重複する正解文字列のどちらがgoldか復元不能で、片方を
推定するのは不適切。

`build_samples.py`を修正し、5択が揃っていてもanswer textがちょうど一つのoptionと一致
しない候補は安全に除外し、曖昧/不一致/選択肢欠損の件数を表示して残りの候補探索を続ける。
固有一致は従来どおり採用する。局所テストで一意一致と重複一致を確認した。後続の修正版
smoke/sample generationおよびGPU測定の実績は次節に記録する。

#### Fresh choice-margin confirmation結果

後続の修正済みrunがGitHub commits `2013b98` (smoke) と `121b289` (heldout) で公開された。
最初に貼られた失敗ログは古いコードの実行であり、その後のsmokeは3/3有効・failure 0。
heldout manifestは60件、解析は57件有効・3件failure。失敗はいずれも
`evidence=[]` (ID `3201948915:6`, `4226455621:0`, `3593365697:1`) で、16 sampled framesに
grounding位置がなくmatched interventionを構成できなかった。破損動画ではない。
sample manifestと過去7 tracked sample manifestsの動画ID重複は0 (新規60、過去unique 184)。

主要評価のbaseline-correct群は38件。selective margin drop平均+0.0849 (SD=0.5868)、
pointwise bootstrap 95% CI [-0.0928,+0.2743]、符号反転p=0.3938、positive 52.6%。
全57件は+0.1018 (CI[-0.0596,+0.2702], p=0.2299)、baseline-incorrect 19件は+0.1355
(CI[-0.1697,+0.4737], p=0.4527)。correct-minus-incorrect interactionは-0.0507
(CI[-0.4250,+0.3059])。独立fresh setでは前回探索で見えたbaseline-correct群の正方向効果
(+0.2125, n=42)は再現されず、群間差も示されなかった。

判断: 一枚置換・A-E margin・このNExT-GQA/Qwen構成の範囲では、根拠位置に特異的な選択
margin低下のconfirmatory evidenceは得られていない。追加sampleやsampling閾値を変えて同じ
効果を追跡することはせず、現仮説を保留する。3件のcoverage失敗を除外理由として報告し、
実験を停止する。将来再開する場合は別途研究上の根拠を定め、coverage基準とprimary estimandを
事前固定した新しいstudyとして扱う。出力はprompt依存のnext-token choice scoreであり、公式
自由生成精度、head causality、一般的XAI faithfulnessの証拠ではない。

#### 次の実装: grounding necessityをchoice marginで測る予備テスト

一枚置換の非再現を受け、同じ置換仮説を反復せず、研究計画本来のMECD-inspired verification
へ移る。`necessity_mask_test.py`はfree-answer sequence logprobを使い、nearest-frame fallback
がgrounding外を根拠扱いしうる。そこで`necessity_choice_margin_test.py`を追加した。strictに
sampled timestampがgrounding intervalに入るフレームのみevidence maskとし、同数の非evidence
controlをrank-distance最小化で選び、full/evidence-black-mask/control-black-maskの5択
gold-vs-best-foil next-token marginを比較する。primaryは介入前にgold choiceを選べた例の
`(full-evidence mask drop) - (full-control mask drop)`。正方向はgrounded subsetのmaskが同dose
controlよりmarginを強く下げることを示す。evidence coverage無し、control不足はfailureとして
残し、nearest fallbackで補わない。mask位置/count一致、統計summary、環境manifestを保存する。

これは当初からのverification段階に沿う別介入だが、新しい評価実装なので確認的な主張ではない。
black-frame OOD、prompt依存next-token margin、正解群への条件付け、時間位置/content差は残る。
古い一枚置換の60件結果は再利用せず、全sample manifestを除外した新規videoを作る。smokeを
先に行い、動作確認後に60件を測る。実装後のCPU unit tests/static checksを行う。GPU inferenceは
未実行。

#### Fresh choice-margin necessity結果と再現前監査

smokeと60件実行はGitHub commits `1c40b40` / `a4f48e8`で公開された。RTX 5070 Ti、
Qwen2.5-VL-7B-Instruct、4-bit/bfloat16、16 frames。smokeは3/3成功・failure 0だが、
primary baseline-correct cohortはn=1の平均selective margin drop=-1.250で、推論CI/pは
算出できない。heldoutは要求60、有効55、failure 5。baseline correctnessは43 correct/12
incorrect。事前指定primary (baseline-correct) は平均+0.3343 (SD=1.0683、pointwise
bootstrap 95% CI [+0.0233,+0.6541]、two-sided sample sign-flip p=0.0400、positive 60.5%)。
正方向は、grounding frame black-maskでgold-vs-best-foil marginがmatched controlより
大きく低下したことを示す。これは唯一の事前primaryであるが、小標本・対称帰無仮定の
sign-flip検定であり、CIはpointwise。all 55は+0.2045 (CI[-0.0568,+0.4750], p=0.1468)、
baseline-incorrect 12は-0.2604 (CI[-0.5938,+0.0417], p=0.1622)。correct-minus-incorrect
interactionは+0.5947 (pointwise bootstrap CI [+0.1519,+1.0603])だがsecondaryで未補正、
事前primary結論に代用しない。

重要なpilot重複: smokeの3 recordsはheldout JSONにそのまま含まれる。これらを除いた
52件の感度分析では全体平均+0.2764 (95% CI [+0.0288,+0.5601], sign-flip p=0.0462)、
baseline-correct n=42は+0.3720 (CI [+0.0595,+0.7054], p=0.0221)。これはpilot除外の
感度分析であり、結果を見てから算出したためprimaryの置換ではない。確認的独立性を
厳密に主張する際は、pilot 3件を最初から除外した新規sampleで再現する。

failure 5件のうち3件は16 sampled framesがgrounding intervalに一つも重ならず、2件は
evidenceが14/16または15/16を占め、同数の非-grounding controlを作れなかった。全manifest
60 video IDは一意で、ローカルにある過去の63 IDとは重複0。sample builderの実行ログでは
生成時に既使用275 video IDsを除外している。maskなし/black maskという分布外摂動、
choice-letter next-token marginのprompt依存性、baseline correctness条件付け、5 failureを
残す。従って結果は、このモデル・データ・設定でのgrounding-mask感度を支持する予備的
証拠であり、一般的faithfulnessやhead因果性の証明ではない。

次は効果を追って閾値やmetricを変えず、別seed・新規video IDによる事前固定replicationを
行う。GPU重いforward前に`preflight_necessity_choice_manifest.py`を使い、同一16-frame
sampling、strict evidence overlap、同数control構成可能性をCPU/video passで検査する。
候補80件から有効63件以上を必要とし、先頭3件をsmoke専用、次の60件をheldout専用に固定。
両manifestを分離し、audit reportもrun wrapper経由で公開するためsmokeとheldoutを重ねない。
preflightはラベルcoverageで解析対象を選ぶため、その適格基準と除外数を報告し、推論後に
成功sampleだけを選び直さない。今回の閾値/primary/scoreは固定し、独立replicationまで
確証的な一般化表現は避ける。

#### 独立sample replication結果

preflightと結果はGitHub commits `effc64f` / `e4dd895`で公開。80候補中72がcoverage/control
条件を満たし、8を事前に除外(7 insufficient_control、1 no_evidence)。先頭3件smokeと続く
60件heldoutは別manifestでsample overlap 0。heldoutは60/60成功、failure 0。以前の
necessity-choice heldout video ID 60件とのoverlapも0。

固定primary baseline-correct cohortはn=37、selective margin drop平均+0.3345、SD=.7448、
pointwise bootstrap 95% CI[+.1115,+.5811]、two-sided sample sign-flip p=.00530、正方向
64.9%。前回black-mask studyのfull primary (+.3343, n=43)と点推定がほぼ一致し、pilot 3件を
除いた前回感度推定(+.3720, n=42)とも整合。新replicationはsmokeとheldoutが分離されており、
独立video set上でprimary効果が再現したと判断する。ただし先行studyのsmoke重複という弱点は
消去できないため、両studyを完全に事前登録された独立confirmationsと誇張しない。

secondary all 60は平均+.2958、CI[-.0458,+.6146]、p=.0823。baseline-incorrect n=23は平均
+.2337、CI[-.5489,+.9457]、p=.5711。correct-minus-incorrect interactionは+.1008、
CI[-.6610,+.8998]。したがって、効果が正答例に特異的とは示されていない。primaryは事前に
一つ指定したが、sign-flip検定はsample差の対称帰無を仮定し、bootstrap CIはpointwise。
smokeはn=3で全例baseline-correct、mean+.583、CI[-.500,+2.500]、p=1.0であり判断材料に
しない。

結論を制限する点: preflight適格条件により80候補中8を除いた対象集団であり、16フレームの
strict grounding coverageと同数non-grounding controlが両立するケースに限られる。black frame
はOODとなり得るため、この効果はmask介入のscore変化であって一般的faithfulnessの証明では
ない。次は同じ60件・同じprimary/score/controlで、black fillをGaussian blur (radius 12 px)
に置き換える頑健性チェックを行う。これは同じvideosでの介入形式感度分析であり、独立replication
ではない。別smoke manifest 3件で実装確認し、heldout 60件でblur条件を測定する。

#### 2026-10-06 Gaussian-blur fill感度分析

blur結果はGitHub commits `84d66df` / `f8fc4c6`で公開。Qwen2.5-VL-7B-Instruct、RTX 5070 Ti、
4-bit/bfloat16、同一16-frame・同一60 video IDsでfull/evidence-blur/control-blurを測定。
smoke 3/3、heldout 60/60成功、failure 0。black条件とのpaired analyzerは両JSONの全sample ID、
baseline正誤、sampled/evidence/control indicesが一致することを検証。

baseline-correct primary n=37ではblur平均+0.4561 (SD=.9318、pointwise 95% CI
[+.1723,+.7601]、sample sign-flip p=.00390)、black平均+0.3345 (SD=.7448、CI
[+.1115,+.5811]、p=.00530)。blur-minus-blackの対応差は平均+.1216、pointwise CI
[-.0068,+.2770]、sign-flip p=.0995で0との差は明確でない。全60件のblur平均+.3604
(CI[-.0354,+.7208], p=.0595)、black平均+.2958 (CI[-.0458,+.6146], p=.0823)。
paired all-sample fill差は+.0646 (CI[-.0500,+.1896], p=.3101)。baseline-incorrect群と
correctness interactionも両fillでCIが0を含む。全p/CIはpointwiseで、black/blurとsecondary
cohortを跨ぐ多重比較補正は行っていない。

解釈: baseline-correct群のpositive selective margin dropはblack-only artifactだけでは
説明しにくく、強いGaussian blurでも同方向で観測された。ただしfill差自体は有意と示されず、
同じ60 videos上の頑健性分析であって独立replicationではない。blurもOOD摂動であり、残存する
色・粗い構図情報を保持する。よって「このprompt/model/sample cohortで、strict-grounding位置の
強い視覚情報抑制がchoice marginをmatched temporal controlより低下させる」と限定して報告する。
baseline correctness群以外への一般化、説明手法の妥当性、head因果性は主張しない。

次のアクション: 追加mask fillや同じ指標のGPU探索は停止し、この二つの独立video-set結果と
同一sample fill感度を卒論の限定的なnecessity評価として整理する。必要な追加作業はCPU上での
解析再現と図表作成に留め、主張/対象集団/除外基準を明記する。

#### 2026-10-06 卒論用の再生成可能な図表・集計

保存済みblack/blur/comparison JSONから、標準ライブラリだけで集計CSV、SVG forest plot、
再現性metadataを生成する`plot_necessity_choice_results.py`を追加した。GPU forwardは行わない。
実行前に両条件の全sample ID、video/qid、sampled/evidence/control indices、mask数、baseline
正誤、gold choiceが対応していることを検査し、比較JSONのpaired nも一致する場合だけ生成する。
入力JSONのSHA-256、生成日時、source commit、モデル/precision/frame数、CIと多重比較の注記を
metadataへ記録する。出力が既にあれば上書きせず停止する。

生成物: `results/necessity_choice_margin_thesis_summary_final_estimates.csv`,
`results/necessity_choice_margin_thesis_summary_final_forest.svg`,
`results/necessity_choice_margin_thesis_summary_final_metadata.json`。CSVにはall (n=60)、事前primaryの
baseline-correct (n=37)、secondaryのbaseline-incorrect (n=23)それぞれについてblack/blurと
paired blur-minus-blackの平均、SD、pointwise 95% bootstrap CI、sign-flip pを収録する。SVGは
black/blurの推定とpointwise CIを比較表示し、差の直接検定はCSVに残す。

CPUの合成データunit test 5件とpy_compileが成功。実JSONのsample対応を確認して3成果物を生成した。
中心的な読み取りは従前の通り: baseline-correct群でblack +0.3345 (CI[+0.1115,+0.5811],
p=.0053)、blur +0.4561 (CI[+0.1723,+0.7601], p=.0039)、blur-minus-black +0.1216
(CI[-0.0068,+0.2770], p=.0995)。blur差のCIは0を含み、どちらのfillが優れるとも示さない。
全区間はpointwiseで多重比較補正なし。black/blurはいずれもOODとなり得て、この図表は指定モデル・
prompt・16-frame sampling・eligible sample cohortにおけるchoice-margin介入感度を示すに留まり、
一般的なexplanation faithfulnessやhead因果性を示さない。

次は生成SVG/CSVを卒論で使う際、primary cohort・estimand・適格基準・OOD限界を結果本文と図注に
明記する。GPU実験の追加はこの整理のためには不要。

#### 2026-10-06 Ubuntu生成後のpublish経路を追加

直前のUbuntuコマンドは一時directoryへ図表を生成してテストするだけで、成果物をcommit/pushしなかった。
このためUbuntu実行後に結果がGitHubへ現れない運用上の欠落があった。`run_thesis_summary_and_publish.py`
を追加し、tracked変更/Git identity/入力/origin branchを事前検証、CPU tests実行、UTC付きunique prefixへ
集計生成、生成したCSV/SVG/metadataの3ファイルのみforce-add・commit・pushする一連の処理にした。
既存untracked dataは保持して対象限定commitする一方、tracked変更やbranch divergenceでは実行前に停止し、
force-pushは行わない。push時にoriginが進んだ場合は
既存のsafe fetch/rebase helperを使い、競合時はcommitを保持して停止する。成功出力にはbranch/commitを表示。

publisherのpath/上書き拒否/worktree状態/PYTHONPATH設定5 testsと既存図表・統計15 testsを実行し、
合計20 tests成功。runnerがunittest subprocessに`scripts/`をPYTHONPATHとして渡すこともテストした。
GPUは使用していない。次はUbuntu上で下記runnerを実行し、出力されたartifact commitを確認後、卒論原稿へ統合する。

Ubuntu実行で`git diff --cached --check`がexit status 2となった。再現検査でCSV writerの既定CRLFが
各行末のCRをtrailing whitespaceとしてGitに検出されることを特定した。CSV出力をLFに固定し、LF-onlyの
unit testを追加。publisherはwhitespace check失敗時にgitの診断内容を含む短いエラーを返すようにした。
失敗時にstage済みとなった生成3点は、次回runner起動時にファイル名・完全な3点組が一致した場合だけ
unstageする（生成ファイル自体は残す）。部分的なstageや他ファイルの変更は自動操作せず停止する。
修正版でpublisher 7 testsと既存図表・統計15 testsの計22 tests、py_compile、60件paired inputからの
図表生成、CSVのLF-only検査が成功した。再実行はGPU不要。修正版runnerをUbuntuへ提示する。

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_thesis_summary_and_publish.py
```

#### 2026-10-06 卒論本文・図注への転記ドラフト

確認後の次アクションとして`docs/thesis_results_draft.md`を追加した。保存済みJSONに基づき、estimand、
全件数、baseline-correct primaryとbaseline-incorrect secondary、black/blur各推定、paired fill差を
本文用に整理し、SVG向けの図注と限界文を作成した。一枚フレーム置換のfresh confirmationは非再現であり、
後続mask実験とは介入・estimandが異なることを明記し、前者の失敗を後者が覆したという誤読を避ける。
これは文章成果物であり、新しい実験や統計解析を追加したものではない。数値をtracked JSONと生成CSVに
照合した。次は指導教員の用語・論文構成に合わせて本文へ統合する。

#### 2026-10-06 Ubuntuでの集計成果物公開を確認

Ubuntu実行時の`git diff --cached --check`失敗はCSVのCRLF改行が原因だった。writerをLF固定し、
失敗後に残る完全一致の3成果物だけを次回runnerでunstageする修正を入れた後、Ubuntuの
`run_thesis_summary_and_publish.py`が成功し、GitHub commit `bbf8007`で集計CSV・SVG・metadataが
公開された。metadataのsource commitは`1ac9778`で、入力JSON 3件のSHA-256、Qwen2.5-VL-7B-Instruct、
4-bit/bfloat16、16 frames、paired n=60が記録されている。公開CSVの値を既存の本文ドラフトと照合し、
集計値に相違がないことを確認した。

再確認したprimary baseline-correct cohort (n=37) はblack +0.3345 (pointwise percentile-bootstrap
95% CI [+0.1115,+0.5811], sample sign-flip p=.00530)、blur +0.4561 (CI [+0.1723,+0.7601],
p=.00390)、paired blur-minus-black +0.1216 (CI [-0.0068,+0.2770], p=.09949)。全60件ではblack
+0.2958 (CI [-0.0458,+0.6146], p=.08229)、blur +0.3604 (CI [-0.0354,+0.7208], p=.05949)。
baseline-incorrect secondary n=23の区間はいずれも0を含む。区間とp値はpointwise/unadjustedである。

この確認でpublisherのCRLF障害が解消し、GPU再実行なしに成果物が再現可能な形で公開された。
結果の科学的解釈は既存ドラフトの範囲に留める: 指定モデル・prompt・eligible cohortにおける
choice-margin介入感度であり、blurがblackより優れる証拠、一般的faithfulness、head因果性ではない。
black/blur maskingによるOOD、baseline-correct条件付け、同一動画上のfill比較は残る。追加mask探索や
GPU実験は行わず、公開CSV/SVGとmetadataを本文・図表へ統合する。

CPU確認: 公開CSVの主要行と既存本文ドラフトの数値を照合し、metadataのpaired count・モデル設定・
入力ハッシュ記録を確認した。新規GPU実験は未実施・不要。

追加の再現性確認として、同じblack/blur/comparison JSONから一時directoryへCSVとSVGを再生成した。
CSV・SVGはGitHub公開版とバイト単位で一致し、再生成metadataもpaired n=60、model、4-bit/bfloat16、
16 framesが一致した。これは集計・図表生成の再現確認であり、元のGPU推論を再実行したものではない。
一時生成物は`/tmp`以下にあり、tracked成果物は変更していない。

次アクションの依存条件を確認した。リポジトリにある卒論関連ファイルは`docs/thesis_results_draft.md`
のみで、本文へ統合すべき実際の卒論原稿(docx/tex等)は存在しない。このため本文への直接統合は原稿の
所在が分かるまで保留する。新しいGPU実験・追加解析は不要。原稿がリポジトリ外にある場合は、対象ファイル
または貼り込み先の章・形式を受け取ってから編集する。

#### 2026-10-07 grounding-mask結果の研究上の位置づけを修正

ユーザーの指摘を受け、ground-truth根拠区間を隠してスコアが下がるという単純効果は予想可能であり、
それ自体を卒論の研究成果と呼べないと判断した。保存済みblack/blur heldout JSON (各n=60、failures=0)
からraw evidence/control margin dropを記述集計した。blackでは全件evidence +0.208 / control -0.088、
baseline-correct n=37で+0.304 / -0.030、baseline-incorrect n=23で+0.054 / -0.179。blurでは全件
+0.196 / -0.165、baseline-correctで+0.257 / -0.199、baseline-incorrectで+0.098 / -0.109。
これらのraw component平均にCI・検定は計算していない。control側の平均が負、すなわち平均的にmarginを
上げているため、selective差を「根拠maskでの低下」だけとして説明するのは不適切。

事前primaryであるbaseline-correct群のselective差はblack +0.334 (95% pointwise CI [+0.111,+0.581],
sign-flip p=.0053)、blur +0.456 (CI [+0.172,+0.760], p=.0039)だが、全60件ではblack +0.296
(CI [-0.046,+0.615], p=.0823)、blur +0.360 (CI [-0.035,+0.721], p=.0595)。baseline correctnessとの
interaction CIも0を含み、未補正のpointwise推定である。時間位置対応controlは映像内容・顕著性・回答識別
情報量を揃えていない。よって本結果は説明忠実性の実証ではなく、mask probeの予備的感度分析へ格下げし、
旧「卒論本文用ドラフト」を「内部検討メモ」に改めた。研究概要にもこの判断を反映した。

次のアクションは同じmask/score/subgroupで追加のGPU検定を行うことではない。もし研究を継続するなら、
説明手法が実際に選択した根拠と意味・識別情報量を対応させたcontrol、またはラベル因果性が既知の
counterfactualを用いて、「説明が選んだ部分の介入効果は、妥当なcontrolより予測的か」を事前固定する。
この再設計が卒論の期間・スコープに合うかは、プロトコル固定前に決める。追加GPU runは未実施。

#### 2026-10-07 evidence/control componentの探索的分解

ユーザーがUbuntuで再計算したraw平均は、保存済みheldout JSONの記録と一致した。単純平均ではblackの
全件evidence/control dropが+.208/-.088、blurが+.196/-.165。baseline-correct n=37ではblack
+.304/-.030、blur +.257/-.199である。margin dropはfull margin minus masked marginなので、control側の
負値はcontrol mask後にmarginが平均上昇したことを意味する。

この構造を再現可能に確認するため、`scripts/analyze_necessity_margin_components.py`を追加した。
black/blurのsample ID・protocol・mask indexを検証し、各sampleでevidence drop minus control dropが
記録済みselective値に一致することを確認した上で、raw各componentの平均・SD・pointwise percentile
bootstrap CI・sample sign-flip pを表示する。CPU tests 4件が成功し、実JSONはn=60、両fill failure=0で
解析できた。10,000 bootstrap draws、seed 20261007での事後的分解では、全件evidence componentはblack
+0.208 (95% CI [-0.096,+0.494], p=.1685)、blur +0.196 ([-0.127,+0.490], p=.2252)。primary正答群では
black +0.304 ([+0.101,+0.544], p=.0032)、blur +0.257 ([+0.041,+0.507], p=.0349)。同群のcontrolは
black -0.030 ([-0.203,+0.149], p=.7666)、blur -0.199 ([-0.389,-0.037], p=.0332)。

component analysisは結果を見た後に行った12比較（2 fills×3 cohorts×2 components）で、多重比較補正なし。
従ってこれらのp値をconfirmatory evidenceとして使わない。全件evidence-only平均のCIが0を含む一方、
blur primaryではcontrol mask自体がmarginを上げており、selective差を「根拠maskで下がる」だけと
説明できない。これはユーザーの「単純mask効果は自明」という批判を支持し、現在のmask研究を卒論の
主要成果にしない判断を補強する。GPU再実行は不要。次は研究質問自体を再設計するか、現範囲を限界付きの
予備probeとして閉じるかを、追加実験前に判断する。
