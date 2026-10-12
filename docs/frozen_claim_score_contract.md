# 動画claimの教師強制score契約（未確定・実験前の設計メモ）

更新日: 2026-10-12

状態: scoreの確定・faithfulness実行承認ではない。人間QA、rights、claim対応、独立video評価のゲートは別に残る。

## 現行コードで確認できること

[`soft_ablation_sweep.py`](../scripts/soft_ablation_sweep.py)は動画とquestionをuser messageとしてchat templateに渡し、`add_generation_prompt=True`でpromptを作る。ターゲットには入力JSONの`sample["answer"]`を使い、`tokenizer(answer, add_special_tokens=False)`で単独tokenizeしてprompt tokenの後ろへ追加する。現在のコードには`claim`欄の取得、回答からclaimを抽出する処理、claimと時間区間を結びつける処理はない。このため既存出力の`correct_logprob`を動画説明claimのscoreとは呼べない。

`sequence_logprob`ではターゲット各tokenの条件付きlog probabilityを足し合わせる。prompt長をP、target長をAとすれば、`sequence_answer_logprobs`は`start = (P + A) - A - 1 = P - 1`からA個のlogitを切り出し、位置ごとに`log_softmax`を取る。長さで割る処理はなく、主値は**sum log probability**である。よって異なる長さの回答・claimを直接比較すると長さの影響を受ける。first-token metricはprompt末尾のlogitから`answer_ids[0]`を評価する別のscoreで、sequence scoreと同一のestimandではない。

現行helperのtext controlは、正解targetの各prefix位置でtoken log probabilityが近い語彙項目を選び、各control列を合計してcontrol列間で平均する。これはvideo内のmatched evidence/control chunkではなく、説明claimとの意味対応も持たない。

このhelperはfull assistant responseを生成・採点する処理でもない。回答候補のtarget文字列をpromptへteacher-forceする実装であり、回答生成時に何が出たか、どこで生成が止まったかは記録しない。

ターゲットに明示的なEOSやassistant turn terminatorを足していない。したがって、測っているのは指定された文字列token列のteacher-forced likelihoodであり、生成停止や完全なassistant responseの確率ではない。EOSを将来含めるなら対象列・予測logit位置・集計規則を先に固定し、既存scoreへ暗黙に追加しない。

## Token境界の実測と限界

[固定tokenizer診断](../results/qwen_claim_tokenization_cpu_20261012_01.json)は、固定した合成promptと12個の文字列について、`tokenize(prefix + target)`と`prefix IDs + standalone target IDs`を比較した。11/12で列が一致し、assistant完了messageをtemplate化した場合もprefixは11/12で保持された。唯一の不一致は固定fixtureの先頭改行を含むcase 10である。この結果は境界依存の可能性を示すが、旧sampleの実answer列が誤tokenizeされた、旧スコアが変わる、claimが意味的に正しい、とは示さない。

この文字列境界監査が適用されるのは、人間が付与したannotation文字列など、**文字列をtargetとしてteacher-forceする場合**である。その場合は固定したchat prefixとannotation文字列を連結したtoken列を監査し、prefix offsetとscore対象token範囲を記録する。

**モデル生成claimには連結再tokenizeを適用しない。** 生成後の対象は実際のgeneration tailのraw token IDsであり、文字列へdecodeしてtokenizerへ再入力すると、生成されたID列そのものを置き換えうる。`frozen_teacher_forced_score.py`はprompt長でtailを一度切り出し、ID列のhashを記録してimmutable targetにする。凍結後はID列をdecode/re-tokenizeして置換しない。合成tokenizer診断の11/12という結果は、generated raw IDsを再tokenizeする理由にはならない。

## 凍結target scorer / selectorの検証済み範囲

[`frozen_teacher_forced_score.py`](../scripts/frozen_teacher_forced_score.py)は単一のraw sequenceと呼び出し側のprompt長からtarget tailを凍結する。`caller_provenance_unverified=True`が明示され、target hashは受け取ったIDsの一貫性を確認するだけで、実際に指定model/promptから生成されたことやtokenizer/model revisionを認証しない。採点はbatch 1、paddingなし、長さが`P + L`のfull-sequence logitsだけを受け付け、target ID `i`をrow `P − 1 + i`で採点する。cache-sliced logits、beam/multisequence、別のoffset規則はunsupportedである。

現行のforward長は`P + L`、採点rowは`P − 1`から`P + L − 2`で、最後の出力rowは採点しない。因果的な厳密算術では`P + L − 1`個の入力で必要なrowを得られるが、これは異なる系列長での実行方針になる。固定GPU診断では長さ585と586で同じrow584のtoken log probabilityが0.0077603 nat異なった。短い方・生成scoreに近い方を結果から選んで現行policyへ代入しない。将来変更する場合はforward長・prefix scheduleを別versionの事前protocolで固定する。

scorerは機械的な全target sumとmeanをnatsで返す。呼び出し側が単一EOS IDを与え、最後のtarget IDがそれと等しい場合だけ`terminal_seen`となる。全target scoreはそのterminalを含み、content scoreは末尾terminalを除く。EOS IDや終了設定の出所は検証せず、multi-EOS設定も扱わない。EOSがないtailは`termination_unverified`とされ、生成が自然に終了したと認証するものではない。

[`frozen_target_token_selection.py`](../scripts/frozen_target_token_selection.py)は同じ凍結targetに対して、呼び出し側が指定した0-based・target-relative indexだけを集計する。selection hashはtarget hashとindex列に結び付くが、意味上どのtokenがclaimに属するか、JSON wrapper/EOSを含むか、選択者の判断が正しいかは検証しない。`semantic_selection_verified=False`であり、caller provenanceも未検証のままである。

この二つのpure CPU helperについて、[scorerの13テスト](../scripts/test_frozen_teacher_forced_score.py)、[selectorの7テスト](../scripts/test_frozen_target_token_selection.py)が通った。検査範囲はimmutable ID tail、固定した予測offset、sum/mean、末尾EOSの算術、選択indexのhash・範囲・集計とmalformed入力の拒否である。モデル推論、実生成のprovenance、claim意味、annotation適合性はテストしていない。

現時点で**生成されたJSON claim用のschemaはない**。full assistant responseと一つのclaimは同じ対象ではなく、全responseのlogprobをclaim scoreと読み替えない。生成claimを採点するには、まずresponse形式とclaim抽出/JSON schemaを凍結し、raw generation IDsとscore対象indexの対応を保ったまま、annotation・時間区間との意味対応を別途確認する必要がある。文字列fieldをannotationとして直接採点する場合と、生成raw IDs内からclaim部分を選ぶ場合のprotocolを混ぜない。

`build_samples.py`のwhitespace/casefold正規化は、NExT-GQA answer文字列と選択肢の対応を作る用途である。score対象文字列の正規化ではない。新しいclaim scoreでUnicode正規化、strip、句読点除去、空白圧縮をするかは未決であり、正解ラベル照合の規則を流用しない。

## 実動画GPUでの固定target採点経路の検証

[native生成とfull teacher forcingの比較](../results/qwen_frozen_teacher_forcing_gpu_smoke_20261012_01.json)は、使用条件を確認したNOAA動画1件・4フレーム、7B BF16/4bit/eagerで、2個のraw生成tokenを固定した。生成2回とfull teacher forcing1回のforwardは3/3完了したが、raw logitsの事前条件（atol0.1、rtol0.01）は満たさず、結果は`gate_failed`である。最大差1.0、平均絶対差0.14347だった。

固定target各tokenのlog probability差は最大0.00017959 natで、別の事前条件0.02 nat以内だった。sumの差は0.00018900 nat。予測rowは583/584、1つずらしたteacher forcing controlとの最大token-score差は23.1992 natで、位置ずれを区別した。実際の位置row・cache長も検査した。9.459秒、ピーク割当6.074 GiB。固定1入力・2tokenの数値診断であり、母集団の効果・CI・有意差ではない。

この結果を全体の互換性成功とは扱わず、許容差を変更しない。raw logitsの差がcache経路、系列長による計算、その他の数値要因のどれに由来するかは未同定。短い固定targetでのscore一致から、長い説明claimや介入下のscore一致・faithfulnessへ一般化しない。生成文・IDs・logitsを公開していないため、結果JSONだけから数値集計を独立再計算することもできない。

[後続の系列長・出力層診断](../results/qwen_sequence_length_numerics_gpu_diagnostic_20261012_01.json)は、同じ入力と2 tokenを固定して実行した。生成2回と系列長584/585/586のfull forward3回、出力層への単一行の再投影5回で、logits19組・内部表現9組を比較した。18.42秒、ピーク6.074 GiB、失敗0件。配列を保持した実行中のTorch64・NumPy・Python算術照合は最大約9e−14以内だった。モデル出力の同等性閾値を新設したものではない。

| 同じtoken予測位置の比較 | 内部表現の最大絶対差 | raw logitsの最大絶対差 |
|---|---:|---:|
| 生成token 0とfull584 | 0 | 0.0625 |
| 生成token 0とfull585 | 2.5 | 0.4921875 |
| 生成token 0とfull586 | 7.5 | 0.875 |
| 生成token 1とfull585 | 8.712890625 | 1.03125 |
| 生成token 1とfull586 | 8.484375 | 1.0 |

full584の同じ内部表現を出力層へ単一行で渡すと、生成token 0のlogitsを完全に再現した。この固定行では出力層の入力行数がraw差を説明する。一方、585/586では出力層より前にも差があり、行数による出力層の差だけでは全体を説明できない。5行のfull投影と単一行再投影の差はraw最大0.03125–0.0625、固定tokenのlog probability差は8.55e−12–8.31e−11 natだった。特定のkernelや量子化を原因と同定した、因果maskが正しい、動画説明が忠実である、とは結論しない。現在の重み本体は[別のCPU照合](../results/qwen_checkpoint_bytes_cpu_20261012_01.json)で公式の固定revisionの5ファイルと一致したが、過去の実行や照合後の変更を継続的に認証するものではない。

BNBの行数依存を切り分けるため、動画やモデルを使わない合成3584×3584線形診断も行った。[結果](../results/bnb_row_count_gpu_diagnostic_20261012_01.json)では、BF16線形、BF16値をFP32へ変換した線形、FP4 BNB、BNB重みをBF16へ戻した線形の4条件すべてで、共通行のM=584/585/586の差は0、同一M=584の反復差も0だった。M=1だけは各条件で差が出て、BNBとその直接dequant化線形もM=1では差があった。これはこの合成入力・実装経路での行数と計算経路の診断であり、Qwenのattention・RoPE・causal maskやCycle13の隠れ状態差の原因を同定するものではない。BNBのPython分岐は記録したが、コンパイル済みCUDA内部kernelの同定はしていない。

## CaFとの測定対象の違い

[Yeo et al.のCaF論文](https://aclanthology.org/2025.emnlp-main.529.pdf)は、回答と説明について破損した入力span以降の対象token・layerごとの介入効果を作り、その二つの帰属mapを`CaF = 1 − cosine distance`で比較する。出力tokenの効果を平均したmapの一致であり、本資料で検討している「固定したclaimのlog likelihoodが、対応する動画区間への介入に選択的に依存するか」とは測定対象が異なる。論文のNLP/Gemma2での結果を動画の意味対応・説明の真実性の検証とは扱わない。

[公式実装の固定commit](https://github.com/SenticNet/causal-faithfulness/tree/5fe379e0ac066623c60dcaec98536b3981e22dfe)では、静的に確認したpost-hoc説明経路では、promptにモデルの保存済み予測回答を含める。説明は生成tailを特殊tokenを除いてdecodeし、条件に応じて切り詰め、stripして再encodeしたIDsを固定し、元入力とcounterfactual入力で採点する。固定対象は整形後の説明列であり、raw generation tailをそのまま保持する本helperとは異なる。コードのtoken probability集計もsum log probabilityとは異なる。この違いから論文の結果が誤りとは判断しないが、対象列と集計を明記せずにCaFの測定として流用しない。取得した4ファイルでは`average_sequence`の定義が見つからず、その参照を含む経路の実行可能性は未確認である。コードの実行や論文の数値再現は行っていない。確認したsource hashと範囲は[実験ログ](../experiments/2026-10-12_pdca_8h.md)に記録した。

## 次のscore契約を凍結する前に必要な決定

選択のdigestはtarget ID列のhashと選択indexを結びつけるだけで、渡されたper-token scoreやlogitsの出所・正しさを認証しない。実行runnerの入力・model・score計算と結果manifestを別途確認する。

以下を結果を見る前に一つのversioned protocolへ記録する。現時点で選択済みの主scoreや許容差はない。

| 項目 | 凍結時に明記する内容 |
|---|---|
| score target | model response全体、answer field、または独立に定義した一つの検証可能claimのどれか。現在はclaim field・抽出境界とも未同定。 |
| provenance | 入力schema・prompt・model revision・tokenizer/chat-template・回答/claim作成法と、評価時に使った各artifact hash。hashは人間の意味確認や権利を証明しない。 |
| token boundary | chat prefixの作り方、target単独/連結時tokenization、特殊token、prefix offset、ターゲットtokenの予測logit位置。 |
| normalization | 生文字列をそのまま使うか、どの正規化を行うか。codepoint単位で固定し、前後比較で変更しない。 |
| termination | EOS/assistant terminatorを含むか、生成stop条件とteacher-forced列をどう対応させるか。現行helperは含めない。 |
| aggregation | sumか別の事前規定値か、token長の違うclaimをどう扱うか。主score、符号、失敗時規則を先に固定する。 |
| intervention/control | 同一claimの固定scoreを証拠chunk介入とmatched control介入で比較する方法。OOD・空間/時間範囲・cache位置を記録する。 |
| unit and inference | 独立単位はvideo。複数QA/claimを同一videoから得ても独立標本として数えず、動画単位の不確実性を使う。 |

scoreは「この条件でのモデルの文字列尤度」である。正答性、claimの真実性、動画中の視覚証拠との意味的一致、内部因果経路はそれだけでは測れない。claimに対応づけられた動画区間、人手で確認した意味、適切な対照がなければ、claim faithfulnessのprimary scoreとして使わない。

## 現在の停止条件

NOAAのpilotは3動画・6 QAであり、人間による独立attestationが未完了のためQA gateは`no_go`である。6 QAは独立動画6件を意味しない。この資料はgateを通過させず、権利・回答の人手確認・claimの時間対応を代替しない。人間レビューが完了しても、このpilotだけで独立確認や一般的なfaithfulness主張が成立するわけではない。関連する構造検査は[`faithfulness_protocol_contract.md`](faithfulness_protocol_contract.md)を参照。
