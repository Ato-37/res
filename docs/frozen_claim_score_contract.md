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

scorerは機械的な全target sumとmeanをnatsで返す。呼び出し側が単一EOS IDを与え、最後のtarget IDがそれと等しい場合だけ`terminal_seen`となる。全target scoreはそのterminalを含み、content scoreは末尾terminalを除く。EOS IDや終了設定の出所は検証せず、multi-EOS設定も扱わない。EOSがないtailは`termination_unverified`とされ、生成が自然に終了したと認証するものではない。

[`frozen_target_token_selection.py`](../scripts/frozen_target_token_selection.py)は同じ凍結targetに対して、呼び出し側が指定した0-based・target-relative indexだけを集計する。selection hashはtarget hashとindex列に結び付くが、意味上どのtokenがclaimに属するか、JSON wrapper/EOSを含むか、選択者の判断が正しいかは検証しない。`semantic_selection_verified=False`であり、caller provenanceも未検証のままである。

この二つのpure CPU helperについて、[scorerの13テスト](../scripts/test_frozen_teacher_forced_score.py)、[selectorの7テスト](../scripts/test_frozen_target_token_selection.py)が通った。検査範囲はimmutable ID tail、固定した予測offset、sum/mean、末尾EOSの算術、選択indexのhash・範囲・集計とmalformed入力の拒否である。モデル推論、実生成のprovenance、claim意味、annotation適合性はテストしていない。

現時点で**生成されたJSON claim用のschemaはない**。full assistant responseと一つのclaimは同じ対象ではなく、全responseのlogprobをclaim scoreと読み替えない。生成claimを採点するには、まずresponse形式とclaim抽出/JSON schemaを凍結し、raw generation IDsとscore対象indexの対応を保ったまま、annotation・時間区間との意味対応を別途確認する必要がある。文字列fieldをannotationとして直接採点する場合と、生成raw IDs内からclaim部分を選ぶ場合のprotocolを混ぜない。

`build_samples.py`のwhitespace/casefold正規化は、NExT-GQA answer文字列と選択肢の対応を作る用途である。score対象文字列の正規化ではない。新しいclaim scoreでUnicode正規化、strip、句読点除去、空白圧縮をするかは未決であり、正解ラベル照合の規則を流用しない。

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
