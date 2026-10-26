# 卒研テーマ案：動画VLMの内部介入は根拠依存性を高めるか

## 研究の芯

**題目案:** 「動画VLMの内部視覚介入におけるスコア・出力乖離――NExT-GQA実動画でのmatched-control評価」

動画VLMの内部attentionへ介入すると尤度や回答marginは動く。しかし、その変化だけで「説明が動画根拠により忠実になった」と言えるのか。本研究では、Qwen2.5-VL-7B-Instructを実動画に適用し、介入・baseline・同じ予算のshamを比較する。説明尤度、選択肢margin、離散回答、時間区間を別々に測り、内部スコアの変化が外部の説明・予測挙動と一致する条件を調べる。

## すでに得られている実動画の結果

| 実験 | 観測値 | 言えること |
|---|---|---|
| QTAによるQA時間区間生成（3動画） | baseline・QTA・shamで出力区間が3/3行一致。平均tIoUは全条件0.08300、行ごとの差は全て0。一方、共通生成prefixのtoken尤度はQTA−baselineで各動画+0.00583、+0.01054、+0.02475 nat/token。 | 内部の数値変化が時間出力の改善に結び付かない例。差はpost-hoc、n=3の既知cohortで探索的。 |
| 五択QAへのQTA対matched-sham（27動画） | 各条件15/27正解、全27行で予測選択不変、wrong↔correctは0。gold-marginの最大絶対変化はQTA・shamとも0.375 nat。 | 数値の変化と正答行動が乖離し、shamでもscoreが動く。正答率改善は観測されていない。 |
| claim尤度×QA区間mask（3動画） | QTA−baselineの選択的依存差は平均−0.01209 nat/token（95%探索的t区間[−0.12288,+0.09870]）。QTA−shamは+0.01208（[−0.06674,+0.09090]）。 | 固定条件ではclaim尤度proxyの安定した改善を示さない。QA時刻はclaim固有supportではない。 |
| QA注釈時間へのinternal key bias（validation cohort 1、20動画） | annotation-bias−同数disjoint shamのgold-minus-fixed-foil差は平均+0.01250 nat、中央値≈0、範囲[−0.125,+0.250]、探索的t区間[−0.04085,+0.06585]。正答はbaseline/sham/biasで15/20・15/20・14/20。 | 小さな平均margin差は正答改善にならなかった。行別方向は8正/7負/5≈0。 |
| 同一プロトコルの独立SHA順位cohort 2（20動画） | bias−sham margin平均+0.02500 nat、中央値≈0、範囲[−0.375,+0.375]、探索的t区間[−0.04752,+0.09752]。正答は全条件11/20、予測変更0件。方向は8正/7負/5≈0（1e-7許容幅）。 | 1回目と同じく中央値はほぼ0で、正答改善なし。コホートをまたぐ安定効果や母集団効果は主張できない。説明claimへのfaithfulnessではない。 |
| 事前登録head介入確認（79件） | matched-control差−0.0756 nat（95% t CI [−0.1578,+0.0065]）。事前成功基準+0.25 nat未達。 | この固定head estimandの改善仮説は支持されず、head/layer/alpha探索は再開しない。 |
| fresh-val QA区間claim-mask bias（逐次cohort n=3/6/7、計16動画） | primary `ΔF` cohort平均は+0.01817、−0.00355、−0.00368 nat/token。16件の単純記述値は平均+0.00046、中央値+0.00475、範囲[−0.05460,+0.06046]、正方向8/16。QA−sham平均は+0.03444、−0.00878、+0.01474。全96/96 forwards。 | 逐次探索cohortであり、正式な統合推定や母集団推論はしない。効果方向は安定せず、QA時間区間はclaim支持の人手正解ではない。faithfulness改善を示さない。 |

これらは異なるcohort・estimand間で統合した効果量ではない。小規模・post-hocの結果に母集団推論は付けない。動画説明のclaim-supportに対する直接faithfulness評価は**0/25 formal review**であり、直接改善を実証したとは言えない。人手未レビューのQA intervalを直接faithfulnessの代用にしない。

## 卒論としての貢献候補

1. attention/尤度が動いたことと、予測・説明挙動が変わったことを区別するmatched-sham評価を実動画で実装する。
2. 一般steeringは内部scoreを変えても、選択的な根拠依存改善を保証しないという反証可能な限界を示す。
3. 成功条件をclaim単位のsupport-key対matched-shamと固定し、直接得点が得られない場合も、現行proxyでは改善が確認できなかったという再現可能な探索結果を報告する。

このテーマは「内部介入がfaithfulnessを改善した」という成功談ではなく、**どの測定値が改善を示し、どの測定値では示さないか**を検証する経験研究として成立させる。指導教員には、負・不確定結果を含むmeasurement-validation/negative-result型の卒論として採用可能か相談する。

## 今回のGPU結果を反映した、より具体的な卒論案

**研究質問:** 「動画VLMの内部visual-key steeringでgold-answer尤度が上がったとき、回答margin・正答率・説明の動画根拠依存も改善するか。改善量は同予算shamを超えるか。」

同一手順で固定した独立validation cohortを2組（各20動画）評価した。1組目のQA注釈時間key bias対matched shamの固定gold-minus-foil margin差は平均+0.0125 nat、中央値ほぼ0、探索的95% t区間[−0.0408,+0.0658]。2組目は平均+0.0250 nat、中央値ほぼ0、区間[−0.0475,+0.0975]。いずれも区間は0を含み、行別差は正8・ほぼ0が5・負7。正答率は1組目がbaseline/sham/biasで15/20・15/20・14/20、2組目が全条件11/20で予測変更なし。availability条件付きのSHA順位cohortなので、これらの記述区間は母集団推論ではない。コホートをプールした効果量も計算しない。

この実験だけから説明faithfulnessの成功/失敗を断定しない。卒論の貢献は、(a) 実動画40本を独立した2 cohortに分け、attention intervention・baseline・同key数disjoint shamを比較可能にしたこと、(b) 介入maskを全層で直接監査したこと、(c) 内部margin proxyの小さな差がいずれのcohortでも正答改善にならなかったこと、(d) 既存の区間生成・claim尤度・QA診断と合わせ、proxyごとの不一致を定量化したことに置く。直接faithfulnessを問う最終章では、説明claimに紐づく人手確認support spanがない限り「直接効果未測定」と結論し、QA注釈時間をその代わりにしない。

この計画なら肯定的効果が出なくても、**実動画における内部介入のmatched-control実装と、スコア変化が外的な正答・区間・根拠依存改善を保証しない条件の検証**として研究の形を保てる。結果の詳細・再現コマンド・限界は[第1コホート実験ログ](../experiments/2026-10-21_annotation_key_bias_val_20_gpu.md)と[独立第2コホート再現ログ](../experiments/2026-10-21_annotation_key_bias_val_replication20_gpu.md)を参照。

## 次に直接改善を示すための判定基準

新たな肯定主張には、固定した説明claimについて、動画上の支持区間を独立に決め、その同じ区間のvisual keysへ介入した条件が、baselineおよび同じkey数・重なりを持つshamを上回る必要がある。primaryは、固定claimのfull-video対support-interval ablation尤度差が介入によりsham以上に増えるかとする。QA区間や自動選択windowを意味的なsupport正解として代用しない。現在はレビューとclock/frame対応が未完了のため、この直接試験のスコアはまだない。

## 再確認元

- [内部介入の実験統合ログ](../experiments/2026-10-21_thesis_progress_internal_intervention_synthesis.md)
- [区間生成のGPU実験](../experiments/2026-10-21_qta_intervened_qa_interval_generation.md)
- [27動画の正答診断](../experiments/2026-10-21_qta_full_video_accuracy_diagnostic.md)
- [claim尤度とQA区間の交互作用](../experiments/2026-10-21_claim_text_qa_interval_key_bias.md)
- [head確認試験の再解析](head_faithfulness_confirmation_results.md)
- [validation 20動画のQA annotation-key GPU実験](../experiments/2026-10-21_annotation_key_bias_val_20_gpu.md)
- [独立validation 20動画での固定プロトコル再現](../experiments/2026-10-21_annotation_key_bias_val_replication20_gpu.md)
