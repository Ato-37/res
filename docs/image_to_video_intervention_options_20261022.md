# 画像VLMの内部介入を動画へ移す候補（2026-10-22）

## 文献で報告された機序

- **VISTA** は画像LVLMの生成中に、視覚入力あり／なしの文脈から得た残差ストリーム差分を例ごとの visual steering vector とし、生成時の残差へ加える。論文は別に、最終層より前の複数層のlogitsを平均して最終logitsと混ぜる Self-Logits Augmentation も提案する。評価は画像LVLMであり、係数は別の100画像holdoutで調整されている。したがってQwen2.5-VL動画への効果や、係数の移植可能性は未確認である。[arXiv:2502.03628](https://arxiv.org/abs/2502.03628)
- **HiPrune** は視覚encoderの中間層attentionから物体寄りのanchor tokenを選び、隣接buffer tokenで局所関係を保ち、深層attentionからglobal情報を含むregister tokenを補って、言語モデルへ渡すtokenを削減する。画像実験にはQwen2.5-VL-3B/7B/32Bが含まれる。動画ではVideo-LLaVA-7Bを評価し、論文本文・付録の報告ではVideo-LLaVAのVinoGround性能は視覚tokenを1/16にしても99.2%を維持、MVBenchは低下がやや観察された。これはtoken効率とQA性能の測定であり、説明の忠実性検証ではない。[arXiv:2508.00553](https://arxiv.org/abs/2508.00553)
- **VCD** は元画像と視覚情報を劣化させた画像の出力分布を対比するdecoding法で、言語priorや統計的な物体共起への依存を抑える狙いを持つ。一次論文は画像LVLMでの評価で、動画適用は今後の課題としている。これは内部activation介入ではなく、入力と出力分布を使う対照法である。[arXiv:2311.16922](https://arxiv.org/abs/2311.16922)

## 本研究の既測定値

Minerva-Egoの固定されたデータセット作成 rationale を教師強制評価した3動画の探索的pilotでは、`+ln(2)` visual-key biasのbbox条件は、baselineに対して全行でlog-probabilityが低く、同じdoseの同時刻spatial-shamに対しても全行で低かった。bbox − same-pair spatial-shamの平均差は **−0.019516 nat/token**。このpilotは生成説明を測っておらず、3動画・固定rationaleの予測依存proxyであり、faithfulnessの証拠ではない。先行の1動画pilotは後の3動画pilotの1行（同じ動画）と重複するため、独立した追加標本として数えない。結論はこのbbox bias設定が狙った効果を示さなかった、に限定する。[実験記録](../experiments/2026-10-22_minerva_bbox_spatial_internal_bias_gpu_pilot.md)

## 次に試す単一の探索prototype（提案）

VISTAの**residual-stream steeringのみ**を、既に凍結済みの新規候補動画へ適用する。SLA、layer/head探索、bias強度探索は行わない。まず全予定行で公式time-map/media CPU preflightを通し、Qwenの時間パッチが隣接2フレームを結合する点を、対象anchorと対照の両方で同じにする。

各行について、同じ64フレーム・同じprompt長を保ち、元入力の最終prompt位置の残差と、対象anchorを含む**2フレーム対**の画素を平均色で置き換えた入力の残差との差を、固定した第14 decoder block（0始まりのindex 13、28 blockの中央）で取る。これをanchor寄与ベクトルとする。対照は、対象外の別の2フレーム対を同じ方式で置き換えて得るtemporal-sham寄与ベクトルとする。両ベクトルをFP32で計算し、各々のRMSをその行の元入力・同位置・同block残差RMSの10%へ正規化してから、元入力のprompt最終位置に加算する。ゼロ／非有限RMSなら行を失敗として残す。更新後のRMS、実効変位、hook回数を記録する。この10%は数値安定性のための事前固定量で、データによる最適値とは主張しない。VISTAの「視覚情報あり／なしの残差差分」という発想を、動画内の局所的な有無の比較に変える**本研究の適応案**であり、論文の手法そのものではない。baseline／anchor steering／temporal-sham steeringの3条件で同じ固定系列を一度ずつ教師強制評価する。注入位置・係数・対照対の選び方をGPU結果前に固定し、layer・head・強度探索は行わない。平均色によるOOD入力は限界として残る。

**事前固定する主endpoint**は、介入前にgreedy生成した文に含まれるobject-name tokenを固定し、anchor steeringによるそのtokenの平均log-probability変化から、temporal-sham steeringによる同変化を引いた行内差と、その3行の平均である。教師強制では対象tokenをpromptへ連結し、token `j` の予測位置を `prompt_len+j−1` とする。注入は第14 blockのprompt最終位置だけとし、後続blockの因果attentionを通じて後続の対象tokenにも伝播し得る。介入前の文が事前の一文・対象名条件を満たさなければ、当該行は事前定義した生成形式の失敗として分母に残し、代替文を選ばない。全行と失敗を報告し、3行では記述的な探索値として扱い、有意差や一般化を主張しない。これは内部表現に対する予測依存・感度の測定にとどまり、生成内容の真偽や説明faithfulnessを証明しない。

**Faithfulnessを述べる前提:** 現在の日本語レビュー画面で、条件名を隠した生成文の視覚的支持・反証・判定不能を少人数でも記録する。これは簡易確認であり、回答者の独立性を形式的に保証する手続きは求めない。ラベルと介入効果の対応を示せるまでは、残差実験も予測依存proxyとしてのみ報告する。このレビューはGPUでの探索実験を止める条件ではなく、faithfulnessの結論を制限する条件である。

このprototypeはVISTAの画像用手法を動画用に変える**提案**であり、論文の画像実験やHiPruneのVideo-LLaVA結果はQwen2.5-VLのfaithfulness効果を保証しない。権利状況は未確認であり、保護データ・派生artifactはローカルに留める。
