# Day1続き: 時間的groundingの欠落発見〜layer=6/head=16の発見〜Step2初回試作

前提: [2026-09-29_01_environment_and_diagnosis_pipeline.md](2026-09-29_01_environment_and_diagnosis_pipeline.md)の続き。これも2026-10-01に遡って記録。

## 時間的グラウンディングの欠落(ユーザー指摘による軌道修正)

ユーザーから「ちゃんと時間の接地(grounding)も行わせて根拠を説明させているのか」と指摘され、**それまでのhead ablation実験は正解テキストの確率しか見ておらず、動画のどの時間区間を根拠にしているか(grounding)は一切検証していなかった**ことが判明。確定方針のStep1(LocalizationHeads型: groundingに効くheadの特定)の本来の目的からズレていた重大なギャップだった。

対応として[grounding_check.py](../scripts/grounding_check.py)を新規作成。最後の入力位置から動画トークンへのattentionを全層・全head平均し、時間チャンク単位(Qwen2.5-VLの時間的パッチマージにより複数フレームが1チャンクに融合される)に集約、各チャンクのタイムスタンプを求めてNExT-GQAの正解根拠区間と重なるか(grounding hit)を判定する設計。`video_token_id`は`model.config.video_token_id`から取得。

### ベースライン結果(全layer・全head平均、16サンプル)

**grounding hit rate = 2/16 (12.5%)**。しかも16サンプル中14サンプルでpeak_chunkが常にchunk=0(動画冒頭)になっていた。Transformerで知られる"attention sink"現象(内容に関係なく系列先頭に強い注意を向けるheadが一部存在する)により、全head・全layerの単純平均ではsink的なheadが支配的になり、真にgroundingに効いている少数headの信号が埋もれていると解釈した。

候補層(23, 4, 9, 1, 8)だけに絞っても結果は変わらず(hit rate=2/16のまま)。つまり問題は「どのlayerを見るか」ではなく、layer内の**どのheadを見るか**という、さらに一段細かい粒度にあった。

## 個別head探索とheld-out検証(`head_grounding_scan.py`の初版)

784個(28層×28head)すべてについて、videoトークンへのattentionからpeak chunkを求め、正解根拠区間とのhit_rateを16サンプルでスキャン([head_grounding_scan.py](../scripts/head_grounding_scan.py)、既存のoutput_attentionsを再利用するだけで追加のモデル呼び出し不要)。

ランダム基準(16サンプル平均28.12%)との比較:
- 全head平均(素朴なベースライン): 12.5% — ランダムより悪い
- 上位4head(layer0/head0, layer4/head12, layer8/head24, layer1/head7): hit_rate 43.75% — ランダムより明確に高く見えた

**held-out検証で完全に再現せず**: 784headを16サンプルだけでスクリーニングするのは多重比較問題を抱えている。選定に一切使っていない新規10サンプル([data/nextgqa_samples_heldout.json](../data/nextgqa_samples_heldout.json))でこの4headを検証したところ、**4head全てhit_rate=30.00%、このheld-outセットのランダム基準42.50%を下回った**。最初に見えた「有望な4head」は16サンプルへの過学習(ノイズ)だった可能性が高い。

(2026-10-01追記: この時の43.75%/30.00%はいずれも、後日発見した固定チャンクbaseline(50.00%/60.00%)も下回っていたことが判明している。[2026-09-30_fixed_chunk_baseline_confound.md](2026-09-30_fixed_chunk_baseline_confound.md)参照)

## 大規模再スクリーニング(58サンプル screening + 22サンプル held-out)でlayer=6/head=16を発見

NExT-GQAテストセットからさらに未使用サンプルを大量抽出し、**screening 58サンプル・held-out 22サンプル**(動画の重複なし)で再スクリーニングした([data/nextgqa_samples_big.json](../data/nextgqa_samples_big.json)、[data/nextgqa_samples_big_heldout.json](../data/nextgqa_samples_big_heldout.json)の前身)。一部動画がOpenCVで読めない不良ファイルだったため、`head_grounding_scan.py`にサンプル単位のtry/exceptを追加して自動スキップするようにした。

### screening結果(58サンプル、ランダム基準34.48%)

上位候補がlayer 18〜20付近に集中(layer19/head17, layer19/head23, layer20/head27, layer18/head13, layer6/head16 等、hit_rate 43〜48%)。784回試行して最良値を取ると帰無仮説下でも期待される最大z値はおよそ3前後になるため、単体では確証にならない(今回の最良候補はz値にして約2.2程度)。

### held-out検証(22サンプル、ランダム基準36.36%)で明暗が分かれた

| head | screening(n=58) | held-out(n=22) | 判定 |
| --- | --- | --- | --- |
| **layer=6, head=16** | 44.83% | **45.45%** | **両方で一貫して基準を上回った** |
| layer=19, head=17 | 48.28% | 40.91% | 弱く再現 |
| layer=18, head=13 | 44.83% | 36.36% | 再現せず |
| layer=20, head=27 | 46.55% | 36.36% | 再現せず |
| layer=19, head=23 | 48.28%(screeningの最上位) | 31.82% | 完全に再現せず |

`layer=6, head=16`だけが独立な2つのサンプル集合の両方でランダム基準を一貫して上回った。**この時点でlayer=6/head=16を主要候補として確定**。

(2026-10-01追記・重要な訂正: この結論のうちheld-out側(45.45%)は、後日発見した固定チャンクbaseline50〜52%を下回ることが判明しており、位置バイアスと統計的に区別できない。screening側(44.83%)は固定チャンクbaseline37.70%を上回っており支持されたまま。詳細は[2026-09-30_fixed_chunk_baseline_confound.md](2026-09-30_fixed_chunk_baseline_confound.md)。**現時点で最も頑健な根拠はこのhit_rateではなく、2026-09-30のsteering統計検定([2026-09-30_layer6_head16_heldout_significance.md](2026-09-30_layer6_head16_heldout_significance.md))である**)

## Step2初回試作(`activation_steering_spike.py`)とその設計ミス

layer=6, head=16の出力をo_proj入力スライスの倍率(1,2,4,8倍)で増幅し、held-out 22サンプルで(a)正解トークンのlogprob (b)全head平均のgrounding_hit_rate がどう変化するか測定した([activation_steering_spike.py](../scripts/activation_steering_spike.py))。

**結果**: correct_logprobは1倍→8倍で単調に悪化(-22.37→-22.89)。grounding_hit_rateは全スケールで27.27%のまま完全に不変。

**設計ミスの発覚**: grounding_hit_rateが一切反応しなかったのは、steering hookが`o_proj`入力(=そのheadのvalue出力)をスケールしているのに対し、grounding_hit_rateは**attention weights(softmax前後の「どこを見るか」)だけ**から計算しているため。value側をいくら増幅してもattention weights自体は変化せず、この指標が反応しないのは当然の帰結だった。**介入対象(value)と評価指標(attention weights)が噛み合っていなかった**、というスクリプト設計上の不整合であり、「steeringが無効だった」という結論ではない。

## スケールスイープでsweet spotを発見

scale=0〜2.0まで細かく振ったところ、山型のパターンが見つかった(held-out 22サンプル、correct_logprob):

| scale | 0.0 | 0.25 | 0.5 | 1.0(素の状態) | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| correct_logprob | -22.26 | **-22.19(最良)** | -22.27 | -22.37 | -22.38 | -22.47 |

完全にablationする(0倍)よりも4分の1程度に弱める(0.25倍)方が良く、増幅する(>1倍)ほど悪化する、という山型の反応。

## コントロールheadとの比較で解釈が覆る

head ablation実験でavg_correct_drop≈0(無風)だった`layer=12`から、head番号を揃えて`layer=12, head=16`をコントロールとして同じスケールスイープを実行。

| scale | layer=6/head=16(候補) | layer=12/head=16(コントロール) |
| --- | --- | --- |
| 0.0 | -22.26 | -22.55 |
| 0.25 | -22.19(最良) | -22.58(最悪) |
| 2.0 | -22.47 | -21.88(最良) |

**結果は完全に逆方向だった**: 候補headは弱めるほど改善する山型、コントロールheadは強めるほど単調に改善、しかも効果量(0.49 nats)は候補headの最良効果(0.18 nats)の約2.7倍とコントロールの方が大きかった。

**正直な棚卸し**: 単一のコントロールheadとの比較だけでは「本当にこのheadが特別か」を確立するには不十分。多数のheadで同じスケールスイープを行い、layer6/head16の応答パターンが分布の中で本当に外れ値かを確認する必要がある、という結論に至った。

## 全28層でのスケールスイープ(n=8)- 較正された結論

head番号を16に固定して全28層を8サンプル・3スケール点(0.25/1.0/2.0)でスイープした([steering_layer_survey.py](../scripts/steering_layer_survey.py)、672 forward pass)。

**重要な訂正**: `layer=6`は28層中**5位**(上位18%程度)で、平均よりは明確に「弱めると改善」の方向が強いが、唯一の外れ値ではない(layer=20,13,4,24も同等かそれ以上)。一方、以前コントロールとして選んだ`layer=12`は28層中**27位**、つまり正反対方向のほぼ最極端値だった。**つまり最初のlayer6 vs layer12の劇的なコントラストは、layer12が不運にも対極の極端例だったために誇張されていた**もので、layer6自体は「そこそこ上位だが唯一無二ではない」というのが較正後の結論。

Day1終了時点の位置づけ: layer=6/head=16は、(a) attentionベースのgrounding hit_rateで2つの独立サンプル集合の両方でランダム基準を上回り、(b) スケールスイープでも28層中5位という平均以上の傾向を示した、という2種類の独立した**弱い**証拠を持つ候補。「唯一の正解head」ではなく「現時点で最も複数の弱い証拠が重なっている候補」という位置づけでDay2(2026-09-30)に持ち越した。

## Day2への引き継ぎ

Day2(2026-09-30)では、このn=8スイープをn=30に拡張(layer=6が1位に浮上)→held-outでの統計的検定(p<0.0001)→教師なし head選定の実装→固定チャンクbaselineの発見、という流れに進んだ。詳細は[2026-09-30_layer6_head16_heldout_significance.md](2026-09-30_layer6_head16_heldout_significance.md)以降の3ファイルを参照。
