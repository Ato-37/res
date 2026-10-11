# Head-faithfulness independent confirmation: result and decision

更新日: 2026-10-08

## 結論

事前登録したhead選択的faithfulness仮説は、このモデル・データ・介入設定では支持されなかった。独立確認79動画は全件完了したが、主効果の点推定は負で、95%信頼区間の上端も事前に定めた実質差`+0.25 nat`を大きく下回った。したがって、同じデータでhead/layerや介入量を変えて有意差を探す追加GPU探索は行わない。

これは「どのモデルにもfaithfulnessがない」という証明ではない。今回固定した説明器・内部介入・選択肢margin・データセットの組み合わせについての、事前基準を満たさなかった結果である。

## 何を測ったか

各動画で、介入前にモデルが最も高く評価した選択肢を`ŷ`、その次点をfoilとし、その差をmarginとした。選択headはmarginに対するhead出力のgradient×activation attribution上位4個。選択headの出力を`alpha=0.5`に減衰し、減衰前後のmargin低下を測った。同じ層ごとのhead数、head数、活性値規模を対応させた5個のcontrol groupにも同じ介入を行った。

動画`i`ごとの主評価値は

`d_i = target margin drop_i - mean(control margin drop_i)`

であり、`theta`は動画79件における`d_i`の平均である。正の値なら選択head介入の方がmatched controlよりmarginを大きく下げたことを表す。成功基準は両側95% t信頼区間の下限が`+0.25 nat`を超えることだった。これは「説明全般の忠実性」ではなく、この選択head介入に関する限定的な因果診断値である。

## 実験結果

- 公開commit: independent confirmation `a80ec61`、smoke `ec08f4f`。
- independent test: 要求79、完了79、失敗0、sample key 79種類。run manifestの設定・件数は結果JSONと一致し、sample manifest SHA-256は`2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`。
- 設定: Qwen2.5-VL-7B-Instruct、4-bit、全28層からtop-4、`alpha=0.5`、5 matched controls、16 frames、`cap_pixels_per_frame=true`、`max_pixels=200704`、seed `20261010`、RTX 5070 Ti。
- 選択headの平均margin低下: `−0.0601 nat`。
- controlの平均margin低下: `+0.0155 nat`。
- 主効果`theta`: `−0.0756 nat` (SD `0.3667`)。
- 95% t信頼区間: `[−0.1578, +0.0065] nat`。0を含み、上限も`+0.25`未満。
- `theta=0`に対する記述的両側t検定: `p=0.0706`。0と異なる効果を5%水準で示していない。
- 事前実質境界`theta=+0.25`に対する両側検定: `p=1.52×10⁻¹¹`。これは正のfaithfulnessを示すp値ではない。観測値が`+0.25`より大幅に小さいことと整合し、成功基準を満たさない。
- sample-level contrastは正30、負43、0が6件。baseline選択肢正解は56/79 (70.9%、記述値)。主解析には正誤にかかわらず全件を含めた。
- 独立な再計算スクリプトは選択head/controlの層構成・重複、各recordのmargin差、要約統計、run manifestを検査する。

## 解釈と限界

この結果は、今回のgradient×activation上位4 headが、head数・層構成・活性規模を対応させたcontrolよりも予測を特異的に支えている、という実質的な効果を示さなかった。推定値が負であるため、少なくとも今回の介入定義では、選択headを弱めることがcontrolより強いmargin低下を生む方向ではなかった。

信頼区間は0を含むので厳密なゼロ効果を確定しない。ただし上端`+0.0065`は事前の実質境界`+0.25`よりかなり小さいため、登録した大きさの効果はこの確認データで支持されない。選択headの再選定、正解サンプルだけの後付け分析、別alphaへの変更でこの結果を救済しない。

対象はQwen2.5-VL-7B-Instructの4-bit実行、NExT-GQAから選んだ79動画、16-frame処理、4 headの半減衰、現在のgradient×activation順位付けとmatched-control設計に限られる。また5択log-probability marginは自由生成回答の正確さそのものではなく、head介入も自然な説明文の真偽を直接測らない。したがって「動画VLMの説明は一般に忠実／不忠実」と一般化しない。

## 次の方針

1. この結果を負の確認結果として保存し、このhead-faithfulness primaryを閉じる。追加のlayer/head/alpha探索はしない。
2. 卒論では、問い・estimand・事前成功境界・失敗した結果・限界を明記し、成功したかのように書かない。
3. 研究を続ける場合は、現データの部分群や別指標から始めず、説明対象と独立に正当化できる介入対象・対照・評価単位をCPU上で新しい研究計画として定義する。新しいGPU実験は、その計画が固定され、必要性が確認された後に限る。

## 再計算

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/analyze_head_faithfulness_confirmation.py \
  --input results/head_faithfulness_confirm_20261009.json \
  --manifest results/head_faithfulness_confirm_20261009_run_manifest.json
```

再計算はCPU処理であり、モデル推論や動画アクセスは行わない。
