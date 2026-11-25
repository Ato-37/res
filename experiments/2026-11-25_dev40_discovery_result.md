# dev40 の取り直し (discovery, 2026-11-20〜21) の結果

状態: 完了。dev のみ (test は未使用)。探索的。基準は未凍結。

## 目的
計画 (`experiments/2026-11-18_new_research_plan_map_the_flow.md`) の段階1。窓幅 k の候補 {1,3,5,7,9,13} を、dev の各タスク先頭40本で、現行 null と並べて測る。あわせて、ローカル (5070 Ti) の結果と比べる (引き継ぎ §2 の同等性チェック)。

## 設定
- サーバー: discovery (RTX 3090, ドライバ 580.126.09)。venv310 (Python 3.10.21, torch 2.2.1+cu118, transformers 4.40.0)。res @ c0f08f8。
- `scripts/mtf_v2/queue_dev40_server.sh`: `flow_v2.py --part dev --n_per_task 40 --ks 1,3,5,7,9,13 --null --fp32_readout`。seed 0。eager attention, mask_num_heads=1。
- 実行: 4タスク 2026-11-20 21:27 → 11-21 05:39 (160本)、Action Antonym 05:39 → 07:39 (40本)。エラー 0。1本あたり 約178 s。
- 出力: `results/mtf_v2_discovery/` (jsonl.gz と集計 json, SHA256SUMS)。集計: `scripts/mtf_v2/analyze_dev40.py` (動画単位ブートストラップ 2000回, seed 0, 層帯の平均; real−null に BH-FDR、全 120 セル)。ブートストラップ p の下限は 1/2000 のため、q の最小値は 0.005。
- 比較: `scripts/mtf_v2/compare_local_server.py` (ローカル dev20 + dev20b と、共通の動画・k・層)。

## 結果
### ベースライン精度 (dev 40本, 正答選択なし)
| タスク | 正答 | 偶然水準 | 片側 二項 p |
|---|---|---|---|
| Action Antonym | 20/40 (0.50) | 0.50 | 0.56 |
| Action Sequence | 24/40 (0.60) | 0.50 | 0.13 |
| Moving Direction | 12/40 (0.30) | 0.25 | 0.29 |
| Object Count | 21/40 (0.53) | 0.25 | 0.0002 |
| Scene Transition | 23/40 (0.57) | 0.50 | 0.22 |

偶然水準を有意に上回るのは Object Count だけ。n=40 のため検出力は低い (例: 真の精度 0.65 でも 2択で有意になりにくい)。

### real − null (Δmargin, nat; 平均 [95% CI], q は BH)
- Scene Transition: k≥3 の L0-5・L6-15 で一貫して負、q=0.005〜0.04 (例 k=13 L6-15: −0.27 [−0.40, −0.14])。
- Action Antonym: L6-15 で負 (k=5: −0.14 [−0.27, −0.01], k=13: −0.27 [−0.55, −0.01])。CI は 0 をわずかに外れるが、FDR 後は q ≥ 0.12 で有意でない。
- Object Count: k=1 L6-15 で負 (−0.10, q=0.005)。k≥7 の L6-15 では逆に正 (k=13: +0.46 [+0.10, +0.80], q=0.08)。null の方が real より大きく margin を下げる。
- Action Sequence: L6-15 で正 (null の方が下げる) だが CI は 0 を含む。L26-31 の小さな正の差 (+0.005〜0.008) が q<0.05。
- Moving Direction: k=5 L0-5 で +0.05 (q=0.049) 以外は 0 付近。
- L16 以降は、全タスクで real・null とも ≈0。

### ローカルとの比較 (共通 156 本)
- 予測の一致 156/156。base_margin の差 中央値 0.030, 最大 0.18 nat。
- real の Δmargin: r=0.999, |差| 中央値 0.016, 95%点 0.084。k=13 L6-15 のタスク平均は 0.01 nat 以内で一致。
- null の Δmargin: r=0.976, |差| 95%点 0.41。ランダムに選ぶセルが torch 版で変わるためと考えられる (未検証)。

## 解釈 (この dev 40本・このモデル・この指標の範囲)
- フレーム間注意の遮断が、null より明確に margin を下げるのは Scene Transition だけ。Action Antonym は同じ向きの傾向があるが、多重比較の後は有意でない。
- Object Count と Action Sequence では、現行 null (vision×vision の下三角から同数のセルをランダムに遮断) の方が real より効く。現行 null は対照として強すぎる可能性があり、real−null だけでは解釈できない (dev20 で見た傾向が n=40 でも再現)。→ null の作り直し (フレーム位置だけをずらす null) が必要。
- ベースラインが偶然水準付近のタスク (Moving Direction は計画の除外ルールに該当。Action Antonym・Action Sequence・Scene Transition も偶然水準と区別できない) では、margin の変化は「正しく解く経路」ではなく、選択肢の対数確率の差への影響として読むべき。
- 環境の違い (GPU 世代と torch などの版) による real の差は、k=13 L6-15 の効果の大きさに比べて小さい。ただし記述的な比較で、要因は切り分けていない。結果は合算しない。
- 限界: margin は gold と最良の誤答の対数確率の差で、注意の遮断で生じる分布外の入力を含む。層帯の平均は事後に決めた区切り。n=40/タスク。

## 次のアクション
1. Moving Direction を除外する (計画 §0 のルール)。Action Antonym も偶然水準のため、一次解析 (Action Antonym → 他4タスクの転用) の起点タスクにしてよいかを再検討する (ユーザー判断)。
2. フレーム位置シャッフル null を実装し、dev で再実行する (k は 5, 9, 13 に絞る)。
3. 正答した問題だけでの解析を、副次的に追加する (n が減る点に注意)。
