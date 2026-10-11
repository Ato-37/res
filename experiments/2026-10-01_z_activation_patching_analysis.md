# Layer-level patching: exploratory recovery and analysis repair

## 目的・設定

単一headに限定せず、full入力のlayer attention出力全体をframe置換入力へ移植して
回復を測定する。16フレーム、layers=0,6,12,24,27、回答sequence logprob。
このログはユーザーが共有したremote desktop出力に基づく。元JSONはMacにはない。
対象: `results/activation_patching_confirm_heldout.json`。

## 結果

25件中21件処理、swap_drop<0は11件。4件の構成失敗はID=10,15,21,23。

| layer | 全21件 mean rescue | 悪化11件 mean rescue | 悪化11件の正方向割合 |
| --- | ---: | ---: | ---: |
| 0 | -0.158479 | +0.011481 | 7/11 |
| 6 | -0.008522 | +0.040854 | 7/11 |
| 12 | -0.170259 | -0.107408 | 5/11 |
| 24 | +0.020230 | +0.203490 | 7/11 |
| 27 | +0.069456 | +0.470246 | 9/11 |

layer27/sample9のrescue=+3.3397は悪化群rescue合計の約65%。この1件を除く平均は
約+0.1833（表示済み丸め値からの計算）。旧mean_rescue_fraction=-0.308858は
サンプルごとの小さい分母に強く影響され、mean rescueの符号と一致する必要はない。

## 実装改善・検証

`analyze_activation_patching.py`を更新:

- 全件と悪化群を保持。平均・中央値・正方向割合・bootstrap 95% CIを一度に出力。
- 回復率は悪化群の `sum(patch_rescue)/sum(full-swapped)` を使用し、同じサンプルを
  分子分母で再標本化してCIを計算。個別比率平均と同じ推定対象ではなく、低下量で
  重み付けされた集約回復率であり、極端値への依存がなくなるわけではない。
- 最大絶対rescueのID、その1件を除く平均、leave-one-out平均範囲を併記。
- 保存済みtimestampとgroundingから、根拠外フレームをevidence扱いしたIDを監査。
- 空集合・1件・非有限値・重複ID・不完全な層情報を扱い、JSONに設定と結果を保存。
- 標準ライブラリだけのCPU回帰テストを追加。新しいGPU推論は行っていない。

## 解釈と以前の説明の訂正

1. layer27は探索候補。悪化群は結果を見て選んだsubsetで、5層からの選択もある。
   CIは点ごとの探索的区間で、多重比較やsubset選択を補正した有意性判定ではない。
2. full入力由来activationの移植は情報経路の診断。通常入力で使える補正手法や
   QA精度・忠実性の改善が実証されたわけではない。現在は回答prefixを含む全位置を
   patchするため、動画token経路に局在した効果とは言えない。
3. 実フレームであっても複製・置換後の動画がin-distributionとは限らない。
4. nearest-frame fallbackは根拠区間外を選びうる。以前の「全て根拠区間mask/swap」
   という解釈は強すぎる。今回の監査で実数を確認する。
5. ID10/15は根拠外フレーム数不足。ID21/23は総数不足だけではなく、連続した
   非根拠windowを要求する制約により失敗している。21件は選択された評価集合。
6. 過去の非有意結果は「効果がない」の証明ではない。位置ごとのmatched control
   scoreも代替回答全体の生成確率ではないので、忠実性の直接測定と同一視しない。

## 次のアクション

1. 既存JSONを新分析スクリプトで解析し、CI・外れ値依存・根拠区間coverageを確認。
2. 次のGPU実験前に、同入力self-patchの一致とfull/swappedのtoken配置一致を検証し、
   patch位置（prompt/回答位置）を分ける。根拠外fallbackを明示的な別群として扱う。
3. layer27やhead群の探索はこの集合上では探索的。新規データで検証する前に、
   control patch・評価指標・対象群を固定する。条件を変えて有意になるまで追試しない。
