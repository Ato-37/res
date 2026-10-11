# Soft ablation中心の方針更新と実装

## 目的

完全なhead zero化(hard ablation)による内部表現の崩壊を主分析にしないため、
対象attention headの出力を連続的に減衰させるsoft ablationを実装する。新しい
研究方針では、特定headの発見・精度改善を主目的にせず、attention相関と内部
介入が忠実性についてどのように異なる情報を示すかを検証する。

## 実装

- スクリプト: `scripts/soft_ablation_sweep.py`
- 対象: `layer=6/head=16`(過去のscreening候補)と`layer=12/head=16`(control)を既定値とする
- 減衰率: `alpha=1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0`
- 指標: 正解・top・control tokenのlogprob変化、entropy変化
- 再現性: モデル、量子化、dtype、attention実装、フレーム数、実効サンプル数、
  失敗サンプルID、サンプル単位の記録をJSONに保存可能
- 数値設定: Qwen2.5-VL-7Bは`bfloat16`、診断と同様に`eager` attentionを使用

## 実験結果

VP6F動画の終端付近でOpenCVのフレーム取得が失敗したため、フレーム数の
メタデータを信用せず、実際に逐次デコードできるフレーム数を数えてから
均等サンプリングする2パス方式へ修正した。修正後、n=2のsmoke testは
失敗なしで完了した。

held-out 25サンプルでsequence logprobを評価した。layer=6のmean correct
deltaはscale=0.05で`+0.1539`、scale=0で`+0.1541`、layer=12ではscale=0で
`-0.5532`だった。

ただし、position-matched controlとの差(selective delta)はlayer=6の全scaleで
95% CIが0を含み、Holm補正後も非有意だった。layer=12の未補正の有意傾向も、
7scaleのHolm補正後はscale=0で`0.5186`、0.01で`0.5337`、0.05で`0.5253`
となり、非有意だった。

layer=6とlayer=12のpaired layer contrastも、最大の未補正結果はscale=0.75の
`p=0.0278`だったが、Holm補正後は`holm_p=0.1949`であり、確証には至らなかった。

### 独立サンプルによる追試

候補選択に使っていない新規動画・QAを25件作成し、同じ設定で追試した。
layer=6のselective deltaはscale=0.05で`+0.2059`が最大だったが、Holm補正後は
`holm_p=0.7130`だった。layer=12も全scaleで非有意だった。
layer=6 - layer=12のpaired contrastも全scaleで非有意で、scale=0.05の
`holm_p=0.4824`が最小だった。

したがって、既存held-out 25件と独立追試25件の両方で、soft ablationによる
head特異的なselective effectは確認できなかった。

### 根拠区間maskによるnecessity予備検証

独立25サンプルに対して、16フレーム入力のgrounding区間を黒塗りする
`evidence_mask`と、同数の非根拠フレームを黒塗りする`control_mask`を比較した。
grounding区間が広くcontrol windowを作れない4件を除き、21件を分析した。

`necessity_contrast = evidence_mask_logprob - control_mask_logprob` と定義すると、
mean=`+0.061884`、sd=`0.682813`、t=`+0.415`、p=`0.682325`、95% CI=
`[-0.248934, +0.372702]`だった。負の値が根拠区間maskによる追加的な悪化を
示すが、平均はむしろ正方向で、necessity効果は支持されなかった。

なお、黒塗りはOOD介入であり、今回の結果は「根拠区間の必要性がない」という
一般的主張ではなく、このモデル・入力・mask方式・21サンプルに限定した陰性結果
として扱う。4件の失敗も、広いgrounding区間では同数の非根拠controlを構成できない
という評価設計上の制約として記録する。

## 解釈

soft ablationは完全zero化による情報損失を緩和する可能性があるが、内部表現の
分布外化を完全には防がない。したがって、単一のalphaで結論を出さず、alphaに
対する用量反応曲線、control token、entropy、control layer/headを併記する。

今回の結果から、layer=6/head=16を弱めると正解sequence logprobが改善する
傾向は見られたが、matched controlとの差としては再現されなかった。したがって、
現時点でこのheadをgrounding head、または忠実性に特異的なheadと結論してはならない。
独立サンプルでも同じ結論になったため、今後はこのheadの追加探索やscaleの
後付け選択を主分析にせず、根拠区間maskによるnecessity検証へ進む。

## 次のアクション

1. 既存サンプルのsoft ablation再実行は行わず、今回の結果を探索的な陰性結果
   として固定する。
2. attention hit rateのfixed-chunk baseline比較を補助分析として併記する。
3. necessity maskの結果を陰性の予備検証として固定し、追加のhead探索ではなく、
   より自然な時系列摂動（区間入れ替え・時間反転など）を含む検証設計を検討する。

## remote desktop / GitHub運用

実験はGitHubから取得したcloneをremote desktop上で実行する前提に整理した。
対象リポジトリは`https://github.com/Ato-37/res`である。
実行手順、CUDA別のTorch導入、動画をリポジトリ外に置く方法、smoke testの
コマンドは[docs/remote_run.md](../docs/remote_run.md)に記載した。コードと実験
ログはGitHubへ戻すが、生動画・zip・モデルキャッシュ・remote固有の絶対パスは
戻さない。`results/`は`.gitignore`で除外する。
