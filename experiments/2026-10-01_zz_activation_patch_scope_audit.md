# Activation patch位置と同入力対照の監査

## 目的・今回の結果解釈

ユーザー提供のanalysis_v2: 有効21/25、悪化群11。
swap_drop平均 -0.041385、95% CI [-0.496349,+0.377272]。
layer27の悪化群回復率0.728437のCIは[-0.024592,1.079190]。
全件回復平均0.069456はsample9を除くと-0.094058。
したがって有望な探索候補ではあるが、回復経路の確立とは言えない。
layer12全件の負のCIも点ごとの探索結果で多重比較未補正。
sample6は根拠区間外fallback。既存結果を黙って書き換えない。

## 実装・設定

activation_patching_sweepに追加:

- --patch-scope all/prompt/answer（既定all、1実行1scope）。
  promptは動画・質問・chat template全体で、最後の位置は回答第1tokenを予測。
  answerはteacher-forced回答prefix位置（最後の未採点位置も含む）。
  一語回答ではanswer側のpatchは採点対象に影響しないはず。
  prompt効果も動画tokenだけに局在するとは言えない。
- --self-check: 各層・指定scopeでfull→fullとswapped→swappedを検証。
  許容絶対誤差既定1e-4 nats（運用上の検査値、数値誤差の実測保証ではない）。
  失敗sampleはfailureへ記録し分析対象に入れない。誤差は成功recordにも保存。
- processorの非pixel出力の一致、patch形状/dtype/device、hook発火回数、有限値を検査。
- --strict-grounding: 時刻が根拠区間に入るsampled frameだけ採用。
  空ならskip。既存のcontiguous control制約は維持する。

全位置のfull由来patchは診断であって、実運用可能な補正ではない。
今回も複数headの相互作用自体は未検証。まずpatchの測定妥当性を監査する。

## 検証と次のアクション

依存なしのテストで範囲境界、未選択位置保持、hook解除、形状エラー、
未発火、入力alignmentを検証する。fake tensorを使用しCUDA数値は検証しない。
既存解析5件と監査4件の計9テストが成功。ローカルにはtorch/GPUなし。
3件×3scopeをGPUで実行し、self_check PASSを
全件確認する。失敗時は許容誤差を機械的に緩めず原因を調べる。
その後同じscope設定で25件を探索的に比較する。strict条件で対象集合が変わる
ため旧結果との単純な平均比較はしない。新規データに進む前に対象scopeを固定。
