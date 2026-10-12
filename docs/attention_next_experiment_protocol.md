# 次の attention 数値診断プロトコル（事前固定）

現在の合成診断は、固定した Q/K/V で BF16 manual eager と FP32 中間計算・SDPA を比較し、長さ 584/585/586 の共通行差を測った。次の診断を実行する場合は、同じ問いを別条件で確認するため、実行前に次を固定する。

## 目的

Qwen の eager/SDPA 差が、単一clip・2 target positionsに固有か、独立clipでも同じ方向に現れるかを記述する。これは数値経路の再現性を測る診断であり、causal maskの正しさ、QA精度、説明faithfulnessを検定するものではない。

## 固定条件

- checkpoint revision、5 shardのロード前後SHA256、model dtype (`bfloat16`)、attention implementation (`eager`, `sdpa`)、PyTorch/CUDA/Transformers/bitsandbytes版をmanifestへ記録する。
- 独立clip候補母集団を実行前に固定したmanifest（sorted clip hash list）として保存し、その先頭の未使用clipを1本追加する。選定したclip ID・media hashを実行manifestへ書き、結果を見てclipを交換しない。各clipで target position を事前に2点（生成列の token 0 と token 1）選ぶ。結果を見て位置を追加・除外しない。
- 各clip・各位置について、generation、full prefix 584/585/586 の同じ19 logit比較・9 hidden-row比較を実行する。失敗は除外せず、理由と実効数を記録する。
- eagerを先に実行するかSDPAを先に実行するかを固定し、同一入力hash・target hash・video hashを両条件で一致確認する。
- 統計的有意性や実用閾値をこの診断結果から新設しない。報告はclip/位置ごとの raw max（絶対値）、`D = generation selected-token log-probability - full-prefix selected-token log-probability`、symmetric KL、およびDの符号・失敗数に限定する。「同じ方向」はこのDの符号が既存clipと一致することを指すが、符号一致を確認的な成功基準にはしない。

## 事前の停止条件

入力hash、checkpointのロード前後witness、または固定比較計画が一致しない場合は、そのclipの比較を停止して修復する。人手レビューを必要とするNOAA gateが未完了の間は、この診断をQA/faithfulnessの有効性結果として扱わない。head/layer/alpha探索は再開しない。

## 解釈範囲

独立clipでも同じ数値差が観測されれば、固定clip固有という説明は弱くなるが、原因となるCUDA kernelやattention演算段を特定しない。差が再現しなくても、現行clipでの数値差を否定せず、clip依存性を示す探索的結果として扱う。いずれの場合も、モデルの予測品質や説明のfaithfulnessの結論には接続しない。
