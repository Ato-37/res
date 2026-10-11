# Qwen2.5-VLの時間位置とキャッシュ継続の実行条件

2026-10-12時点の既存Transformers5.17.0に対する実行経路の監査。これは時間理解・QA精度・faithfulnessの改善ではない。`.venv`を変更せず、旧確認結果も変更しない。

## 実測した範囲

[固定CPU診断](../results/qwen_cache_positions_cpu_20261012_02.json)は、seed20261012、ランダム初期化した小型Qwen2.5-VL、CPU FP32、1系列・36 token、video placeholder32 token、grid[8,4,4]を使う。vision encoderを動かさず、実動画・注釈・回答を使わない。7B BF16/4bitの数値や動作を再現したとは扱わない。

| 経路 | 観察 | 解釈できる範囲 |
|---|---|---|
| 通常の連続generate | native自動位置とpacked明示nativeのscores・sequenceが一致 | この小型モデルのno-op |
| fractional temporal rowだけを明示してgenerate | prefillだけ変更し、次のtext位置はnativeと同じ6 | 最後のtext座標+1で継続 |
| 明示位置によるprefill | rope_deltasが空なら空、古い値があれば保持 | 状態を自動更新する前提は置けない |
| キャッシュからgenerateを再開 | 空/native/candidate/古いdeltaで次の位置36/6/13/11 | 状態・継続方針で位置が変わる |
| 位置なしの直接cached forward、全mask37・current token1 | position37個を計算しRuntimeError | この固定API呼び出しの形状不一致 |
| 対応するfresh cache、同じmask、位置1個を明示 | 有限logitsを返す | 形状契約の修復。意味妥当性ではない |

5秒間隔の別固定geometryでは、nativeの最大位置+1は73、最後のtext位置+1は6。これはgeometryの実測であり、この条件のdecodeは実行していない。最大位置を使う方式を「正しい継続」と決めつけない。

## 区別する二つの規則

通常の`generate`はQwenのpacked位置 `[text row; temporal,height,width rows]` を使い、継続時に各rowの最後の値へ1を足す。対して、別途キャッシュから再開する経路はmodel内の`rope_deltas`を参照できる。native実装のdeltaは `max(3つの位置row)+1−有効prompt長` であり、最後のtext座標と一致する保証はない。[公式5.17モデル](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)、[公式5.17生成処理](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)。

時間rowだけの変更と、全位置・text継続方針の変更は別介入である。`replace_video_temporal_rope_row`は前者だけを行い、cacheやdeltaを管理しない。今回のコードを全バージョン対応のmodel修正として使わない。

今後の実装では、prefillの位置と継続方針を一緒に記録し、別のモデル入力から残ったdeltaを再利用しない。位置を明示するならcurrent input長に対応する位置だけを渡す。通常生成と再開を切り替える際は、実際にdecoderへ渡った位置・cache長を検査する。batch、padding、beam、複数video、images、static cacheの成立性は今回の小型検証に含まれない。

## Ubuntuでの再検証

以下は実行済みの8テストを再検証するCPUコマンド。

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_audit_qwen_cache_positions_cpu
```

新しい記録を作る場合は、未使用のoutput名を指定する。既存結果は上書きしない。

```bash
PYTHONPATH=scripts .venv/bin/python scripts/audit_qwen_cache_positions_cpu.py --output results/qwen_cache_positions_cpu_reproduction_NEW.json
```

後者の例名では未実行。実GPUの互換性確認を行うなら、目的を7B BF16/4bitの実際の視覚入力・packed位置・cacheの接続に限定し、CPU結果と混ぜずに別manifestへ記録する。QAゲートを通過させるものではない。
