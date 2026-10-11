# 動画時間位置のランタイム検査とQA窓サンプリングの限界

更新日: 2026-10-12

## 目的と範囲

本メモは、Qwen2.5-VL-7Bの動画時間RoPE位置IDに関する技術検査と、NOAAの小規模QAパイロットでの窓・フレーム交差数を記録する。位置IDやlogitの変化は実装挙動であり、動画内容の理解、説明のfaithfulness、QA精度を測った結果ではない。QA窓との幾何的交差も、実際にモデルが根拠を見たことを意味しない。

## 時間位置の実装差

Qwen2.5-VLの技術報告は、動画のMRoPE時間IDを絶対時刻に合わせ、時間ID間隔からイベントの速度・時間関係を表現する設計を説明する。[原論文 §2.1.2–2.1.3](https://arxiv.org/html/2502.13923)。公式Transformers v5.17.0では、`second_per_grid_ts`をまず整数化してから`tokens_per_second`を掛ける。したがって0.75秒のグリッド間隔は、2 tokens/sの場合に`2 × int(0.75) = 0`となり、隣接する2つの時間グリッドが同じRoPE時間IDを得る。[公式v5.17.0実装](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)。

公式v4.49.0は秒数とtokens/sを先に掛け、最後に座標を整数化する。現在の公式mainも小数の秒間隔を保って時間位置を作り、結果を整数座標へ量子化する。[公式v4.49.0](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)、[公式main](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)。これは「秒を先に切り捨てる」処理と「小数秒間隔を時間IDへ変換してから切り捨てる」処理の差であり、秒単位のアノテーション精度や意味的な時刻正当性を示すものではない。

比較runnerは動画時間軸だけを置換し、空間軸と動画外の位置IDを実行中のTransformersから引き継いだ。これは現在のmainの時間計算に限定して合わせる選択で、v4.49.0全体へのバックポートではない。特に動画後テキストの位置開始規則は旧版と現行版で異なる。今回の結果から、その差を不具合または意味的影響と断定しない。

## 位置IDのランタイム結果

以下は各JSONに保存された実測値である。いずれも1動画、4フレーム、3回のforwardで、同一native-ID no-opはbitwise一致した。3番目の条件は、時間IDだけを小数秒計算へ置換した条件である。

| 入力 | 実効時間グリッドと秒間隔 | native → 小数秒候補の時間anchor | 変更ID数 | 最大/平均絶対logit差 | JS差 (nat) | top-1 |
|---|---|---|---:|---:|---:|---|
| NOAA dive03、300秒動画を4点一様抽出 | `T=2`, 199.999802秒 | `[15, 413] → [15, 414]` | 276 | 0.375 / 0.100811 | 0.000002389 | 一致 |
| 固定合成フレーム、0–1.125秒 | `T=2`, 0.75秒 | `[15, 15] → [15, 16]` | 256 | 0.5625 / 0.073542 | 0.000053647 | 一致 |

参照: [NOAAランタイム結果](../results/noaa_temporal_position_gpu_check_20261012_01.json)、[合成subsecondランタイム結果](../results/qwen_temporal_position_synthetic_subsecond_20261012_01.json)。両方とも要求/実効forwardは3/3、失敗0。合成入力は決定的なPIL画素配列で、pixel SHA-256は結果JSONに記録される。実際のprocessor metadataでは4フレーム、8/3 FPS、時刻`[0, .375, .75, 1.125]`、`T=2`、`second_per_grid_ts=.75`が確認された。

両条件のJS・logit差は、位置IDを変えたときの次トークン分布感度である。どちらが正しい回答か、時間IDが映像時刻に意味的に合うか、説明が忠実かは示さない。合成検査はprocessorとGPU forwardを通した技術回帰検査で、実動画の代替ではない。NOAAの4フレーム検査も1本の動画に限り、QA・音声・回答ラベル・生成を使っていない。

## NOAA QA窓とのサンプリング交差

対象は3つの動画に含まれる6つのQA窓であり、6項目を独立した動画標本として数えない。サンプリング監査は動画長・FPSとQA窓の数値境界を読み、抽出フレームの表示時刻と窓の交差を計算した。元の質問・選択肢・回答は監査出力へ含めていない。

| サンプリング規則 | 窓内に点サンプルが1つ以上ある窓 | 点サンプル0の窓 | 2点以上ある窓 | 窓と交差するendpoint pair span* |
|---|---:|---:|---:|---:|
| 全動画から一様4点 | 1/6 | 5/6 | 0/6 | 4/6 |
| 全動画から一様8点 | 3/6 | 3/6 | 0/6 | 4/6 |
| 各QA窓内から個別に4点 | 6/6 | 0/6 | 6/6 | 6/6 |
| 各QA窓内から個別に8点 | 6/6 | 0/6 | 6/6 | 6/6 |

\* endpoint pair spanは隣接する抽出点間の時間区間であり、その間の映像フレームをモデルへ渡したことを意味しない。全動画4点条件では、span交差4窓のうち3窓に実際の抽出点が入らなかった。全動画8点条件でも、spanだけが交差し点サンプルがない窓が1つあった。したがってspan交差を「証拠を見た」割合として報告しない。

各QA窓からの個別抽出はQA時刻境界で条件づけた方式で、全動画一様抽出と同じ入力分布ではない。窓に入ること自体は可視根拠との一致、回答正当性、適切なnegative controlを保証しない。この監査はサンプリング可能性の比較に限定する。[窓coverage監査](../results/noaa_sampling_coverage_20261012_01.json)。

## QA・faithfulnessのゲートと次の判断

QA gateはcanonicalな3動画群、合計6サンプル・gold・auditの6件join、固定rights/media hashesを監査した。label leakは0/6で、AI blind reviewは6件。人間の別個なattestationがなかったため判定は`no_go`、唯一のfailureは`human_attestation_missing`である。[QA gate結果](../results/noaa_qa_gate_20261012_01.json)。このgateは動画のみのQA推論の前提を確認するもので、人間が実際に動画から回答と証拠を再確認したことを捏造したり、QA efficacyを示したりはしない。

**次の実データfaithfulnessへ進む条件:**

1. まず、責任を持つ人間が全6件のgold回答と時間証拠を動画で確認し、各入力hashとレビュー済み項目に紐づくattestationを作る。現在の一般的な実験許可やAI reviewer記録を人間の確認として扱わない。
2. QA gate通過後も、この3動画・6 QAは注釈とサンプリングのfeasibility cohortに留める。独立単位は動画IDで`N=3`であり、6 QAを独立な確認標本として扱わない。十分な独立holdoutでの検証精度を裏付ける根拠がないため、確証的faithfulnessの主張には使わない。
3. faithfulnessを問う別プロトコルでは、モデル説明を検証可能なイベントclaimへ分解する規則、人手で独立に確認したclaim–時間区間、対応する動画内controlの意味妥当性、primary score・動画分割・除外・停止条件を、結果を見る前に固定する。単純な時間ずらしやmaskを意味的counterfactualと呼ばない。十分な独立動画holdoutを確保できない場合は、探索的feasibilityに限定する。

今回の結果は新しいhead/layer/dose探索の根拠ではない。既報のhead確認は、その固定estimandについて平均選択的margin差`−0.0756 nat`、95% t CI `[-0.1578, +0.0065]`で、`+0.25 nat`実質基準に届かなかった結果として維持する。[確認結果](../experiments/2026-10-08_head_faithfulness_confirmation.md)。保存結果はTransformers `5.17.0`を記録しているが、各例へ実際に渡った全position-ID列やモデルコードsource hashは保存していない。今回の時刻位置検査から過去効果を再解釈・上書きしない。

## CPU再構成で分離した追加の限界

現在の旧入力builderを16枚・2FPSの固定合成JPEGで再構成すると、FPS metadataを渡さない経路はprocessorの24FPS fallbackを使用した。明示metadataとの比較で画像tensor・token列・gridは一致し、時間座標1792/2048だけが変化した。このCPU再構成は過去79件で実際に使った座標の証明ではない。[設定を一致させた結果](../results/legacy_video_time_metadata_20261012_02.json)。

固定8条件の精度監査では2条件でPython fractional floorとfloat32の丸め結果が異なった。GPU runnerは今後のforward前に一致を検査し、不一致なら停止する。今回の2つのGPU入力はCPU上で一致を確認したが、GPU結果をこの新guard下で再実行したわけではない。[精度監査](../results/qwen_fractional_precision_20261012_01.json)。旧79件はhash一致する入力manifestで別ID・別path・別byte列を確認したが、独立なsceneや統計的独立性までは証明しない。[入力identity監査](../results/head_confirmation_video_identity_20261012_01.json)。
