# Qwen系列長差分のattention backend診断

2026-10-12時点で、固定したNOAA 1 clip・4 frames・2 target positionsについて、Qwen2.5-VL-7B-Instructの系列長診断をeagerとSDPAで比較した。入力、checkpoint revision、5 shardのロード前後SHA256、prompt/target hash、19 logit比較、9 hidden比較を固定した。

観測された値は次の通り。

| 条件 | full-prefix hidden差 | full-prefix logit差 | generation token 1との差 |
|---|---:|---:|---:|
| eager, BF16 | 最大7.5 | 最大0.875 | hidden最大8.7129 / logit最大1.03125 |
| SDPA, BF16 | 0 | 0 | hidden最大1.0 / logit最大0.21875 |

SDPAでは、full584/585/586の共通行について、hidden・raw logit・centered logit・symmetric KLがすべて0だった。generation token 1とfull prefix row 584の差は残り、selected-token log-probability差は7.05072e-8 natだった。したがって、SDPAはeagerで観測されたlong-prefix driftの一部を消すが、cached generationとfull forwardを完全同値にはしない。

Q/K/Vを固定した合成causal-attentionでも、manual eagerのBF16計算は長さ584/585/586でraw差0.0009765625、0.00048828125、0.0009765625を示した。SDPAは3比較すべて0だった。BF16入力をFP32へ昇格してmanual eagerの積和・softmaxを計算し、出力をBF16へ戻すcontrolも3比較すべて0だった。FP32全体のmanual eagerも3比較すべて0だった。

この組合せは、今回の固定環境で「BF16の低精度accumulationとmanual eager attentionのshape依存丸め」がQwenのlong-prefix driftと整合する候補であることを示す。Qwenの全層での因果を証明したわけではなく、causal maskの正しさ、CUDA kernel全般、QA精度、説明faithfulnessを確立しない。対象は1 clip・2 target positionsであり、別clip・別モデル・別GPUへの一般化は未検証である。

次の確認は、同じ固定比較を独立clipまたは明示的に固定した複数target位置へ拡張し、eager/SDPA差が再現するかを事前に決めた件数で測ること。ただしNOAA human-attestation gateは別系統で未完了のため、これらの数値診断をQA/faithfulness efficacyへ接続しない。
