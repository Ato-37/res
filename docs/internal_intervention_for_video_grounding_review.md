# 内部介入による動画grounding精度・忠実性向上：文献レビュー

調査日: 2026-10-09
対象: 動画VLM（当面はQwen2.5-VL）の中間表現・visual token・attention経路への推論時介入

## 要旨

研究目標としては、単に「動画QAの正答率や幻覚スコアを上げる」ではなく、次の二つを区別したうえで同時に改善を目指すのが妥当である。

1. **Grounding精度**: 回答・説明中のイベント／主張を、動画中の正しい時刻（必要なら領域）に結び付ける精度。
2. **Faithfulness**: その説明・主張が、モデル内部で実際に予測を支えた動画証拠に因果的に依存している度合い。

両者は関連するが同じ指標ではない。根拠区間へattentionが集まること、QA正答率が上がること、幻覚ベンチマークが改善することのいずれも、それ単独では説明のfaithfulnessを証明しない。したがって、内部介入で grounding を改善したうえで、別の選択的介入・反事実評価によりfaithfulnessを検証する二段の評価が必要である。

今回確認した文献のうち、動画への技術的転用可能性が最も高いのは、(a) Qwen2.5-VLを直接扱う **TA-AE** と **VERHallu/KFP**、(b) 時空間局所証拠の再注入と局所反事実を組み合わせる **STEAR** である。ただし、いずれも既報の主評価は幻覚・QA・イベント関係タスクであり、NExT-GQA型のgold時間区間IoUと自然言語説明のfaithfulnessを同時に確認した結果ではない。よって、これらは手法候補／比較対象であって、本研究の成果を先取りする証拠ではない。

## 研究上の用語と評価対象

本レビューでは「grounding精度」を、質問・回答・説明内のイベントや事実主張に対して、その根拠となる動画時刻（または時空間領域）をどれだけ正しく指せるか、と置く。NExT-GQAの時間的根拠ラベルはこの問いに使える。ただし、そのラベルを使って介入位置を選べば評価情報の漏洩になるため、介入側は推論時に利用可能な内部信号で決め、gold intervalは独立評価に留めるべきである。

Faithfulnessは、出力が正しいか／もっともらしいかではなく、説明された主張が該当する動画証拠に因果的に依存するかを問う。例えば、説明中の主張について選択された証拠表現を弱めたときの予測変化を、同数・同条件の非証拠control介入と比較する設計が候補となる。ただし、masking自体がOOD入力を作る可能性、controlの情報量不一致、margin指標のprompt・選択肢依存は残る。単純なattention可視化を因果的faithfulnessと呼ばない。

## 文献比較

| 論文 | 内部介入の要点 | 動画・Qwen2.5-VL適合性 | 本研究への評価 |
|---|---|---|---|
| [TA-AE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html) | 時間変化に応じたactivation offsetとモジュール選択で、推論時に内部表現を補正 | 動画を直接評価。対象モデルにQwen2.5-VLを含む | 最も近い既報基準。ただし主に幻覚ベンチマークで、時間区間IoU／説明faithfulnessの改善を直接示すものではない |
| [VERHallu / KFP, arXiv 2026](https://arxiv.org/abs/2601.10010) | 中間層のvisual tokenをkey frame選択・隣接時刻へ伝播して強調 | 動画を直接扱い、QwenVL-2.5-7Bで評価 | モデル・モダリティ適合性が高い。イベント関係タスクに有用だが、CFQAで悪化しうる。gold temporal localizationとは別物 |
| [STEAR, arXiv 2026](https://arxiv.org/abs/2604.03045) | grounding-sensitiveな中間層で選択した局所証拠を再注入し、同じ証拠に局所時間counterfactualを適用 | 動画専用。確認した本文ではQwen2.5-VL評価を確認できず | 発想上もっとも有望。証拠選択と因果的なnegative branchを共有。ただし査読前preprintで、現状の評価は主にhallucination/QA |
| [RIVS, IJCAI 2026](https://www.ijcai.org/proceedings/2026/94) | 生成中のvisual grounding shift近傍からhallucination表現方向を作り、hidden stateを射影 | Qwen2.5-VLを含むが、画像タスク | 生成時点に応じたrepresentation介入の参考。動画の時間局在への転用は未検証 |
| [CausalLens, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) | attention headのvisual/text/system経路を分解し、視覚寄与を中間表現で補正 | 画像VLM。Qwen2-VL等で、Qwen2.5-VL動画ではない | 視覚経路への選択的介入の設計参考。head探索の再開を正当化するものではない |
| [Modular Attribution and Intervention, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html) | モジュール単位のcounterfactual編集で幻覚に関わる寄与を同定し、介入 | 画像VLM | 内部介入の因果分析の先例。動画時刻groundingを評価していない |
| [VTI, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4008025c2182bfe16fcc8566ee14d64-Abstract-Conference.html) | query/image perturbationを使うvisual・textual latent steering | 画像VLM | 潜在表現steeringの比較候補。動画の時間証拠選択への転用は推測段階 |
| [Localization Heads, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html) | 少数headのattentionから画像領域を読む | 画像VLM。主にgrounding readout | 局在指標・候補信号の参考。ただし読み出し／可視化であり、モデル内部介入の効果証拠ではない |

### 直接転用候補の詳細

#### 1. TA-AE：動画×Qwen2.5-VLの最も近い比較対象

TA-AEは、同一動画・質問に対し時間的情報を劣化させた入力と通常入力を用意し、その差から内部activation offsetを構成する。時間変化の性質に応じてモジュールを選び、推論時にoffsetを適用する設計で、追加のLLM fine-tuningを必要としない。NeurIPS 2025 Main Conference掲載で、動画モデル群にQwen2.5-VLを含めるため、実装・再現性を調べる第一候補である。[公式論文ページ](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html)・[論文PDF](https://papers.nips.cc/paper_files/paper/2025/file/5683583b88da51c79d2eb263f295286e-Paper-Conference.pdf)・[公式コード](https://github.com/cai-jianfeng/TA-AE)

注意点は、時間間引きで作る“hallucination-inducing”側が、単に証拠を弱めるだけでなく、動作・時間構造や入力分布まで変えうること。さらに既報のhallucination score改善は時間区間の境界精度や説明の因果的忠実性とは異なる。したがってTA-AEをそのままfaithfulness手法として採用せず、まず既報の対象・データ・指標を再現可能な比較として位置づけるべきである。

#### 2. VERHallu/KFP：Qwen2.5-VLの中間層・時間近傍frame介入

VERHalluは因果・時間・subevent関係について、関係分類、QA、counterfactual QAを評価する動画ベンチマークであり、論文は574動画・7,676サンプルと報告する。KFPは中間層（論文記載では主にlayer 8–15）のvisual hidden tokenにおいて、重要frameの重みを近傍frameへGaussian型に広げて再強調する。QwenVL-2.5-7Bを直接評価している点は、本研究の対象との技術距離が短い。[論文本文](https://arxiv.org/html/2601.10010)

ただし、QwenVL-2.5-7BのKFP結果は一様な改善ではない。論文の表ではCFQAが50.6から44.6へ低下する一方、QA-Causalは40.8から41.9、QA-Temporalは50.5から49.1へ変化し、QA-Subeventは40.7から45.3、SRHは27.8から43.5へ上がる。評価表の各指標を混同せず、改善とトレードオフを並べて読む必要がある。[KFP結果表](https://arxiv.org/html/2601.10010#S6)

この差は研究設計上重要である。正例の証拠を強調する介入は、情報不足・証拠不在を認識すべき例でもモデルを視覚的に過信させうる。したがって、NExT-GQA等でgrounding改善を主張する場合も、根拠不在／曖昧例や反対証拠を含む例を独立に評価し、選択的忠実性が損なわれていないかを確認する必要がある。

#### 3. STEAR：局所証拠の再注入と時間counterfactualを結ぶ

STEARは、危険度の高い生成stepでtokenに関連する時空間patchを選び、中間層で証拠を再注入し、同じ証拠仮説を使った局所時間摂動で後段推論を検査する構成を提案する。LLaVA-Video-7Bでは論文報告値としてEventHallusion 46.70→54.77、VidHalluc 63.07→65.81、NExT-QA 58.43→61.88。ただし、これらは個別benchmarkのQA/hallucination指標であり、gold interval IoUや自然言語説明のfaithfulness推定値ではない。[論文本文](https://arxiv.org/html/2604.03045)

著者らのablationでは、全stepへ一律適用すると効果が落ち、random patchでも一部の改善が残る。これは「どこかの介入なら効く」ことと「選択した正しい証拠への介入が効く」ことを区別する必要性を示す。一方、現在確認できる版はarXiv preprintであり、Qwen2.5-VLを評価した記述は見当たらない。よって、発想を採る場合もQwen2.5-VL固有のpatch/frame token構造に合わせた実装・再検証が必要である。[ablation表](https://arxiv.org/html/2604.03045#S5)

### 画像VLM研究から借りられるもの／借りられないもの

RIVS（IJCAI 2026）は、出力生成中の視覚attention低下とhidden representation変化を結び付け、hallucination方向への射影介入を行う。Qwen2.5-VLを含む評価が報告されているが、対象は画像ベンチマークである。生成stepごとの表現変化を利用する設計は動画にも移植可能な仮説だが、動画では自然な時間的attention移動と「根拠からの逸脱」を区別しなければならない。[IJCAI公式ページ](https://www.ijcai.org/proceedings/2026/94)

CausalLens（CVPR 2026）やICLR 2025のModular Attribution and Interventionは、視覚・テキスト経路や内部moduleの寄与を介入で調べる先例として有用である。しかし画像評価を動画のtemporal groundingへ外挿できない。特に、本プロジェクトでは事前確認が失敗した既存head/layer/alpha探索を繰り返す根拠に使わず、異なるestimand・介入対象・独立確認コホートをあらかじめ定義できる場合に限って参考にする。

Localization Heads（CVPR 2025）はattention mapから画像領域を読むgrounding法である。これは内部介入後の読み出し指標候補を考える材料にはなるが、attentionの領域一致だけでその領域が生成説明の因果的根拠だとは結論できない。動画化には時刻方向のtoken割当て・フレーム抽出方式の影響も検証が要る。

## 本研究に対する提案

現時点で妥当なのは「内部介入でgroundingを良くすればfaithfulnessも上がる」と仮定することではなく、次の仮説を分離して検証することである。

> **仮説G:** 推論時の時空間visual evidenceに対する選択的内部介入は、介入なしと比べて、動画中の根拠時間区間の局在精度を改善する。
> **仮説F:** その改善は単なる視覚過信や全般的なlogit変化ではなく、説明された主張が当該証拠に依存する選択性の改善を伴う。

### 事前に固定すべき測定

- **Grounding primary**: 主張／イベントごとの予測時間区間とgold intervalのtemporal IoU、および複数IoU閾値でのRecall。予測区間の出し方を固定し、gold intervalを介入選択に使わない。モデルに区間を明示させる場合は、構造化出力の解析失敗も数える。
- **Faithfulness co-primaryまたは独立secondary**: 同じ出力主張について、根拠に対応する内部状態の選択的介入と、フレーム数・時間位置・介入強度を可能な範囲で合わせたcontrol介入との差を測る。marginだけでなく、正答保持・反対主張への移行・予測分布崩壊を報告する。
- **品質・安全性control**: QA accuracy、根拠不在／曖昧例での過剰断定、反対証拠条件、生成長・不確実性を報告する。根拠の強調でCFQAが落ちたKFP結果は、このcontrolを省けない具体例である。
- **推論時の選択信号**: attention／TAMは候補信号として使えても、因果的寄与の証明とは扱わない。可能なら、選択信号が介入対象を選ぶ能力と、gold intervalを独立に当てる能力を別々に評価する。
- **統計**: 動画単位で対応を保った効果量と95%区間を主に報告し、少数の事前指定primaryに限定する。複数thresholdやサブグループは多重性を処理するか探索的と明記する。成功境界はデータを見る前に実質的・測定論的根拠を示す。

### 優先順位

1. **第一候補の比較対象: TA-AE** — 対象モデルと動画が一致し、activation engineeringの直接比較ができる。論文のプロトコル・コードと本プロジェクトのモデル版／frame samplingの差をCPUで監査する。
2. **内部介入の設計候補: STEAR/KFPの「局所証拠を選び中間層で扱う」考え方** — ただし既報で用いたhead/layer/thresholdを後付けで最適化しない。Qwen2.5-VLのtoken-frame対応を確認し、候補介入を一つに固定する。
3. **独立評価設計: NExT-GQA時間区間でgroundingを測り、別のmatched control／証拠不在条件でfaithfulnessの選択性を検査する。** 既存データのgold intervalが説明主張にどの程度対応するか、annotation粒度・複数根拠・曖昧例をCPUで先に監査する。
4. **Go/no-go** — 有効な主張単位区間ラベル、推論時に漏洩しない介入選択法、OODを抑えたcontrol、独立confirmation設計のいずれかが定義できなければ、GPU実験は開始しない。

既存のhead-faithfulness確認は事前実質境界を支持せず、CLEVRERの構造的pair監査も意味的counterfactualを確立しなかった。このレビューはその結論を覆すものではなく、過去のhead/layer/alpha探索の再開も提案しない。次の実験を行うには、上記のgrounding primaryとfaithfulness評価を明示した**新しいestimand**、データ適合性ゲート、および独立確認計画を事前に登録する必要がある。現段階では、文献の再現可能性とNExT-GQAラベル対応のCPU監査が先で、GPU実験は未承認・未実施である。

## 書誌・調査上の注意

- 本レビューは主要な候補を絞ったもので、関連研究全体のsystematic reviewや網羅性保証ではない。
- 数値は各論文の報告値で、こちらで再実行・独立検証した結果ではない。異なるbenchmark間の絶対値を直接比較しない。
- TA-AEはNeurIPS 2025 Main Conference、RIVSはIJCAI 2026 Main Track、CausalLensはCVPR 2026、Modular Attribution and InterventionおよびVTIはICLR 2025掲載。VERHallu/KFPとSTEARはこの調査時点でarXiv preprintとして扱った。
- 先行するfaithfulness指標・内部介入研究の整理は[faithfulness methods review](faithfulness_methods_and_internal_interventions_review.md)も参照。
