# Faithfulness 閾値・関連研究レビュー

日付: 2026-10-09。文献・設計レビュー。GPU実験・動画取得・モデル推論は未実施。

## 結論

事前登録の成功境界 +0.25 nat は今回の確認で事後変更しない。79/79件、平均 −0.0756 nat、95% t CI [−0.1578, +0.0065]で不達。head/layer/alpha探索を再開する根拠はない。

+0.25 natは尺度上は説明できる。log-odds差とすればexp(0.25)=1.284で、target介入後の回答対foil oddsはcontrol後の約0.779倍。ただし回答確率が28%変わることを意味しない。1.284がなぜ最小実質効果なのか、実務的帰結・理論予測・同一estimandの先行データ・独立校正による根拠は記録されていない。したがって境界は「解釈可能」だがSESOIとして未妥当化。標本数設計は実質的根拠にならない。

観測SD .3667、n=79の下でCI下限が+.25を超えるには平均が概ね+.332を超える必要がある。観測分散を仮定した参考計算では、真の差+.30で判定通過約22%、+.35で約67%、+.50で99.9%超。事後powerではなく基準の識別特性の概算。

結果は推定値と区間を中心に報告し、「事前固定だが実質的根拠は未校正」と注記する。境界を下げて今回を成功扱いしない。将来の別estimandでは、データを見る前に理論・実務上の帰結・先行研究・独立判断からSESOIを決める。根拠が弱ければ効果推定と精度を主目的にする。

根拠: Lakens et al., “Sample Size Justification,” Collabra (2022), https://doi.org/10.1525/collabra.33267 （SESOI・期待効果・最小検出効果を区別）。2025 methods review https://www.nature.com/articles/s41599-025-06436-w （理論、知覚閾値、意思決定上の帰結等を根拠候補として論じる）。

## 一次資料と転用可能性

| 論文 | 使える着想 | 限界・注意 |
|---|---|---|
| Kang et al., CVPR 2025, [Localization Heads](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_For_Visual_Grounding_CVPR_2025_paper.html) | 少数headのtext-image attentionで物体grounding | 空間局在≠予測への因果的重要性。動画時間拡張は別検証 |
| Kaduri et al., CVPR 2025, [What’s in the Image?](https://openaccess.thecvf.com/content/CVPR2025/html/Kaduri_Whats_in_the_Image_A_Deep-Dive_into_the_Vision_of_CVPR_2025_paper.html) | query-token圧縮、層別attention knockout、回答/人手/物体局在評価 | 静止画中心、Qwen動画へ直接一般化不可 |
| Zhang et al., CVPR 2025, [Cross-modal Information Flow](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.html) | image→question→answer edge knockout。動画token/時間controlの設計源 | 画像GQA中心。correct-only解析の選択バイアスに注意 |
| Wu et al., CVPR 2024, [SaCo](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html) | salience順位と摂動強度のdose-response比較 | pixel手法でありheadへ直移植不可、mask OODも残る |
| Li et al., NeurIPS 2023, [M4](https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html) | 複数faithfulness指標の比較 | 動画VLM benchmarkではない |
| Li et al., CVPR 2026, [TCAS / Temporal Logic Consistency](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Understanding_Temporal_Logic_Consistency_in_Video-Language_Models_through_Cross-Modal_Attention_CVPR_2026_paper.html) | 時刻別cross-modal attention識別性、言い換え・grounding整合性。Qwen2.5-VLを含む。最も近い直接関連 | 時間grounding/論理一貫性で、説明の因果的faithfulnessとは異なる。本文・補足・コード精読を優先 |
| Li et al., CVPR 2025, [VidHalluc](https://openaccess.thecvf.com/content/CVPR2025/html/Li_VidHalluc_Evaluating_Temporal_Hallucinations_in_Multimodal_Large_Language_Models_for_CVPR_2025_paper.html) | 動作・順序・場面遷移の時間hallucinationとvideo pairs | 説明faithfulnessではない。pair差分局所性を要確認 |
| Upadhyay et al., CVPR 2026, [Time Blindness / SpookyBench](https://openaccess.thecvf.com/content/CVPR2026/html/Upadhyay_Time_Blindness_Why_Video-Language_Models_Cant_See_What_Humans_Can_CVPR_2026_paper.html) | 時系列だけに信号を置く時間知覚評価 | 合成・ノイズ様刺激から自然動画へ一般化しない |
| Bansal et al., CVPR 2024, [VideoCon](https://openaccess.thecvf.com/content/CVPR2024/html/Bansal_VideoCon_Robust_Video-Language_Alignment_via_Contrast_Captions_CVPR_2024_paper.html) | 対照caption、行為/順序差のtaxonomy | 主に同一動画+contrast captionで、編集video pairではない |
| Zanella et al., CVPR 2025, [SynViTA](https://arxiv.org/abs/2503.18507) | negative caption条件の動画生成 | 他要因・品質も変化し得る。盲検品質判定、sham、局所性必要 |
| Park et al., CVPR 2018, [Multimodal Explanations](https://openaccess.thecvf.com/content_cvpr_2018/html/Park_Multimodal_Explanations_Justifying_Decisions_and_Pointing_to_the_Evidence_CVPR_2018_paper.html) | 活動認識の説明文と視覚evidence併記 | 人のもっともらしい説明≠モデル内部理由 |
| Wu et al., EMNLP 2023, [ACQUIRED](https://aclanthology.org/2023.emnlp-main.719/) / [COVER](https://aclanthology.org/2025.findings-acl.151/) Findings ACL 2025 | 動画counterfactual QA、同一動画のoriginal/counterfactual質問。COVERはQwen2-VLを含む | 多くは質問だけを変え、動画編集pairでない。QA正答は忠実性でない |
| Li et al., CVPR 2022, [Causal-VidQA](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_From_Representation_to_Reasoning_Towards_Both_Evidence_and_Commonsense_Reasoning_CVPR_2022_paper.pdf) | description/explanation/prediction/counterfactual QAの統合 | 説明の正答性は内部因果寄与を保証しない |
| Chuang et al., EACL 2026, [FaithLM](https://aclanthology.org/2026.eacl-long.177/) | contrary-hint介入で説明fidelityを測る | テキストhintが視覚を上書きし得る。[Words or Vision? CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Deng_Words_or_Vision_Do_Vision-Language_Models_Have_Blind_Faith_in_CVPR_2025_paper.html)のようなmodality-bias controlが必要 |
| Uppaal et al., EACL 2026, [Journey Before Destination](https://aclanthology.org/2026.eacl-long.194/) | 視覚claimと推論を分けたCoT評価 | judge/人手評価はgroundednessを支えてもcausal faithfulnessは証明しない |
| Jung et al., CVPR 2025, [On the Consistency of Video-LLMs in Temporal Comprehension](https://openaccess.thecvf.com/content/CVPR2025/html/Jung_On_the_Consistency_of_Video_Large_Language_Models_in_Temporal_CVPR_2025_paper.html) | 時刻を特定した後、言い換え・時間シフト・検証質問でgrounding/verification consistencyを測る | 近接する動画時間評価。時間groundingの一貫性であり、自然言語説明が内部予測原因を表すかは測らない |
| Yeo et al., EMNLP 2025, [Activation Patching for Faithful NLEs](https://aclanthology.org/2025.emnlp-main.529/) | 説明と内部活性の因果帰属をactivation patchingで照合し、入力摂動OODを避ける狙い。方法論上かなり近い | LLM/NLP実験で、動画token・時間局在・説明claimへの妥当性は未検証。パッチングの因果解釈と定義を精読する |
| Zaman & Srivastava, EMNLP 2025, [A Causal Lens for Evaluating Faithfulness Metrics](https://aclanthology.org/2025.emnlp-main.1496/) | model editingでfaithful/unfaithful説明対を作り、指標自体の診断性を検査。連続metricのnoise/model感度も報告 | fact-checking等の言語タスク。動画への直接転用ではなく、評価指標を先に検証する必要性を示す |
| Siegel et al., ACL 2024, [The Probabilities Also Matter](https://aclanthology.org/2024.acl-short.49/) / Atanasova et al., ACL 2023, [Faithfulness Tests for NLEs](https://aclanthology.org/2023.acl-short.25/) | 予測分布全体を使うcounterfactual test、説明にない理由を加える/記載理由から入力を再構成する二方向テスト | 主にNLP。理由文と予測に共通原因があるだけでは媒介・因果寄与を証明しない。動画反実仮想の構成妥当性が別途必要 |
| Lyu et al., Computational Linguistics 2024, [Towards Faithful Model Explanation in NLP](https://aclanthology.org/2024.cl-2.6/) | 内部構造、勾配、counterfactual、自己説明などfaithfulness評価の分類と限界整理 | NLP surveyで動画実験ではない。用語・構成概念・手法比較の導入資料 |
| Balasubramanian & Basu, [Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.723/) / Wojciechowski et al., [Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.130/) | 画像CoT bias、画像説明とneuron介入 | 動画時間根拠には直接答えない |

## 次の作業と判断

groundedness（claimが動画に実在）、faithfulness（claimが予測に因果的寄与）、temporal grounding、回答正答率を分ける。attentionが指すこと、遮断で回答が変わること、説明が内部理由を表すことは同値ではない。

第一候補は、説明から検証可能なイベントclaimと時間区間を抽出し、独立注釈で正確性を測る。次にそのイベントだけを変えた品質確認済み動画counterfactualと対応shamで固定回答margin/回答反転を比べる。妥当なpairが作れなければ実験に進まない。black/blur、質問反転、contrary textは動画counterfactualの代用にならない。

当面はCPUで、CLEVRER simulatorの介入・再render経路、VideoCon/SynViTA、ACQUIRED/COVER/Causal-VidQAのpaired videoの有無、時間ラベル、ライセンス、Qwen入力適合性を一次資料で監査する。既に不成立だったCLEVRER質問pairのpost-hoc距離閾値探索はしない。TCAS方向に進むなら説明faithfulnessとは別の問いになること、新規性を明確化する。
