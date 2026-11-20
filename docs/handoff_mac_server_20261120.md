# 引き継ぎ (2026-11-20): 以降の実験はラボのGPUサーバーで実行する

別チャット(Mac側のClaude Code)が、このファイルと `experiments/2026-11-18_new_research_plan_map_the_flow.md` だけで続きを進められるようにした資料。

## 1. 目的と方針 (変更しない)
- 研究: 動画VLMの説明忠実性。基礎論文 "Map the Flow" (ICLR 2026, byminji/map-the-flow) の再現と拡張。
- モデルは**公式の LLaVA-NeXT-7B-Video-FT** と公式コード (https://github.com/byminji/map-the-flow, commit 566f3ed) のみ。
- 旧実験は「なかったもの」として、更地から再実験する (ユーザー指示 2026-11-18)。
- 結論は作らない: 一致/部分一致/不一致を、そのまま報告する。
- 計画: `experiments/2026-11-18_new_research_plan_map_the_flow.md` (走らせて基準を修正する方式: dev=探索 → 凍結 → test で1回だけ評価。基準の変更は同ファイル末尾の「実行ログ」に必ず記録)。

## 2. 重要: 環境を統一する
- **ローカル(WSL, RTX 5070 Ti, torch 2.11.0+cu128, transformers 4.40.0)で出した結果は、サーバーの結果と合算しない。**
- 以降の実験は、すべてサーバーの1つの環境で取り直す。ローカル結果 (`results/mtf_v2_local_5070ti/`) は「探索の参考」と「環境差の比較用」であり、確認用の証拠ではない。
- サーバー環境の同等性チェック: 同じ dev 動画 (各タスク5本, k=9) を両環境で処理し、`base_margin` と層ごとの `d_margin` を動画ごとに比較する。許容差は比較前に決めて記録する。

## 3. サーバー
| 名前 | GPU | VRAM | CUDA(表記) | RAM |
|---|---|---|---|---|
| atlantis | RTX 3090 | 24GB | 11.6 | 128GB |
| discovery | RTX 3090 | 24GB | 11.6 | 128GB |
| perseverance | RTX 3090 | 24GB | 11.6 | 64GB |
| enterprise | RTX 3080×2 | 10GB×2 | 11.6 | 128GB |
| deimos | RTX 3060 Ti | 8GB | 12.2 | 32GB |
- 7B を bf16 のまま載せられる(約15GB)のは **3090 の3台だけ**。enterprise と deimos は、4bit化や分割が必要で、公式条件から外れるため使わない。
- 使い方: `ssh gw.dbms.tunnels` (gw) → `ssh <server>`。**入る前に Zabbix で使用状況を確認し、他の人が使っていたら極力使わない。** 入ったら `top` と `nvidia-smi` を確認。実行中は
  `nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv -l 1`
- アカウント作成は先輩に相談が必要 (ユーザーが対応)。
- **atlantis の確認結果 (2026-11-20, ユーザー確認)**: RTX 3090 は空き(75MiB/24GB, 使用率0%)。ドライバ 590.48.01 (CUDA 13.1 まで対応)。表の「CUDA 11.6」は古い情報で、**cu118 でも cu128 でも動く**。→ ローカルと同じ `torch==2.11.0+cu128` を入れられる(CUDAの版は揃えられる。残る差はGPU世代: Blackwell vs Ampere のみ)。CPU/RAM は余裕 (空き約116GB)。OS Ubuntu 24.04, pyenv Python 3.12.3 (conda なし; ローカルは 3.12.14 なので3.10は不要)。他の利用者 takumi が CPU を少し使うのみ。他のサーバーの CUDA 表記も古い可能性があるが、未確認。
- **ディスク**: ホームは共有NAS (/nas.dbms, 11TB中98%使用, 残り約273GB)。大きなファイルは控えめに置く。目安: モデル約15GB + 動画約3.2GB + venv約8〜10GB = 30GB前後。NTU の元zip(1.6GB)は置かず、取り出した320本だけ置く。`HF_HOME` などのキャッシュ先を指定して、ホームに無駄に溜めない。NAS は読み込みが遅い可能性があるので、ローカルディスク (`/tmp` 等) が使えるか `df -h` で確認し、使えるなら速度を比較する。
- 共有サーバーにデータを置く場所とライセンス: 動画 (TVBench, NTU RGB+D) は再配布不可。置き場所を先輩に確認。生動画や重みをgitに入れない。

## 4. 用意するもの (gitに入っていない)
1. 公式コード: `git clone https://github.com/byminji/map-the-flow && git checkout 566f3ed`。`tasks/eval/config_dataset.py` の `DATASET_PATH['tvbench']` を、TVBench のルート(下記の構成)に書き換える。環境変数 `MTF_CODE_DIR` でこのcheckoutを指す。
2. モデル: 公式の LLaVA-NeXT-7B-Video-FT (公式 README のHugging Face 配布先から取得。約15GB)。環境変数 `MTF_MODEL_PATH` で指す。
3. TVBench: `<root>/json/*.json` と `<root>/video/{action_antonym,action_sequence,moving_direction,object_count,scene_transition}/`。Action Antonym は NTU RGB+D の元zipから `scripts/extract_ntu_action_antonym.py` で必要な動画(320本)だけを取り出す。
4. Python環境(ローカルの版; atlantis では同じ torch が使える): python 3.12.14, torch 2.11.0+cu128, transformers 4.40.0, tokenizers 0.19.1, peft 0.10.0, accelerate 1.15.0, decord 0.6.0, av 19.0.2, timm 0.9.16, numpy 2.5.2。サーバーでも、できるだけ同じ版を入れ(pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 など。公式の手順で確認)、実際に使った版を meta と実験ログに記録する。

## 5. 実装済みスクリプト (`scripts/mtf_v2/`)
- `make_split.py`: dev/test を動画単位で作る(ハッシュ規則; 既存出力は上書きしない)。出力済みの `tvbench_split_v1.json` (sha256 d8a689bf2f7702387e6aba08493b2b5469b2c13322e8aa85a58bb2955d521c6d) をそのまま使う。dev/test 動画数: Action Antonym 186/134, Action Sequence 172/164, Moving Direction 124/108, Object Count 60/87, Scene Transition 99/86。
- `flow_v2.py`: 窓幅 k (層方向) を中心にずらして、フレーム間 attention を遮断し、正解選択肢と最良の誤答のマージン(nat) の変化を、全32層・全問(正答/誤答を選ばない)で記録する。`--null` で同数のランダムセルを遮断する対照、`--fp32_readout` で lm_head を fp32 で読む(必須: bf16 だと 0.125 nat 刻み)、`--skip_n`, `--done_from`(完了済み動画をスキップ)。出力は JSON-lines (先頭行が meta)。既存パスには書かない。
- `summ_pilot.py` / `analyze_pilot.py`: 解析(動画を単位にした bootstrap)。
- 実行例 (スモーク→本番):
  ```
  export MTF_CODE_DIR=~/map-the-flow MTF_MODEL_PATH=~/models/LLaVA-NeXT-7B-Video-FT
  python scripts/mtf_v2/flow_v2.py --out out/smoke.jsonl --split scripts/mtf_v2/tvbench_split_v1.json --tasks "Action Antonym" --n_per_task 1 --ks 5 --null --fp32_readout
  python scripts/mtf_v2/flow_v2.py --out out/dev20.jsonl --split scripts/mtf_v2/tvbench_split_v1.json --part dev --n_per_task 20 --ks 5,9,13 --null --fp32_readout
  ```
  実行は `setsid nohup ...` で切り離し、GPUジョブは1つずつ(同時実行は約10倍遅くなった)。
- 速さの目安 (RTX 5070 Ti): 動画1本・1つのkで約125秒(64 forward)。3090で同じかは未計測 → スモークで測る。

## 6. ローカルで得た結果 (5070 Ti, fp32読み出し, 記述のみ, 基準未確定)
ファイルは `results/mtf_v2_local_5070ti/*.jsonl.gz` (SHA256SUMS つき)。
| ファイル | 内容 | 完了 |
|---|---|---|
| dev_pilot_fp32_20261118 | dev 先頭5本/タスク, k=1,5,9,13, null | 100/100 video-k |
| dev20_fp32_20261119 | dev 先頭20本/タスク, k=5,9,13, null | 300/300 |
| dev20b_fp32_20261119 | dev 20–39本目, k=5,9,13, null | 168/300 (中断: AA20, AS20, OC16本) |
| dev20b_resume_fp32_20261120 | 同, 残りの再開分 | 5/132 (中断) |
dev20 の観察 (n=20/タスク, L6–15 平均, k=13, real / null, nat): Action Antonym −0.31 / −0.01, Scene Transition −0.15 / +0.07, Action Sequence −0.07 / −0.28, Object Count −0.34 / −0.73, Moving Direction −0.02 / +0.03。L16以降はほぼ0、k=1 はほぼ0。ベースライン正答 (dev 20本中): AA 9, AS 12, MD 4 (偶然水準 → 除外候補), OC 8, ST 11。
- 5本の予備実験では Object Count で real < null だったが、20本では逆転(null の方が強く効く)。→ **現在の null (vision×vision 下三角からランダムに同数セル) は real より強く効くタスクがあり、real−null 差だけでは解釈できない。** 同じフレーム間領域内でフレーム位置だけをずらした null への再設計が必要。
- これらは dev の探索であり、主張には使わない。

## 7. 次にやること (優先順)
1. サーバーに入り、環境確認(ドライバ・CUDA版・空きGPU・ディスク)、torch の版を決める。
2. 環境構築、公式コードの動作確認、`flow_v2.py` のスモーク(1本)。
3. ローカル結果との同等性チェック (§2)。
4. null の再設計 (フレーム位置シャッフル null) を実装し、dev で検証。
5. dev を取り直す (サーバー環境で)。dev20 と dev20b を合わせて 40本/タスク。その上で、層の基準 (bootstrap 区間の上限 < −δ_min, BH-FDR) と窓幅 k を決め、凍結。
6. test で1回だけ評価 → 転用 5×5 → ヘッド単位の探索 (ランダム部分集合 → 相互作用 → 最小集合; `scripts/run_llava_head_experiment.py` にヘッド単位マスクの実装がある)。
7. 結果は privacy-safe な要約だけ `results/` にコミット(生動画・重み・絶対パスは入れない)。

## 8. 未解決・注意
- ベースモデルの processor は、repo からの読み込みに失敗し compat tokenizer へフォールバックする(video モデルは同一出力)。影響は未検証。
- 論文の「有意な低下」(5層ビンの相対確率変化の平均 < −5%) は検定ではない。本計画では margin (nat) と null 対照、動画単位の bootstrap を使う。
- Moving Direction は 4択 (偶然水準25%) で、ベースライン正答がほぼ偶然水準 → 解釈不能。dev ベースラインが偶然水準近傍なら除外する(計画 §0)。
- 公式の fast copy (mask_num_heads=1 のブロードキャスト) は、ヘッド単位には使えない。`flow_v2.py` は自分でマスクを作る。
- 実行環境の権利: 動画・モデルのローカル限定利用。再配布の権利は未確認。
