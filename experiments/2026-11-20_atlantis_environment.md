# atlantis の実験環境 (2026-11-20)

状態: 環境構築の途中。この環境での推論・実験は**まだ1回も実行していない**。

## 目的
Map the Flow 再現 (`experiments/2026-11-18_new_research_plan_map_the_flow.md`) の以降の実験を、ラボの GPU サーバー atlantis の1つの環境で実行する。

## 決定事項 (ユーザー決定, 2026-11-20)
- サーバー: atlantis だけを使う (同じ構成の discovery/endeavour は予備。結果を混ぜない)。
- 環境: **公式準拠** (公式 README の動作確認環境 Python 3.10 / PyTorch 2.2.1 / CUDA 11.8 と、公式 `requirements.txt` の版)。ローカル (RTX 5070 Ti, torch 2.11+cu128) の版には揃えない。
- 作業場所: `/nas.dbms/okamoto/mtf` (共有NAS のホーム。/nas2.dbms は個人フォルダが管理者作成制で未作成)。

## ハードウェア・OS (確認済み)
- GPU: NVIDIA GeForce RTX 3090 24GB (Ampere), ドライバ 590.48.01 (CUDA 13.1 まで対応)。確認時の使用 75MiB / 使用率 0%。
- RAM 125GiB (利用可能 約116GiB), OS Ubuntu 24.04.4 LTS。
- ディスク: `/nas.dbms` 11TB 中 98% 使用 (空き 約273GB)。

## ソフトウェア (確認済み)
- Python 3.10.21 (pyenv でビルド, GCC 13.3.0)。ssl/sqlite3/bz2/lzma/ctypes/readline の import を確認。
- venv: `$W/venv310`。torch 2.2.1+cu118 / torchvision 0.17.1 / torchaudio 2.2.1。`torch.cuda.is_available()=True`、3090 上で bf16 行列積を確認。
- 公式コード: byminji/map-the-flow commit 566f3ed064d324408fdf96f7a921ebf1e65f0908。変更は `tasks/eval/config_dataset.py` の `DATASET_PATH['tvbench']` 1行のみ。
- 主要パッケージ: transformers 4.40.0, tokenizers 0.19.1, peft 0.10.0, accelerate 0.26.1, timm 0.6.12, numpy 1.26.4, av 16.1.0, decord 0.6.0, safetensors 0.4.2, huggingface-hub 0.21.4, einops 0.8.2。全体は atlantis の `$W/pip_freeze_venv310.txt` (122行)。
- `tasks.eval.model_utils` と `analysis.causal_intervention_tools` の import を確認 (flash-attn 未導入の警告のみ。flow_v2.py は eager attention)。

## 公式からの差分 (再現性のため記録)
1. `requirements.txt` から次を除外: apex, deepspeed, flash-attn, en_core_web_lg, spacy, setuptools(固定版), triton(固定版; torch 2.2.1 が triton 2.2.0 を導入), wandb, notebook。理由: 学習・外部連携用で、eager attention の推論には不要と判断。**不要であることは import までしか確認していない。**
2. `einops 0.8.2` を追加 (公式 requirements に無いが `models.pllava.modeling_pllava` の import に必要)。
3. ローカル結果との比較: ローカルと atlantis では GPU 世代 (Blackwell / Ampere) と torch・timm・accelerate・numpy・av の版が異なる。同等性チェック (引き継ぎ資料 §2) で差が出ても、どの要因かは切り分けられない。比較は記述的にとどめ、ローカル結果は合算しない。

## データ
- モデル: byminji/LLaVA-NeXT-7B-Video-FT, revision 74b4fb86ee34b4322de65efc622abd25144bcb75 (ダウンロード中)。
- TVBench (FunAILab/TVBench): json と、Action Sequence / Moving Direction / Object Count / Scene Transition の動画zip を取得中。
- **Action Antonym の動画は Hugging Face の TVBench に含まれない** (json のみ)。NTU RGB+D は利用登録制のため、取り出し済みの320本を手元から転送する必要がある (未解決)。
- 動画・モデルは git に入れない。再配布の権利は未確認。

## 次のアクション
1. ダウンロード完了を確認し、TVBench を `<root>/json`, `<root>/video/<task>/` の構成に展開する。
2. Action Antonym の320本を転送する。
3. `flow_v2.py` のスモーク (1本, k=5, null, fp32 読み出し) で動作と所要時間を測る。
4. `--done_from` の不具合 (途中まで書かれた動画を完了扱いにする) を直す。
