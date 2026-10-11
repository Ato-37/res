# GitHub経由・remote desktopでの実行手順

このリポジトリは、コードと小さなサンプル定義をGitHubで同期し、動画データと
モデルキャッシュはremote desktop側に置いて実行する前提で管理する。生動画や
24GBの`NExTVideo.zip`をGitHubへアップロードしない。

## 1. remote desktopで更新

remote desktop側には、すでに`home`以下へこのリポジトリの`res`ディレクトリを
配置済みである。そのため、通常の作業ではcloneは不要で、次のように更新する。

```bash
cd ~/res
git pull --ff-only
```

remote desktop上で実験コードを変更した場合は、変更を確認してcommitし、GitHubへ
pushする。

```bash
git status
git diff
git add scripts docs experiments
git commit -m "Describe the experiment or code change"
git push origin main
```

Mac側で変更した場合も、まずremote desktop側に未commit変更がないことを確認して
から、同じく次を実行する。

```bash
git status
git pull --ff-only
```

pull前に未commit変更がある場合は、変更内容を確認してcommitまたは退避してから
実行する。`--ff-only`で更新できない場合は、無理にmergeや破壊的操作を行わず、
状態を確認する。

## 2. Python環境

Python 3.12を推奨する。まずGPUに合ったPyTorchとtorchvisionを、PyTorch公式の
CUDA対応wheelからインストールする。CUDA wheelの種類はGPU・ドライバによって
異なるため、`cu124`や`cu128`を推測せず、remote desktopの既存環境と公式配布
状況を確認する。

例(既存の研究環境と同じCUDA 12.4 wheelを使う場合):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r scripts/requirements-spike.txt
```

別のCUDA wheelを使った場合は、実験ログにTorchのバージョン、CUDA、GPU名を
必ず記録する。Qwen2.5-VL-7Bでは`bfloat16`を使うため、`float16`へ変更しない。

確認:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## 3. 動画データ

動画はremote desktop上の任意の場所に配置する。推奨例:

```text
~/nextgqa_sample/NExTVideo/1015/4631119360.mp4
```

`data/nextgqa_samples_*.json`の動画パスは`~/nextgqa_sample/...`形式なので、
remote desktopのユーザー名が変わっても同じ配置規約ならそのまま使える。
別の場所に置く場合は、実験専用のサンプルJSONを作り、`video`フィールドだけを
その環境のパスに変更する。動画ファイル自体はGitHubへ追加しない。

## 4. soft ablationのsmoke test

大規模実験の前に、まず1〜2サンプルだけで動作確認する。

```bash
source .venv/bin/activate
mkdir -p results
python scripts/soft_ablation_sweep.py \
  --samples-file data/nextgqa_samples_big_heldout.json \
  --n-samples 2 \
  --layers 6,12 \
  --scales 1,0.75,0.5,0.25,0.1,0.05,0.01,0 \
  --seed 0 \
  --output results/soft_ablation_smoke.json
```

成功条件は、全scaleについて処理が完了し、JSONに次が含まれること:

- `n_effective`と`failures`
- layerごとの`mean_correct_delta`
- control tokenとentropyの変化
- Torch、Transformers、OpenCV、GPU、seedの実行環境情報

動画読み込み失敗がある場合は、出力の`failures`に記録されたIDを確認する。
黙ってサンプル数を置き換えて本実験へ進まない。

## 5. GitHubへ戻すもの・戻さないもの

GitHubへ戻す:

- `scripts/`のコード
- `docs/`と`experiments/`の設定・解釈・実験ログ
- 再現に必要な小さなサンプルJSON

GitHubへ戻さない:

- 生動画、`NExTVideo.zip`、個人情報を含みうる画像
- Hugging Faceモデルキャッシュ
- `results/`以下の大容量JSONや一時フレーム
- remote desktop固有の認証情報・絶対パス

結果を共有する場合は、数値を実験ログへ要約し、必要なら小さな集計CSVだけを
別途検討する。未実行の結果をログへ書かない。

## 6. necessity集計図表の再生成・自動publish（CPUのみ）

保存済みblack/blur結果からCSV、SVG、metadataを再生成し、CPUテスト後にその3成果物だけを
commit/pushする。GPU推論は行わない。runnerはtracked worktree/indexに変更がないこと、Git identity、入力
ファイル、originとのbranch関係を事前確認する。既存のuntracked data directoryは触れず、生成出力だけを
明示的にstageする。出力prefixはUTC時刻付きで生成して既存ファイルを上書きしない。

Ubuntuで実行するコマンド:

```bash
cd ~/res &&
git pull --ff-only &&
python scripts/run_thesis_summary_and_publish.py
```

これは同期だけの操作ではなく、pull成功後にテスト・図表生成・対象限定commit・pushまで続ける。
認証、Git identity、競合、tracked worktree変更などで止まった場合はエラーとlocal commitを確認し、GPUや集計を
再実行せず復旧する。直前の失敗runがrunner命名のCSV/SVG/metadata 3点をstageしたままの場合、完全な同一prefix
の組だけ自動でunstageし、ファイルは残して新しい組を生成する。`results/`の他ファイルやユーザーの変更は
commit対象に含めない。
