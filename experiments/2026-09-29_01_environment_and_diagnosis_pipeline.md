# Day1: 環境構築〜診断パイプライン(head ablation)の確立

このファイルは2026-09-29に実施した作業の記録。当時はまだ`experiments/`配下にログを残す運用を始めておらず、後日(2026-10-01)に遡って記録した。以後の`experiments/*.md`はすべてこの続き。

## 背景・目的

確定方針(診断=head ablation → 補正=activation steering → 検証=MECD型necessityテスト、詳細は[research_overview.md](../docs/research_overview.md))の最初のステップとして、Qwen2.5-VL上でattention weightsを実際に取得できるかを確認し、head ablationによる診断パイプラインを構築する。

## 環境構築

### サーバー1: sakurazaka(研究室、RTX 4060 8GB)

`sudo`も`conda`も使えず、ホームディレクトリがNASマウント(`/nas.dbms/okamoto`)というサーバー向けの手順:

1. `curl https://pyenv.run | bash`でpyenvをユーザー権限インストール
2. `pyenv install 3.12.3`でPythonをソースからビルド
3. リポジトリで`pyenv local 3.12.3`
4. `python -m venv .venv`
5. `pip install torch --index-url https://download.pytorch.org/whl/cu124`(ドライバCUDA13.1でもcu124ビルドで動作確認)
6. `pip install torchvision --index-url https://download.pytorch.org/whl/cu124`
7. `pip install -r scripts/requirements-spike.txt`

### サーバー2: 自宅WSL2機(RTX 5070 Ti 16GB、Windows+WSL2 Ubuntu)

atlantis/discovery(研究室の24GB機)が他ユーザー使用中で使えなかったため、当面の主力機として採用。

- GitHubのパスワード認証は廃止済みのため`gh auth login --web`→`gh auth setup-git`が新しいマシンごとに必要
- WSLのUbuntuデフォルトPython(3.14)はPyTorchのwheelが存在せず失敗 → `deadsnakes` PPAでPython 3.12を別途導入
- RTX 5070 Ti(Blackwell世代)は**cu128系のPyTorchビルド**を使用(cu124は未検証)

## 8GB級GPUでOOMを避ける工夫

- `BitsAndBytesConfig(load_in_4bit=True, ...)`で4bit量子化
- `AutoProcessor(min_pixels=max_pixels=256*28*28)`で画像/動画トークン数を強制的に絞る

## 画像での動作確認

[spike_test_attention.py](../scripts/spike_test_attention.py)でQwen2.5-VL-3B-Instruct(4bit量子化)×画像1枚が成功。`output_attentions=True`+`attn_implementation="eager"`で実際にattention weightsを取得できることを確認(36層、shape `(1,16,280,280)`、peak_gpu_memory=2.83GB)。sakurazaka・自宅WSL2機の両方で完全に同じ結果が再現し、パイプラインの環境間移植性を確認した。

## 動画への拡張

[spike_test_video_attention.py](../scripts/spike_test_video_attention.py)で、`qwen_vl_utils`のdecord依存を避けるためOpenCVで自前にフレーム抽出し、Qwen2.5-VLに「フレーム画像のリスト」として渡す方式を採用。4フレームで成功(`layer0_attention_shape=(1,16,563,563)`、動画トークンがattention計算に正しく含まれることを確認)。

フレーム数とメモリのスケーリング(Qwen2.5-VL-3B):

| フレーム数 | seq_len | peak_gpu_memory_gb |
| --- | --- | --- |
| 4 | 563 | 3.16 |
| 8 | 1095 | 4.37 |
| 16 | 2159 | 8.79 |

eager attentionはO(seq_len²)でattention行列を保持するため、フレーム数が増えるほど二乗で効いてくることを実測で確認した。

## 7Bモデルへのスケールアップと重要なハマりどころ: fp16はQwen2.5-VLの罠

3Bで問題なかった`torch_dtype=torch.float16`が、7Bでは`torch.multinomial`が「probability tensor contains inf, nan or element < 0」でCUDA device-side assertを起こして失敗した。4bit/8bit/量子化なしのいずれでも再現したため、**量子化ではなくfp16の数値レンジ不足が原因**と判明。`torch_dtype`と`bnb_4bit_compute_dtype`を両方`torch.bfloat16`に変更して解決。

**以後このリポジトリの実装は必ずbfloat16を使う**(float16は使わない)。7Bでの成功: 8フレーム・4bit量子化・bfloat16、`num_layers_with_attention=28`(3Bの36層・16headと異なり、7Bは28層・28head)、peak_gpu_memory=8.59GB。

## NExT-GQA実サンプルでの検証

### データ取得

- NExT-GQAの生動画は個別配布されておらず、train/val/test全部入りの単一24GB zip(`NExTVideo.zip`、Google Drive)としてのみ配布。
- Google Driveの当該ファイルは公開ダウンロード上限に達しており`gdown`で直接取得できなかった。**`copy_file`で自分のDriveに複製→リンク共有ON→新しいfile IDで`gdown`**することで回避(今後も同様の場面で使える手)。
- `unzip -l archive.zip | grep <video_id>`でzip内の該当パスを確認し、`unzip archive.zip "path" -d dest/`で1本だけ展開すれば24GB全体の展開を避けられる。

### アノテーションの読み方(NExT-GQA公式リポジトリ: `github.com/doc-doc/NExT-GQA`)

- `datasets/nextgqa/test.csv`: video_id, question, answer, qid, 選択肢
- `datasets/nextgqa/gsub_test.json`: `{video_id: {duration, location: {qid: [[start_sec, end_sec], ...]}, fps}}`
- `datasets/nextgqa/map_vid_vidorID.json`: video_id → zip内パス

### 最初の実サンプルでの発見

video_id=2574374895(30秒動画)、質問「what did the baby do after throwing the green cup away while on the floor near the end」、正解「lay on floor」(根拠区間23.0〜27.7秒)に対し、モデルは誤答(「the baby stood up and walked away from the camera」)。8フレーム均等サンプリングには根拠区間に近いフレームも含まれていたが、それでも誤答した。

**意義**: NExT-GQA論文の核心的発見(「正答率は高くても正しい根拠に基づいているとは限らない」)を自分の手元で再現した実例であり、この卒研が診断・補正しようとしている「不忠実性」そのもののインスタンスを確保できた。

## Head ablation(診断)の初回実装

[head_ablation_spike.py](../scripts/head_ablation_spike.py)で、`model.model.language_model.layers[i].self_attn.o_proj`にforward_pre_hookを仕掛け、該当head(またはlayer全体)の出力をゼロ化し、正解トークンのlog確率の低下を測定する仕組みを実装。

**モジュールパスの注意**: `named_modules()`で見える相対パス`model.language_model.layers.N`は誤りで、実際には`model.model.language_model.layers[N]`(先頭に`.model`が一段余分)。

### 層単位ablationの初回結果(1サンプル、baseline_logprob=-23.35)

layer=0(drop=-9.05)、layer=27(drop=-5.20)が突出。当初は「誤答へのショートカットを担っている」と解釈したが、後述のコントロールトークン検証でこの解釈は誤りと判明した。

## コントロールトークン導入による再検証(重要な訂正)

「layer 0/27の劇的な効果はショートカット除去なのか、単なる分布平坦化なのか」を切り分けるため、ベースラインで正解トークンと同程度に低確率な無関係トークン(control、`pick_control_token`関数で機械的に選択)を導入し、同じablationでどう動くか追跡した。

**結果、layer=0/27の効果は正解に特異的ではないと判明**:
- layer=0: avg_correct_drop=-9.90 に対し avg_control_drop=-11.89(controlの方が大きく変動)
- layer=27: avg_correct_drop=-7.40 に対し avg_control_drop=-6.60(ほぼ同水準)

→ **この2層の効果は「ショートカット除去」ではなく単なる出力分布の崩壊(confidence collapse)と解釈するのが妥当**。layer=23も同様の問題(entropy変化が小さいことだけでは「クリーンな信号」と判断できない)を抱えていた。

選択性(correctだけが上がりcontrolが上がらない)が見られた候補: idx=26、idx=14、idx=8(絶対値は小さいが、最もマシな「正解に特異的な」候補)。

補足: `pick_control_token`が選ぶトークンがしばしば不完全なバイト列になっていたため、後に正規の英単語トークンのみに限定する改良(`build_valid_word_mask`)を加えた。

**教訓**: これは研究テーマ自体が扱おうとしている「忠実性指標そのものが忠実性を測れているとは限らない」問題の実例。診断パイプラインでは常にcontrol比較を併用する必要がある。

## 16サンプルへの拡張

コントロールトークン選定を正規の英単語のみに限定した上で、NExT-GQAテストセットから16サンプル(video_idを分散)に拡張して層単位ablationを再実行した([data/nextgqa_samples.json](../data/nextgqa_samples.json))。

選択性が確認できた層: idx=23(correct=+1.16, control=+0.75)、idx=4(+0.84, +0.50)、idx=9(+1.21, +0.97)、idx=1(+1.60, +1.29)、idx=8(+1.23, +1.08)。layer 0/27は16サンプルでも非選択的なまま再確認された。

## 次に読むファイル

このDay1の続き(temporal grounding欠落の発見、layer=6/head=16の発見経緯、Step2初回試作)は[2026-09-29_02_grounding_discovery_and_steering_v1.md](2026-09-29_02_grounding_discovery_and_steering_v1.md)を参照。
