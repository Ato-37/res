# データの配置

`data/nextgqa_samples_*.json`にはNExT-GQA動画へのパスが含まれるが、動画本体は
このリポジトリに含めない。現在のサンプルJSONは次のようなremote desktop側の
パスを想定している。

```text
~/nextgqa_sample/NExTVideo/<directory>/<video>.mp4
```

別の環境で実行する場合は、動画をGitHubへアップロードせず、実験専用のJSONを
作成して`video`フィールドを環境内のパスへ変更する。データセットのライセンス
条件と共有範囲を確認してから使用すること。
