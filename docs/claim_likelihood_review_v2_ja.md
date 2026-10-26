# 動画claimの簡易レビュー手順

このページは、NExT-GQAの実動画3件と、モデルが生成した英語の説明文を照らして見るための簡易確認です。厳密な正解判定や研究者の資格確認は行いません。説明文を十分に読めない場合は判断を入力しないでください。

## 開き方

WindowsのChromeのアドレス欄へ次のパスを貼り付けて開きます。

```text
\\wsl.localhost\Ubuntu\home\ato121\res\data\private_nextgqa\claim_likelihood_human_review_v2_20261021_01.html
```

ファイルはこのWSL作業フォルダ内にあり、WindowsのDownloadsには自動コピーされません。

## 回答の流れ

1. ニックネームを入力して「開始」を押します。
2. 質問・説明、動画、モデルが使った32フレームを見ます。画像の0〜31番号はモデル入力のフレーム番号です。
3. 「だいたい合っている」「一部異なる」「判断できない」から一つ選び、「次へ」で進みます。任意メモは必要な場合だけ書きます。
4. 3件目まで終わると、次の担当者のニックネーム入力へ進めます。一人で終える場合は保存へ進みます。
5. 全員が終わったあと、最後に一度だけ「全レビュー担当者の回答をJSON保存」を押します。Windowsのブラウザ設定に応じてDownloadsなどへ保存されます。

レビュー担当者ごとにニックネームは入力し直します。実名や連絡先は書かないでください。保存JSONにはニックネームと任意メモが入るため、公開・共有せずローカルで扱ってください。

## JSONの検査

保存されたJSONをDownloadsからWSLで読む場合の例です。実際のファイル名に合わせてパスを直してください。

```sh
PYTHONPATH=scripts .venv/bin/python scripts/inspect_claim_likelihood_human_review_v2_response.py \
  --artifact data/private_nextgqa/claim_likelihood_human_review_v2_20261021_01.html \
  --response <windows-user>/Downloads/claim_likelihood_human_review_responses.json
```

検査結果には各選択肢の合計と完了セッション数だけを表示し、ニックネーム・メモ・個別動画は表示しません。検査が通っても、ニックネームによる本人確認や担当者の独立性は証明されず、この簡易判断だけで正式な直接faithfulness scoreになるわけではありません。
