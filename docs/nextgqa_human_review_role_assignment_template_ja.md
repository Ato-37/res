# NExT-GQA 人手レビュー担当者の割当票

英語の[元テンプレート](nextgqa_human_review_role_assignment_template.md)の日本語版です。レビュー資料を誰かが開く前に、ローカルで記入してください。これは候補者リストであって、記入しただけでは本人確認・信頼・レビュー開始許可にはなりません。

## 1. 信頼の起点（root）

- root authorityの識別名：
- root公開鍵ファイルのローカルパス：
- 別経路で確認した公開鍵DER SHA-256 fingerprint（64桁の小文字16進数）：
- そのfingerprintを確認した独立経路・手順：

**注意：** root鍵とfingerprintを同じ未検証の資料から受け取っただけでは信頼確認になりません。秘密鍵は記入・共有しないでください。

## 1a. 担当者が自分の鍵を作る

各担当者は、自分に割り当てられたkey IDを使い、自分だけが管理する端末で次を実行してください。例では `replace-with-your-key-id` を担当者へ割り当て済みのkey IDに置き換えます。担当者ごとに別の鍵を作り、秘密鍵は送信・共有しません。freeze ownerとcustodianを同一人物にする場合だけ、同一鍵を兼用できます。

```bash
set -euo pipefail
umask 077
KEY_ID='replace-with-your-key-id'
KEY_DIR="$HOME/.local/share/nextgqa-review-keys"
if [[ ! "$KEY_ID" =~ ^[A-Za-z0-9._-]{1,64}$ ]]; then
  printf 'key IDは英数字・ピリオド・下線・ハイフンの1〜64文字にしてください。\n' >&2
  exit 1
fi
if [[ -L "$KEY_DIR" ]]; then
  printf '鍵保存先にシンボリックリンクは使えません。\n' >&2
  exit 1
fi
mkdir -p -- "$KEY_DIR"
if [[ -L "$KEY_DIR" || ! -d "$KEY_DIR" ]]; then
  printf '鍵保存先にシンボリックリンクは使えません。\n' >&2
  exit 1
fi
if [[ "$(stat -c '%a' -- "$KEY_DIR")" != 700 ]]; then
  printf '鍵保存先はmode 0700で用意してください。既存ディレクトリの権限は変更しません。\n' >&2
  exit 1
fi
ROLE_KEY_DIR="$KEY_DIR/$KEY_ID"
if [[ -e "$ROLE_KEY_DIR" || -L "$ROLE_KEY_DIR" ]]; then
  printf '既存の鍵を上書きしません。別のkey IDを選んでください。\n' >&2
  exit 1
fi
mkdir -m 700 -- "$ROLE_KEY_DIR"
cleanup_role_key_dir() {
  rm -f -- "$ROLE_KEY_DIR/private.pem" "$ROLE_KEY_DIR/public.pem"
  rmdir -- "$ROLE_KEY_DIR" 2>/dev/null || true
}
trap cleanup_role_key_dir EXIT
openssl genpkey -algorithm ED25519 -out "$ROLE_KEY_DIR/private.pem"
openssl pkey -in "$ROLE_KEY_DIR/private.pem" -pubout -out "$ROLE_KEY_DIR/public.pem"
chmod 600 -- "$ROLE_KEY_DIR/private.pem" "$ROLE_KEY_DIR/public.pem"
openssl pkey -pubin -in "$ROLE_KEY_DIR/public.pem" -outform DER | sha256sum
trap - EXIT
```

最後の行に表示される64桁のSHA-256は照合用です。公開鍵は `$KEY_DIR/$KEY_ID/public.pem` に作成されます。担当者はこの公開鍵ファイルとkey IDだけをroot authorityへ安全な経路で渡し、root authorityは実在の担当者と公開鍵の対応を独立経路で確認します。秘密鍵は `$KEY_DIR/$KEY_ID/private.pem` に保存され、秘密鍵、秘密鍵のパスフレーズ、連絡先をこの票やGitへ記入・共有しません。root authorityは別途、root秘密鍵を管理し、root公開鍵とDER fingerprintを互いに独立した経路で配布・確認します。担当者用の鍵やこの作業環境からroot鍵を作ってはいけません。

## 2. 役割別の担当者・公開鍵

| 役割 | 担当者の安定ID | key ID | 公開鍵ファイルのローカルパス | 独立性・兼務の確認 |
|---|---|---|---|---|
| 事前凍結担当（freeze owner） |  |  |  | 説明生成・採点・全レビュー・撮影から独立 |
| 保管担当（custodian） |  |  |  | 凍結担当との兼務のみ可。レビュー担当・撮影担当との兼務不可 |
| 撮影担当（capture operator） |  |  |  | 他の役割から独立 |
| 第1段階・説明レビュー担当 |  |  |  | 説明生成・採点・保管・撮影から独立 |
| フレーム対レビュー担当 |  |  |  | 撮影・保管・ペア準備・採点から独立 |
| 第2段階・予測リンクレビュー担当 |  |  |  | 第1段階・生成・採点・保管・凍結・立会いから独立 |
| 開封前の独立立会人（release witness） |  |  |  | 全員と別人・別鍵。rootから独立確認済み |

公開鍵ファイルは通常ファイルで、本人所有のモード0600にします。秘密鍵やパスワードをこの票に書かないでください。担当者ID・鍵IDはレビュー資料を開く前に確定します。実在する担当者と鍵の対応は、rootの独立した確認経路で確認してください。

鍵の使い回しは原則禁止です。凍結担当と保管担当を同一人物とする場合に限り、同じ鍵を共有できます。立会人と第2段階レビュー担当は、他の全役割と異なる鍵が必要です。

## 3. 開封前チェック

- [ ] 凍結担当が対象ファイル・手順・samplerを選び、レビュー結果やモデル出力を見る前にhash固定した。
- [ ] root fingerprintを独立経路で確認した。
- [ ] rootが担当者と公開鍵の名簿へ署名した。
- [ ] 凍結担当が `review_only` receiptへ署名した（モデル読込・採点・実験実行は許可しない）。
- [ ] 立会人が未開封の対象とreceiptを直接確認し、凍結担当から保管担当への引渡しを見届けた。
- [ ] 開封直前に `scripts/audit_nextgqa_review_release_order.py` が全入力を監査し、`package_opening_permitted=true` を確認した。
- [ ] 全レビュー担当者が、説明・盲検・開示に関する注意を読んだ。

どれか一つでも未完了または監査失敗なら、資料を開かずに止めて原因を解消します。署名済みreceiptは自動的に採点を許可しません。

## 4. レビュー完了チェック

- [ ] 第1段階：3/3件の説明支持判定と十分な出現区間を記録し、担当者が署名した。
- [ ] 撮影：計15箇所のプレイヤー時刻・画面・デコード対応記録を作成した。
- [ ] フレーム対レビュー：独立担当者が15/15組を確認し、記録へ署名した。
- [ ] 第2段階：固定済み予測と対照を使った予測リンク判定を別担当者が3/3件完了した。
- [ ] 各段階の構造監査と次段階の専用score-freeze監査が通った。

現在の記録では上記レビューは未実施（0/3、0/15、0/3）です。レビュー完了はfaithfulnessの向上を意味しません。実際の内部介入スコアとその不確実性を得るには、別途スコア実験が必要です。
