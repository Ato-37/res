# QTA Stage-1 bundle のローカル監査

`scripts/audit_qta_stage1_bundle_cli.py` は、Stage-1の監査素材を1つのprivate directoryから読み、既存の `audit_qta_stage1_bundle()` をそのまま実行します。ファイルパスをPython APIへ個別に渡す作業を減らすためのCPU-only CLIです。レビューの代行、本人確認、時刻/フレーム照合、実験・採点の許可はしません。

## bundleの形

bundle rootとその中のdirectoryはmode `0700`、通常ファイルは所有者だけが読めるmode `0600`にします。`bundle.json`もmode `0600`です。パスはすべてbundle rootからの相対pathで、symlink、`..`、絶対path、同一pathの使い回しはできません。

```json
{
  "schema_version": 1,
  "purpose": "qta_stage1_audit_bundle_non_authorizing_v1",
  "sources": {
    "worksheet": "sources/worksheet.json",
    "lineage": "sources/lineage.json",
    "review": "sources/review.json",
    "review_index": "sources/review_index.json",
    "custody": "sources/custody.json",
    "pre_review_receipt": "sources/pre_review_receipt.json",
    "integrity_roster": "sources/integrity_roster.json",
    "release_roster": "sources/release_roster.json",
    "inventory": "sources/inventory.json",
    "release_witness": "sources/release_witness.json",
    "reviewer_attestation": "sources/reviewer_attestation.json",
    "root_public_key": "sources/root_public_key.pem"
  },
  "media_dir": "media",
  "artifact_sources": {
    "claim_mapping_worksheet": "artifacts/worksheet.json",
    "claim_lineage": "artifacts/lineage.json",
    "review_custody": "artifacts/custody.json"
  }
}
```

`artifact_sources`は例の3行だけでは足りません。`sources.inventory`に列挙された**全roleを過不足なく**含め、各roleを実ファイルへ対応させます。監査器がmedia directory内に必要な匿名token名の動画3本だけがあることも検査します。ファイルごとの上限は128 MB、bundle全体（manifest・source・artifact・動画）の合計上限は1 GBです。動画は監査器による読み込み直前の差し替えを考慮し、実サイズにかかわらず3本分の最大量（384 MB）を予約して上限を検査します。そのため、1 GB以内の実サイズでも監査を拒否する場合があります。

## 監査の実行

root公開鍵fingerprint、lineage hash、custody hash、ceremony IDはbundleの値を信用せず、信頼できる別経路で得た値をコマンドへ渡します。値を入力しただけでは、その配布元や人の本人性は認証されません。

```bash
.venv/bin/python scripts/audit_qta_stage1_bundle_cli.py \
  --bundle-dir /local/private/qta_stage1_bundle \
  --expected-lineage-sha256 <別経路で確認したSHA-256> \
  --expected-custody-sha256 <別経路で確認したSHA-256> \
  --expected-root-public-der-sha256 <別経路で確認したroot fingerprint> \
  --expected-ceremony-id <別経路で確認したceremony ID>
```

結果は構造監査のJSONです。成功しても `model_load_permitted=false`、`score_ready=false`、`execution_authorization=not_granted`のままです。別途必要な時計/フレーム照合やcalibrationは未監査として残ります。署名付きreceiptの検証も、それだけでレビューの意味的正しさを証明しません。bundle・出力は保護対象になり得るため、ローカルに保管し、公開しません。権利状態は未確認です。
