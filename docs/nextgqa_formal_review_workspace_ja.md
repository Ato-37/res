# 正式レビュー監査ワークスペース（ローカル）

`docs/nextgqa_human_review_workspace_ja.html` のニックネーム簡易確認とは別の、Stage-1 / Stage-2 検証画面です。簡易版の質問・回答やJSONは読み込みません。

画面を開く前に、正式bundleと別経路で渡されたpinを用意します。protected inputはすべてローカルに置き、Gitへ追加しないでください。起動時にPythonがStage-1 bundleとStage-2 bundleのdirectoryを読み、ブラウザには検証reportだけを返します。

```bash
.venv/bin/python scripts/nextgqa_formal_review_workspace.py \
  --stage1-bundle-dir /absolute/private/stage1_bundle \
  --stage2-bundle-dir /absolute/private/stage2_bundle \
  --expected-lineage-sha256 <out-of-band-sha256> \
  --expected-custody-sha256 <out-of-band-sha256> \
  --expected-root-public-der-sha256 <out-of-band-sha256> \
  --expected-ceremony-id <out-of-band-ceremony-id> \
  --expected-stage2-context-sha256 <out-of-band-sha256>
```

HTTP serverは`127.0.0.1`のephemeral portだけでlistenし、起動ごとのtoken、Host/Origin検査、同一origin POSTを要求します。起動時にtoken付きlocalhost URLを端末へ表示するので、Linux側のbrowser openerが使えないWSL環境でもそのURLを手動で開けます。アップロード用file pickerはありません。Stage-1検証には既存のsecure bundle loaderと`audit_qta_stage1_bundle`を使い、Stage-2はStage-1 bundle内の正確な`pre_review_receipt` bytesから得たSHA-256をupstream pinとして、baseline manifest v3、prediction-link receipt、attestationへ伝播します。Stage-2 bundleはowner-only directoryに以下の`workspace.json`と参照ファイルを置きます。

```json
{
  "schema_version": 1,
  "purpose": "qta_formal_workspace_stage2_bundle_v1",
  "baseline_manifest": "baseline.json",
  "prediction_link_receipt": "receipt.json",
  "stage2_attestation": "attestation.json",
  "context": "context.json"
}
```

`context.json`は自己申告値を信頼しません。起動引数`--expected-stage2-context-sha256`で、その正確なbytesのSHA-256をbundle外から固定します。文脈データにはbaseline v3 SHA-256、protocol SHA-256、ordered `cohort_tokens`、および`source_field_sha256_by_token`を含めます。baseline、receipt、attestation、context各ファイルはmode `0600`、Stage-2 directoryとその親はmode `0700`である必要があります。ファイル名はbundle直下の単純な名前だけ受け付けます。

画面はStage-1 reportと、Stage-1 receipt hash / baseline v3 / Stage-2 receipt / key attestationの検証結果を表示します。ファイルtreeが変わるとStage-1またはStage-2の結果を消去し、Stage-2はStage-1が再検証されるまで無効になります。出力が成功しても、本人性・実際の視聴・意味判断・clock/frame correspondenceを認証せず、`model_load_permitted=false`、`execution_authorization=not_granted`、`authorized_forwards=0`を維持します。この画面はscore runnerを起動しません。

HTTP応答はno-store、CSP、no-referrer、frame denial、same-origin CORP/COOP、機能制限headersを付けます。現在のfocused testsは合成controller/stateとlocalhost policyを検査します。実HTTP/browser E2Eは未実施です（通常sandboxではsocket bindがPermissionError）。Rights status remains unverified. Protected questions, media, reviews, keys, and derived artifacts must remain local. Synthetic tests validate software wiring only and do not establish a real review or execution gate.
