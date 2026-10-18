# QTA phase receipt verifier のローカルbundle形式

`scripts/verify_qta_phase_receipts.py` は、phase-1/phase-2 receipt候補と関連artifactのバイト結合を読み取り専用で確認するCLIです。結果は構造検証に限られ、モデル読込・採点・forwardを許可しません。

## bundleを用意する

保護対象データを含むbundleはローカルだけに置き、ディレクトリをmode `0700`、ファイルをmode `0600`にします。bundle rootに次の固定形式の`bundle.json`を置きます。各pathはbundle rootからの相対pathです。

```json
{
  "schema_version": 1,
  "purpose": "qta_phase_receipt_bundle_non_authorizing_v1",
  "phase1_receipt": "phase1_receipt.json",
  "phase1_artifacts": {
    "stage1_review_and_lineage": "phase1/stage1_review_and_lineage.bin",
    "clock_frame_evidence": "phase1/clock_frame_evidence.bin",
    "source_backed_targets": "phase1/source_backed_targets.bin",
    "prompt_answer_token_bindings": "phase1/prompt_answer_token_bindings.bin",
    "sampler_model_runtime": "phase1/sampler_model_runtime.bin",
    "condition_scorer_source_closure": "phase1/condition_scorer_source_closure.bin",
    "full_video_processor_batches": "phase1/full_video_processor_batches.bin",
    "atomic_cohort_preflight": "phase1/atomic_cohort_preflight.bin",
    "external_cohort_pin": "phase1/external_cohort_pin.bin"
  },
  "phase2_receipt": "phase2_receipt.json",
  "phase2_artifacts": {
    "phase1_receipt": "phase2/phase1_receipt.bin",
    "baseline_pq_manifest": "phase2/baseline_pq_manifest.bin",
    "stage2_prediction_link_review": "phase2/stage2_prediction_link_review.bin",
    "nine_branch_plan_manifest": "phase2/nine_branch_plan_manifest.bin",
    "condition_scorer_source_closure": "phase2/condition_scorer_source_closure.bin",
    "attention_capture_index": "phase2/attention_capture_index.bin",
    "attention_call_receipts": "phase2/attention_call_receipts.bin"
  },
  "integrity_roster": "integrity_roster.json",
  "release_roster": "release_roster.json",
  "root_public_key": "root_public_key.pem",
  "attention_capture_sidecar_dir": "sidecars"
}
```

`phase2/phase1_receipt.bin`は`phase1_receipt`と同じバイト列でなければなりません。`sidecars/`にはattention capture indexが参照する各SHA-256について、`<sha256>.bin`という名前のファイルだけを置きます。余分なファイル、欠落、symlink、root外へのpath、緩い権限は拒否されます。通常ファイルは最大24 MB、各phaseのartifact合計は最大96 MB、sidecar合計は最大384 MiBです。

## 実行する

cohort pinとtrust-root fingerprintはbundleから計算せず、別経路で受け取った値を必ず明示して渡します。

```bash
.venv/bin/python scripts/verify_qta_phase_receipts.py \
  --bundle-dir /local/private/qta_phase_bundle \
  --expected-external-cohort-pin-sha256 <別経路で受け取った64桁SHA-256> \
  --expected-trust-root-der-sha256 <別経路で受け取った64桁SHA-256>
```

成功時の出力は行tokenやローカルpathを含まないJSON summaryです。byte joinが成功しても`forward_source_authenticated=false`、`attention_capture_authenticated=false`を明示します。`authorized_forwards`は常に`0`、`model_load_permitted`は`false`、採点・実行許可は`not_granted`です。CLI引数に値を渡した事実だけでは、その値の配布元や本人性を認証できません。両fingerprintの別経路が確認できない場合、結果を人手レビューや実験実行の許可として扱ってはいけません。

このbundleと出力は保護対象データを含み得るため、Gitへ追加・外部共有しません。権利状態は未確認のままです。
