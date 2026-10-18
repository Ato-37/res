# QTA phase receipt bundle CLI の非認可検証（2026-10-19）

## 目的と実装

phase-1/phase-2 receipt APIはexternal cohort pinを要求するが、一般のbundleを読み込んで検証するcallerがなかった。読み取り専用CLI `scripts/verify_qta_phase_receipts.py` を追加し、owner-only bundle manifest、phase artifact、roster、root公開鍵、attention sidecarを固定schemaで読み込み、既存のphase-2候補検証器へ渡す。期待cohort pinとtrust-root DER fingerprintは必須CLI引数であり、artifactから計算するfallbackや環境変数fallbackはない。

path traversal、symlink、非regular file、group/world-readable permissions、size超過、manifest duplicate keys、sidecarの欠落/余分/不一致を拒否する。出力項目は許可リストから作り、row tokenやlocal pathを除外する。検証結果を返す前に`forward_source_authenticated=false`、`attention_capture_authenticated=false`、`execution_authorization=not_granted`、`model_load_permitted=false`、`phase_authorized=false`、`authorized_forwards=0`を必須条件として確認する。byte joinの成功を実forwardの認証と誤読しないため、両capture flagsをsummaryへ出す。日本語のbundle手順は `docs/qta_phase_receipt_verifier_bundle_ja.md`。

## 検証

合成Ed25519鍵・合成artifact・小規模sidecarからowner-only bundleを組み立てるCLI統合テストを実行した。**11/11 CLI統合テスト、依存する関連テストを含む114/114 tests passed**。独立監査でも重大な迂回は見つからず、テストは成功時も0 forward/non-authorizingであること、caller pin/root mismatch、両値必須、phase1 receipt copy差し替え、path traversal、file/root symlink、duplicate manifest keys、緩いfile mode、manifest size超過、sidecar欠落/余分/symlinkを確認する。Python compileと`git diff --check`も成功した。real bundle、human review、production runner、GPU forwardは実行していない。

## 解釈と制約

このCLIは独立値を運用時に明示的に渡す受け口を作るが、CLI引数の値がfreeze ownerやtrust rootから別経路で届いたことは認証しない。報告も`external_cohort_pin_value_origin_authenticated=false`および`trust_root_value_origin_authenticated=false`とする。phase2検証候補は実スコアrunnerへ接続されず、実行許可も付与しない。従って、これは実データgateの完成ではなく、caller-side transport contractの工学的前進である。

直接faithfulness進捗はclaim review **0/3**、player-clock/decoder frame correspondence **0/15**、prediction-link **0/3**、直接score **0/3**。実`ΔF`、CI、検定はない。簡易レビューJSONも確認されていない。RTX 5070 Ti/CUDA 12.8は個別許可コマンドで利用できるが、既存の人手不要GPU proxyは完了済みで、今回CLI検証にGPU forwardは必要ない。

## 次の作業

別経路のcohort pin/root fingerprintを現実に用意できる独立担当者と、stage-1 receiptのartifact内検証を先に確立する。次に、bundle verifierを含む非認可runbookを独立監査する。人手レビュー・clock/frame対応・prediction-linkの実データがない限り、claim-linked scoringは起動しない。探索的簡易回答が届いたら正式レビューと混ぜずローカルで項目対応を確認する。
