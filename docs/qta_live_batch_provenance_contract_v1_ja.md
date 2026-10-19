# QTA processor batch の来歴・スコア実行契約 v1（設計案）

## 目的と対象経路

QTA直接faithfulness採点では、署名やmanifestに書かれたbatch hashと、実際にモデルへ渡したtensor bytesが一致する必要がある。現行 `qta_nine_branch_plan.py` は一つの処理済みmodel inputに対して9種類の介入枝を定義する。各recordには一つのbase processor batchがあり、その同じ入力を `baseline/target/sham × full/evidence_ablation/control_ablation` の枝で使い回す。

別経路の `nextgqa_atomic_batch_freeze.py` / `nextgqa_claim_mask_scoring.py` は、一つのrecordについて `full`, `claim_evidence_black`, `temporal_control_black` の3種類のprocessor入力を作る。固定3 recordの9 batchをCPUへsnapshotし、同じdigestをpre-forward・CUDA転送後に再確認する。こちらの3×3入力条件を、QTAの9 intervention branchesと混同してはならない。

## QTA用digest manifestの提案

将来の署名対象 `qta_processor_batch_digest_manifest_v1` は、canonical UTF-8 JSONとし、duplicate key・NaN・未知fieldを拒否する。

- `schema_version`, `purpose`, `status`, `execution_authorization`。
- 3つの不透明な `record_token` を固定順で保持し、各行につき**1つの** `base_processor_batch_sha256` を記録する。動画、質問、回答、ニックネーム、絶対パスは含めない。
- 各rowにprocessor/tokenizer/runtime pin、sampler設定、prompt/input-id binding digest、`video_grid_thw`、およびbase batch digestを対応付ける。
- 3つのbase batch digestは `nextgqa_atomic_batch_freeze.batch_sha256()` のtyped-batch規則で得る。strided・非quantized・非complex・有限tensorを必須とし、各tensorのdtype・shape・連続CPU bytesを含める。float16/BF16はdtypeを保持し、暗黙castはしない。
- 9枝それぞれのtarget/sham/evidence/control位置・介入設定はprocessor batch digestに含めず、別のphase-2 plan descriptorとそのhashで束縛する。全9枝は同じrowの同じbase batch digestを参照する。
- manifest全体はphase receiptの正確なartifact bytesとして署名・外部pinする。候補生成器自身のhashを同じcallerがpinとして再利用するだけでは、外部認証と扱わない。

現行phase receiptの `record_bindings` は既に3行各1件の `processor_batch_sha256` を保持する。pre-baselineは各行の `baseline_full` を含む3 forwards、post-baselineは残り8 intervention branchesを予定する。role `full_video_processor_batches` はopaque bytesであり、中のschemaやper-row digestを検査しないため、新role `processor_batch_digest_manifest` と新しいpre-baseline receipt domain/schema v2が必要になる。旧role/verifierと混在させず、既存receiptの再解釈や遡及適用はしない。manifestに生tensorは含めず、固定source動画・質問・sampler・processorから各rowの同一base batchを再生成できることが前提となる。

`nextgqa_atomic_batch_freeze` の3×3入力条件はこのmanifestに入れない。両経路を将来統合する場合は、processor入力条件軸と9介入枝軸の交差およびforward数を明記した別protocol・別receiptが必要である。

## 将来のQTA one-shot scorerの順序

1. model import/load前にtrusted rootと外部cohort pinを取得し、Stage-1 source/custody/release、人手claim判断、全15件のplayer-clock/decoder対応、baseline `p/q` と独立prediction-link判断、protocol・code・model・tokenizer pinを検証する。
2. 3行それぞれのbase inputを固定sampler/processorから一度だけ構築し、CPU `FrozenBatch` snapshotを作る。3件すべてのdigestが外部署名済manifestと一致しなければ、分母3のまま中断する。
3. Stage-2とprediction-link外部認証を最終確認し、成功時のみ同じ3 batch objectsを保持してmodel weightsをloadする。gate callbackは具体validatorを呼び、caller作成のsuccess booleanを受け取らない。
4. 9枝ごとに同じ対応rowのbatch key・tensor hashを照合し、固定済みintervention descriptorを適用する。forward失敗、digest mismatch、memory guard超過、nonfinite logitでは全cohortをincompleteとし、部分結果を効果推定値として扱わない。
5. 実行receiptにはmanifest/source/protocol/scorer/adapter/model/runtime hashes、要求/完了row・forward数、失敗コード、効果量と事前固定した不確実性計算だけを記録する。質問、選択肢、claim、動画、個別scoreは公開しない。

`nextgqa_atomic_batch_freeze.py` と `nextgqa_claim_mask_scoring.py` は別の3×3 scorer用構成要素である。いずれも上記のtrust root・人手・時計・Stage-2を束ねたproduction callerではない。成功fixtureや構造診断からmodel-load permissionを導かない。

## 次の検証

schemaをコード化する前に、manifestと既存phase receipt/planを結ぶ実callerのrow/batch対応を独立確認する。採用時は合成tensorで同値round-trip、key/row順置換、dtype/shape/byte tamper、nonfinite、未知tensor型、署名hash不一致、pre-forward failureをテストする。9枝が同じ3 base batchを使うことと、別の3×3 claim-mask scorerとの境界を回帰テストで固定する。実レビュー・外部root・15/15時計対応がない状態で実動画をscoreしない。
