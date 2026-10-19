# QTA processor batch の来歴・スコア実行契約 v1（設計案）

## 目的

QTAの直接faithfulness採点では、署名やmanifestに書かれたbatch hashと、実際にモデルへ渡したtensor bytesが一致する必要がある。現行の
`nextgqa_atomic_batch_freeze.batch_sha256()` は、mappingのキー、入れ子、tensorのdtype・shape・連続化後bytesをストリームhashする。さらに
`freeze_all_batches()` は3行×3条件のbatchをCPU上へsnapshotし、期待hashとの一致を確認してから `FrozenBatch` を返す。
`score_preflighted_triplets()` は9個のsnapshotを最初のforward前に再hashし、GPU転送後も各forward前に再hashする。

このため、同じtensorを永続化して別プロセスで復元する方式は採らず、**署名対象は再現用の入力recipeとbatch digest manifest、採点プロセスはそのrecipeから再生成したsnapshotを同一プロセス内で凍結して使う**方式を提案する。pickleや生tensorの巨大な重複保存を避ける。`.cpu().clone()`によるsnapshotと、各forward前のGPU転送で実行中のRAM/VRAM使用量は増えるため、実データ実装では全9 batchの実測peakを含む容量見積もりが必要となる。

## digest manifest の提案形式

将来の署名対象 `qta_processor_batch_digest_manifest_v1` は、canonical UTF-8 JSONとし、duplicate key・NaN・未知fieldを拒否する。

- `schema_version`, `purpose`, `status`, `execution_authorization`。
- 3つの不透明な `record_token` を固定順で保持する。動画、質問、回答、ニックネーム、絶対パスは含めない。
- 3条件 `full`, `claim_evidence_black`, `temporal_control_black` ごとに `batch_sha256`、processor/tokenizer/runtime pin、sampler設定、prompt/input-id binding digest、`video_grid_thw`、evidence/control temporal group indexを保持する。
- 9 batch digestは `nextgqa_typed_batch_stream_v1` により得る。batch内のtensor keyはstring、tensorはstrided・非quantized・非complex・有限値を必須とし、key orderはUTF-8文字列昇順。各tensorのdtypeとshapeを含む連続CPU bytesを使う。float16/BF16はdtypeを保持し、暗黙のcastはしない。
- manifest全体は、`qta_phase_freeze_receipts.py` が署名する正確なartifact bytesとして外部pinに含める。候補生成器自身の出力hashを同じcallerがpinとして再利用するだけでは、外部認証と扱わない。

この形式は設計案であり、現行のStage-1 `full_video_processor_batches.bin` schemaを置き換えてはいない。現在のphase receiptはartifact role集合を完全一致で検査し、pre-baseline署名domainもv1固定である。採用時は新role `processor_batch_digest_manifest` と新しいpre-baseline receipt domain/schema v2を追加し、旧 `full_video_processor_batches` roleを受け入れるv1 verifierと混在させない。既存署名receiptを読み替えたり、既存コホートに遡及適用したりしない。manifestに生tensorは含めず、固定source動画・質問・sampler・processor設定から同じbatchを再構成できることが前提となる。

## 将来のone-shot scorerの順序

1. model import/load前に、trusted rootと外部cohort pinを取得し、Stage-1 source/custody/release、人手claim判断、全15件のplayer-clock/decoder対応、baseline `p/q` と独立prediction-link判断、protocol・code・model・tokenizer pinを全行について検証する。
2. 3行×3条件すべての入力を同じ固定sampler/processorで一度だけ構築し、CPU `FrozenBatch` snapshotを作る。9件すべての `batch_sha256` が外部署名済manifestと一致しなければ、分母3のまま中断する。
3. Stage-2とprediction-linkの外部認証を最終確認し、成功時だけ固定batch objectsを保持したままmodel weightsをロードする。gate callbackは具体的validatorを呼び、caller作成のsuccess booleanを受け取らない。
4. forwardごとにbatch key、全tensor hash、条件・record順を照合する。CUDA転送後のdigestが変わる、memory guardを割る、forward失敗、nonfinite logitが起きた場合は全cohortをincompleteとし、部分結果を有効な効果推定値として扱わない。
5. 実行receiptにはmanifest/source/protocol/scorer/adapter/model/runtime hashes、要求/完了row・forward数、失敗コード、効果量と事前固定した不確実性計算だけを記録する。質問、選択肢、claim、動画、個別scoreは公開しない。

現行 `nextgqa_atomic_batch_freeze.py` と `nextgqa_claim_mask_scoring.py` はsnapshot/hash/scoring mathの構成要素を実装するが、上記のtrust root・人手・時計・Stage-2を束ねたproduction callerではない。成功fixtureや構造診断からmodel-load permissionを導かない。

## 次の検証

schemaをコード化する前に、既存Stage-1 role inventoryとレビュー作業手順にdigest manifest v1を採用できるかを独立確認する。採用する場合は、合成tensorで同値round-trip、key/row/condition順の置換、dtype/shape/byte tamper、nonfinite、未知tensor型、署名済hash不一致、pre-forward failureのテストを追加する。実レビュー・外部root・15/15時計対応がない状態で実動画をscoreしない。
