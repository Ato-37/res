# 簡易動画レビューの回答対応付け（2026-10-19）


独立したUI監査で、簡易チェックのJSON回答は`item_number`だけを持ち、質問順が後から変わると元の動画・質問へ確実に対応付けられない点が分かった。開始までの操作を増やさず、画面には見せない決定的な20桁opaque `item_id`を各回答へ含めるようにした。raw review token・動画ID・質問本文は回答JSONに入れず、IDはローカルHTML内の固定項目との照合にだけ使う。JSON schemaをv2へ更新し、IDは本人性や視聴完了の証拠ではないとlimitationsに明記した。更新したローカルartifactは`data/private_nextgqa/quick_review_ja_video_only_20261019_itemid_v2.html`で、案内ページのリンクも同ファイルへ切り替えた。

検証: item IDの決定性・異なるtoken間の分離・raw token非出力を確認するunit testを追加。簡易レビューsuite **4/4成功**、Python compileと`git diff --check`成功。実画面レビュー・回答受領は未実施であり、artifact/回答JSONはローカル限定。簡易印象記録として後から項目を照合しやすくしただけで、正式レビューに加算しない。

独立GPU候補監査と現地GPU確認では、RTX 5070 Ti/CUDA 12.8は個別許可コマンドで利用可能だった。一方、未閉鎖の人手不要GPU候補は見つからず、直近のquestion-swap (80動画、320/320 forward) とmask-specificity (147動画中140適格、868/868 forward) は完了済み。これらの再実行は重複・post-selectionとなる。直接faithfulnessは引き続きclaim review **0/3**、frame correspondence **0/15**、prediction-link **0/3**、direct score **0/3**、実`ΔF`/CI/testなし。回答JSONも未確認。よってこの追記ではGPU forwardを実施しなかった。

次は利用者が探索的JSONを保存した場合に限り、項目IDを用いて簡易所見をローカルで照合し、正式レビュー記録とは分離したまま報告する。別枝では、stage-1 receiptのartifact内検証と、外部pinを受け取るproduction caller契約の未認可検証を続ける。

## 再生成先の整合（追記）

独立UI監査で、案内ページはtraceable v2 artifactを指す一方、builderの既定出力先は旧v1名のままと確認した。既定先を`quick_review_ja_video_only_20261019_itemid_v2.html`へ揃え、誤った版を再生成しないためのunit testを追加した。既存のprivate artifactは上書きせず維持する。関連テスト5/5成功、`py_compile`と対象差分の`git diff --check`も成功した。これは操作・保存先の整合修正であり、レビュー実施や直接faithfulness測定ではない。
