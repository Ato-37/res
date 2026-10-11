# LocalizationHeads型・教師なしhead選定の実行結果

## 目的

これまでのhead選定(`head_grounding_scan.py`)は、正解根拠区間との重なり(hit_rate)を選定基準そのものに使っていた。これはheld-out検証をしていても「正解ラベルを選定にも評価にも使う」という構造上、過学習のリスクが残る。LocalizationHeads(CVPR2025 Highlight)の本来の手法(選定段階で正解ラベルを一切使わない: Criterion-1 value-based elbow/chord + Criterion-2 spatial entropy、動画版としてtemporal entropyに読み替え)を実装し、(a) layer=6/head=16がこの教師なし基準でも候補として浮上するか、(b) 教師なし基準で選ばれた候補が実際にground truthに対してどの程度当たるか、を確認する。

## 設定

- スクリプト: `scripts/head_selection_unsupervised.py`(commit `06433e4`、device不一致バグ修正後)
- コマンド:
  ```
  python scripts/head_selection_unsupervised.py \
      --samples-file data/nextgqa_samples_big.json \
      --eval-samples-file data/nextgqa_samples_big_heldout.json
  ```
- Criterion-1(選定用、正解ラベル不使用): 各(layer, head)の「最終トークンから動画トークン全体へのattention総量」を`data/nextgqa_samples_big.json`(58サンプル処理成功、2件は動画読み込みエラーでスキップ)で平均し、784head分を降順ソート→chord method(始点・終点を結ぶ弦から最も離れた点をelbowとする)で足切り(`--min-keep`未指定時デフォルト10)。
- Criterion-2(選定用、正解ラベル不使用): Criterion-1生存headのうち、時間チャンク(num_frames=8→t=4チャンク)分布のShannon entropyが低い(集中している)順にランク。
- 評価(ここでのみ正解ラベル使用): 上位20候補+layer=6/head=16を、`data/nextgqa_samples_big_heldout.json`(n=22処理成功)でhit_rate評価。

## 結果

**Criterion-1: 784head中88headが生存**(elbow法による足切り)。

**layer=6/head=16はCriterion-1の時点で脱落**: attention magnitude=0.0410で784head中390位。足切りラインの88位よりはるかに下で、Criterion-2(entropy)の土俵にすら乗らなかった。

**教師なし基準トップ20候補**(Criterion-1生存者をentropy昇順でランク):

| rank | layer | head | attn_mag | temporal_entropy |
|---|---|---|---|---|
| 1 | 9 | 24 | 0.3229 | 1.0174 |
| 2 | 9 | 8 | 0.2248 | 1.0477 |
| 3 | 9 | 21 | 0.3075 | 1.1151 |
| 4 | 9 | 1 | 0.2441 | 1.1209 |
| 5 | 9 | 23 | 0.2857 | 1.1479 |
| ... | | | | |
| 20 | 9 | 15 | 0.2242 | 1.2937 |

**layer=9が上位20のうち11個を占める**(head 24,8,21,1,23,25,2,4,19,22,15)。次点はlayer=27(3個)、layer=26(2個)。

**evaluation(held-out n=22、ground truth使用はここのみ)**:

| layer | head | hit_rate |
|---|---|---|
| 6 | 16(参考、Criterion-1脱落済) | **45.45%** |
| 20 | 19 | 36.36%(=ランダム基準と同値) |
| 9 | 22 | 36.36%(=ランダム基準と同値) |
| 9 | 4 | 31.82% |
| (以下、教師なしトップ20の残り) | | 22.73%〜27.27%(ランダム基準36.36%を下回る) |

ランダム基準は36.36%。**教師なしトップ20のうち、ランダム基準を明確に上回った候補はゼロ**(最良でも同値)。一方layer=6/head=16は45.45%で唯一明確にランダムを上回った。

## 解釈

- **この実装での教師なし基準は、layer=6/head=16を再発見できなかった**。両者の選び方(ground-truth hit_rateベース vs attention形状ベース)は、少なくとも今回の実装では一致しない。
- **教師なし基準が選んだ候補は、実際のgroundingでは機能していない**(ランダム基準を上回らない)。これは「教師なし選定は無効」という結論と、「今回の時間軸への読み替え方に問題がある」という2つの解釈がありうる。以下の理由で後者の可能性を優先的に疑うべき:
  1. **temporal entropyの絶対値が理論上の最大値に近い**: t=4チャンク(num_frames=8による)での一様分布のエントロピーはln(4)≈1.386。トップ20候補のentropy値(1.02〜1.29)はこれにかなり近く、「特定の時間帯に強く集中している」とは言い難い、ほぼ一様に近い分布。LocalizationHeads本来の空間版(パッチ数がはるかに多い画像)と比べ、t=4という粗い時間解像度ではentropy基準の識別力がそもそも乏しい可能性が高い。
  2. **layer=9が上位20の過半数を占める偏り**は、「多くの多様なheadが偶然似た性質を持つ」というより、「layer=9特有の何らかの定型的パターン(attention sinkのような、内容に依らない固定的注意)」を拾っている可能性がある。これは本スクリプトでは未検証(`head_grounding_scan.py`にあった`sink_fraction`診断をこのスクリプトには実装していないため)。
- **したがって現時点では「LocalizationHeads型の教師なし選定は動画の時間的groundingには使えない」と断定するのは時期尚早**。まず(a) sink_fraction相当の診断を追加してlayer=9の偏りの正体を確認する、(b) num_framesを増やしてtの解像度を上げ、entropy基準の識別力が上がるか確認する、の2点を先にやるべき。

## 次のアクション

1. ~~`head_selection_unsupervised.py`に`sink_fraction`診断を追加し、layer=9の候補群が単なるsink headでないか確認する。~~ **完了・確認済み(下記追記参照)**
2. `--num-frames`を16程度に増やし(t=8チャンクに増加)、temporal entropy基準の識別力が改善するか確認する。
3. 上記2点をクリアしてもなお教師なし基準がlayer=6/head=16を再発見できない場合、これは「ground-truth教師ありでしか見つからない候補」という誠実な限界として卒論に明記する(LocalizationHeadsの前提が動画の時間軸には単純には転用できないという、それ自体が一つの知見になりうる)。

## sink_fraction診断で確認(2026-09-30、commit `12fa0a8`)

`head_grounding_scan.py`と同じsink_fraction診断(peak_chunkが全サンプルで同じ値になる割合)を追加して再実行した。

**結果、上位20候補のうち11個がsink_fraction≥85%(多くは96〜100%)だった**:

| rank | layer | head | temporal_entropy | sink_fraction |
|---|---|---|---|---|
| 1 | 9 | 24 | 1.0174 | 98.28% |
| 2 | 9 | 8 | 1.0477 | 100.00% |
| 3 | 9 | 21 | 1.1151 | 100.00% |
| 4 | 9 | 1 | 1.1209 | 96.55% |
| 5 | 9 | 23 | 1.1479 | 100.00% |
| 6 | 27 | 9 | 1.2076 | 100.00% |
| ...(以下、6個がsink_fraction≥85%) | | | | |

**sink_fraction<85%だった残り9個(layer=27/17, layer=8/0, layer=26/2, layer=20/19, layer=19/6, layer=6/11, layer=9/25,4,22)も、held-out評価でランダム基準(36.36%)を上回れていない**(最良でも同値36.36%、大半は27〜32%)。

**結論**: 教師なし基準(attention magnitude elbow + temporal entropy)が候補として拾ったheadの過半数は、内容に依らず常に同じ時間帯を見ているattention sinkであり、真の時間的localizationではなかった。低entropyだけではsinkと本物の集中を区別できないことが実証された。sink_fractionでフィルタしても残る候補(sink_fraction<85%の9個)すら実際のgroundingには機能していないため、**sink除去だけでは教師なし基準は救えない**。

**重要な参照**: LocalizationHeads原論文の要約(Claude Docsアーティファクト)には「残ったheadのうち...(下部行フォーカスのフィルタも併用)」という記述があり、原論文自身も画像ドメインで同種の退化パターン(画像下部に固定で注意を向けるhead)を別途フィルタしていたことが分かる。つまり**「entropy基準単体ではsink的な退化パターンを弾けない」という問題は原論文の設計にも織り込み済みであり、今回の動画版でも同様の追加フィルタが必要**という、方法論としては筋の通った次のステップが見えている。

**次のアクション(更新、2026-10-01時点で2は完了)**:
1. `--num-frames`を増やして(例: 16、t=8チャンク)temporal entropy基準の識別力が改善するか確認する(未着手)。
2. ~~sink_fraction自体を選定基準に組み込む~~ **完了・実施済み**。`--max-sink-fraction`フィルタを実装して再実行したところ、sink除去後の上位候補(layer=0系が多数)がhit_rateで好成績を示したが、これ自体が別の交絡(固定チャンクへの位置バイアス)であったことが判明した。詳細・結論は[2026-09-30_fixed_chunk_baseline_confound.md](2026-09-30_fixed_chunk_baseline_confound.md)を参照(このファイルの続きとして読むこと)。
3. 上記を試してもlayer=6/head=16が教師なし基準で再発見できない場合、「この卒研のパイプラインでは教師なし基準が機能しない」という誠実な限界として卒論に明記し、ground-truth教師ありの選定(現状の方法)を正当化する根拠として使う。
