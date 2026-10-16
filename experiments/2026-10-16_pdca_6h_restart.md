# Fresh six-hour PDCA — 2026-10-16

Requested work window: **6 hours**. The message timestamp was unavailable, so
accounting starts at the observed clock time **2026-10-16 14:43:15 JST**
(**05:43:15 UTC**) and ends at **20:43:15 JST / 11:43:15 UTC**. This is a new
window; no elapsed time or unfinished work from the earlier paused PDCA is
counted as completed in this window. At the first checkpoint (14:50:06 JST),
**6m51s elapsed / 5h53m09s remained**.

## Plan and starting state

At start, `HEAD=64d8592` (`Log paused two-hour PDCA checkpoint`), branch
`main...origin/main [ahead 15]`; the origin tracking ref is cached, not freshly
fetched. The worktree already contained uncommitted edits to `.gitignore` and
four claim-review scripts, plus an untracked private-output helper/test, the
earlier PDCA log, and untracked local datasets/sample manifests. These were
pre-existing at the start of this fresh window. User data was not staged,
committed, or modified. Changes to the output helper were audited again in
this window and used to persist the new private review materials below.

Normal sandbox resource check: 15 GiB WSL-visible RAM, 13 GiB available; disk
has 892 GiB available. `.venv` is PyTorch `2.11.0+cu128`, CUDA 12.8, but the
ordinary sandbox reports no visible CUDA device (`nvidia-smi`: GPU access
blocked by operating system). Per-command elevated execution is the known WSL
boundary and was used only for the bounded local claim-generation runs below.
The checkpoint was local/offline Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute.

The user confirmed the older review JSON/HTML were not saved. The previous
claim text is therefore unrecoverable; no regenerated text is represented as
the old text. Rights status is unverified. The user's local-only protected-data
authorization was followed: all semantic/model-derived artifacts remain local
in ignored `data/private_nextgqa/` with directory mode 0700 and file mode 0600;
nothing was pushed or externally shared.

## Cycle 1 — fresh claim-review candidate set

Independently reran the outcome-blind CPU cohort selector with the current
sample-manifest exclusions, three videos, 16 frames, and the fixed 50 ms
uniform-grid timing tolerance. It scanned 3 candidates; all **3/3** decoded
and passed timing and interval/control availability, with **0** decode or
timing failures. The deterministic cohort hash is
`32767fd62e6c63bc0b54ba8457ebaa181911b7e614975736b535e63988ab4ec8`, matching
the current-window generation packet. It differs from the historical cohort
hash `bbee8d470655abcd10a6912d69434768c1584ffca703d4b0a26eb9d7250accc5`.
Thus this is a new review set, not a restoration of the old one.

### GPU smoke and bounded generation

Before the three-item run, ran the existing fixed prompt on one separate,
cohort-excluded NExT-GQA video (16 frames). The smoke completed **1/1** video;
claim text was present **1/1**, answer letter **1/1**, and valid self-reported
frame indices **0/1**. It returned `status=partial`, peak allocated VRAM
**7.436 GiB**, and **13.716 s** reported generation elapsed time (the script's
timer excludes model loading). Its private artifact SHA-256 is
`ae14845349357780132b0a6388051286430cf0ca35e3511ebb5c1d41ffff933a`.

The fixed prompt was then run on the frozen three-video cohort, 16 frames per
video. Requested/effective was **3/3**, all three had parsed answer and claim
text (**3/3**), and none had a valid self-reported evidence-index list
(**0/3**), so the generator correctly returned `partial` with zero fully
structured A/C/F records. Peak allocated VRAM was **7.585 GiB** and reported
generation elapsed time **17.777 s**, excluding model loading. The private
packet is `data/private_nextgqa/claim_generation_freshcohort_20261016_02.json`,
SHA-256 `2fc3ed30ffb8409d199c78c5594cc105d0db9c413b3e712819f96fdd7dfd447a`.
No prompt-format search or score computation was performed. The 0/3 frame-index
rate is an engineering-format failure; it does not establish claim quality.

The local source-lineage audit matched **3/3** records to the dataset/video
metadata and media hashes. Its private manifest SHA-256 is
`43a441615ed8cff74923f516fa5f87869a26ec7e52690d6213fd1ba7e2ffe5d9`. The
fixed pre-review policy `nextgqa_claim_review_v1` was applied to build a
three-row blinded mapping worksheet (`draft_not_reviewed`) and a local offline
review page. Independent audit verified matching cohort hashes across
generation, lineage, worksheet and HTML; worksheet `review_completed=false`,
`execution_authorization=not_granted`; all review fields remain blank. Static
checks found no gold answer, gold interval, raw generation, or absolute local
repository path in the HTML. The output hashes are:

| Artifact | SHA-256 | Mode |
|---|---|---:|
| Private generation packet | `2fc3ed30ffb8409d199c78c5594cc105d0db9c413b3e712819f6dd7dfd447a` | 0600 |
| Source-lineage manifest | `43a441615ed8cff74923f516fa5f87869a26ec7e52690d6213fd1ba7e2ffe5d9` | 0600 |
| Blinded worksheet | `3c63b64a165a6e139d8eea5bf66dbfe0f6114c81513c6e8e2f36664a05edc575` | 0600 |
| Offline review HTML | `b490b30f40c87825fb8a96996ad87c7d722644d5b3972542d70f269c92f35243` | 0600 |

The HTML review interaction was not opened in a browser: Chromium, Chrome and
Firefox are absent in this environment. JavaScript syntax check via Node 24.18
passed. Static and schema checks do not establish browser playback, WebCrypto
behavior, actual human review, semantic claim support, or annotation-clock
calibration. The selected source videos are 3.10–6.42 MB, below the review UI's
64 MB file ceiling.

### Outcome-blind source-clock extent check

Ran the existing aggregate-only decoder-clock audit against the three
lineage-hash-bound source videos; requested/effective **3/3**, decode failures
**0**. Each has 32 dataset `frame2time` points. Nearest-frame2time-to-decoded-PTS
median residuals were **0–9.2 ms**, maxima **0–16.7 ms**, and there were no
points outside the decoded PTS range. Dataset durations differed from the
audit's last-PTS-plus-prior-interval extent by **0–0.439 s**. Private aggregate
artifact `/tmp/nextgqa_source_clock_freshcohort_20261016_06.json`, SHA-256
`5605944e78d32b0168aa4eea2c56984ebb9f94a3a16070cead5f3fe5dcc70a4f`, mode
0600. This compares dataset timing metadata with decoder PTS; it does not
identify the annotation player's origin, edits, seek behavior, drift, or
display-to-decoder mapping, and is not a calibration pass.

### Faithfulness progress at 14:50 JST

- **Direct evidence: `未実施` — 0/3 human-reviewed, 0/3 scored, 0 claim-specific
  effect estimates** for Qwen2.5-VL-7B-Instruct / NExT-GQA / human-mapped claim
  interval mask versus equal-count temporal control / fixed gold-vs-best-foil
  log-margin. No effect, CI, test, or multiplicity adjustment exists.
- **Engineering diagnostics only:** cohort freeze and timing feasibility 3/3;
  model claim text 3/3; metadata/media lineage 3/3; review worksheet/page 3/3;
  frame2time-to-decoder PTS residuals above; full CPU suite 893/893; Python
  compilation, Node JavaScript syntax check, and `git diff --check` passed.
  None is faithfulness evidence.
- **Open gates:** independent human claim-to-full-video review is 0/3; browser
  playback was not exercised; frame2time/PTS checks do not calibrate the
  annotation player; inference target/dependence and a justified SESOI remain
  open; the v13 gate still reports `execution_authorization=not_granted`.
  Local-only data/model operations are authorized, while rights remain
  unverified and redistribution is not authorized.
- **Next action:** continue independent audit of the new bundle and protocol;
  reproduce the private geometric measurements and review-output invariants;
  inspect unresolved timing and inference assumptions without accessing score
  outcomes. Do not score claims until a real independent human review and the
  remaining inference/timing gates are resolved. Do not reopen head/layer/alpha
  search.

No code, logs, data, or result artifacts have been committed or pushed in this
fresh window so far. The private results remain local.

## 14:52 JST checkpoint — source duplication and inference target review

Elapsed **9m18s / 6h**; deadline remains **20:43:15 JST** (**5h50m42s
remain**). Independently counted SHA-256 identities for all **85** currently
eligible local candidate media files: **85 unique hashes**, **0 exact duplicate
groups**, maximum identical-file multiplicity 1; their total size is
360,029,896 bytes. The local NExT-GQA video-map file has 10,000 scalar-string
values and no nested creator/event/source-ID fields; the dataset directory
inventory contains no separate local creator/event/series/source-lineage table.
Exact-file uniqueness only rules out byte-identical duplicate clips in this
candidate pool. It does not establish independent recording events, creators,
or upstream source videos; source clustering remains unresolved.

The independent outcome-blind inference review found no defensible confirmatory
population/uncertainty target or SESOI for the current three-item deterministic
feasibility set. The existing two-sided Student-t rule would require an
identified model target and defensible video-level independence/distribution
assumptions; three observations do not establish those assumptions. A finite
three-item descriptive target could be defined, but would require a separately
versioned decision framework before scoring and would not generalize beyond
those exact three records. Do not transfer the prior head-study +0.25 nat
threshold. Keep the current schema-v13 status exploratory/no-go, preserve its
single primary contrast and predeclared Holm-adjusted secondary families, and
do not switch thresholds, subgroups, or estimands after scores.

### Faithfulness progress at 14:52 JST

- **Direct evaluation: `未実施` — still 0/3 reviewed, 0/3 scored, 0 effects**
  on Qwen2.5-VL-7B-Instruct / NExT-GQA / reviewed claim-interval mask versus
  equal-count temporal control / gold-vs-best-foil log-margin.
- **Auxiliary diagnostics only:** 85/85 exact media hashes distinct; source
  lineage metadata needed to assess creator/event clustering is absent from
  the local dataset map; the independent design review confirmed the
  inferential target and SESOI are not presently defensible. No score or
  uncertainty interval was calculated.
- **Unresolved gates:** human review, production player-to-decoder calibration,
  source-event cluster audit, inferential target/method, and SESOI. Browser
  behavior remains untested. The renderer, worksheet, and cohort are local and
  unreleased.
- **Next action:** assess whether any official/public provenance mapping can
  supply source-event clusters without exposing identifiers; otherwise record
  the limit and maintain a finite-cohort descriptive option only as a future,
  pre-score protocol revision. Continue only outcome-blind protocol and
  reproducibility work while the user-specified window remains.

The exact-media uniqueness pass has no inferential meaning, and neither the
design audit nor the clock/geometric checks add direct faithfulness evidence.

## 15:12 JST checkpoint — V3 blinded bundle and custody verification

Elapsed **29m37s / 6h**; deadline remains **20:43:15 JST** (**5h30m23s
remain**). This cycle was justified by the independent audit finding that V2's
HTML still embedded raw media hashes and deterministic IDs even though they
were not shown as visible labels. V2 is therefore not suitable for blind human
review.

### Implementation and artifact verification

Created a new V3 worksheet with freshly generated 24-character random review
tokens, leaving the source worksheet and prior outputs untouched. The reviewer
HTML embeds only the token, question, claim, and frozen policy; it matches the
selected media by `<token>.mp4` filename and exports only token-indexed
judgments/intervals plus reviewer-local attestation. Stable dataset/video/claim
IDs, media SHA values, paths, answers, QA seed intervals, and scores are absent
from the new page. The opaque token-to-row mapping and exact source/staged
media SHA values are confined to a separate mode-0600 custody manifest; the
staged directory is mode 0700 and each video copy is mode 0600.

For this real local three-video feasibility cohort, staged **3/3** token-named
copies. An independent custody audit reread the worksheet, lineage manifest,
custody manifest, and staged file bytes; all expected tokens, file names,
worksheet media hashes, copied bytes, and permissions matched. Manifest hash
was independently pinned at creation and checked by the audit. V3 private
artifact hashes: policy `18c322aec65fdfd30c4b11fce280781a867852e789fe33bac77f4da6e55174f4`;
worksheet `f0f67cdf21ccf88e524a9f9473aa30801e0ad49124bf10b8f6d1c78755d4accf`;
review page `cda8c35b6d339225b9dfa19a036b21c3c1dad55949305ab1cc6ed3172873b004`;
private custody manifest `b3cb5ca3107e837daa9da76d7e7cef12f3bd1281aec7024121bf9db4e76e5bb1`.
These are local-only derived/protected artifacts and must not be published.

The V3 handoff audit now validates the anonymous review schema, token binding,
policy/time consistency, millisecond interval shape and bounds against the
reviewer's player-reported duration, plus custody-manifest and actual staged
media hashes. It returns only aggregate status/failure codes and always leaves
semantic authentication false and scoring authorization `not_granted`. Synthetic
tests include payload/hash tampering, token/path tampering, invalid intervals,
custody media-byte tampering, file modes, and staging's opaque names. After
finding a V3 JavaScript closing-brace error during an explicit syntax check, the
exporter was corrected; the generated page passes Node 24.18 syntax checking.
Static checks confirm offline CSP and absence of stable source values. Full
repository Python suite: **897/897 passed in 9.774 seconds**; `git diff --check`
passed.

### Faithfulness progress at 15:12 JST

- **Direct faithfulness evidence remains none:** **0/3 human-reviewed, 0/3
  scored, 0 effect estimates, no CI/test** for Qwen2.5-VL-7B-Instruct / NExT-GQA /
  human-reviewed claim-interval mask vs equal-count temporal control / fixed
  gold-vs-best-foil log-margin. This V3 engineering repair does not increase
  evaluation sample size or establish claim support.
- **Engineering gate advanced:** a fresh real-media V3 handoff is now
  source-byte/hash-verifiable by a custodian and hides stable source IDs in the
  reviewer page. This establishes packaging and byte integrity, not an
  independent person's identity, actual viewing, semantic claim validity,
  browser playback, or source-player time calibration.
- **Remaining gates:** independent human review (0/3); browser interaction on a
  real browser; source annotation-player to decoder-PTS calibration; defensible
  estimand/uncertainty target and SESOI; and a valid disjoint equal-count
  temporal control. Rights remain unverified. No score was calculated and no
  GPU work is warranted until these gates permit a discriminating run.
- **Next action:** independent audit the V3 implementation and identify a
  practical local browser handoff without exposing identifiers; then continue
  outcome-blind review of timing, controls, and inference protocol. Keep all
  protected materials local.

The reviewer page is still **not** a completed review. The old V2 page was not
replaced in place; only the newly generated V3 page is eligible for a future
review, subject to the independent audit and human execution.

## 14:59 JST checkpoint — provenance route and blind-review repair

Elapsed **16m27s / 6h**; the fixed deadline is **20:43:15 JST**, with
**5h43m33s remaining**. A separate source/design audit and primary-source
inspection found a feasible *structural* route for grouping possible source
accounts: the official NExT-QA README links its video IDs to VidOR paths via
`map_vid_vidorID.json`; the VidOR project page says its `video_id` is the
YFCC100M ID; YFCC100M's paper describes per-item owner metadata. This can at
best support a YFCC uploader-account cluster proxy, not independent creators,
shooting events, or source videos. No item IDs were sent to an external service,
no YFCC metadata was fetched, and no item-level uploader matches were made.
The local map inventory and the prior container-tag audit do not contain the
needed uploader field. Thus source clusters remain unresolved. Primary
references: [NExT-QA repository](https://github.com/doc-doc/next-qa),
[VidOR dataset page](https://xdshang.github.io/docs/vidor.html), and
[YFCC100M paper](https://arxiv.org/abs/1503.01817).

The independent reviewer audit identified that the first private review page
rendered dataset question IDs, video IDs, path keys, and relative source paths.
Those fields could let a reviewer cross-reference benchmark indexes and infer
gold answers or seed intervals, weakening the intended blind. No review had
started. I removed those identifiers from the page data and visible UI, made
the reviewer output omit the per-item source crosswalk, and revised the frozen
procedure as `nextgqa_claim_review_v2` so a separate custodian checks source
lineage and media bytes first, then supplies opaque local media copies. The
v2 policy SHA-256 is
`6a75812793d67b97196b05c5e83bb48a14f0da6fb7fee5a3143fd4dfb7f4d2e2`.
The new private worksheet is
`data/private_nextgqa/claim_mapping_freshcohort_20261016_12.json` (SHA-256
`7317484f09f18260994bb09d8af9a07eec78d11c53d21c6c97dee42a90d61344`); the
new offline review page is
`data/private_nextgqa/claim_review_freshcohort_20261016_13.html` (SHA-256
`0e57bcc63cd5e35a37d170330a0f6afa6e7977644e9025fc29813aac2a502c07`, mode
0600). It contains 3 records and a static scan found none of the source-ID/path,
gold-answer, seed-interval, raw-generation, opaque source ID, or displayed
media-hash fields. Node syntax validation and the same static scan pass. Three source videos were
byte-hash verified and copied under random opaque filenames into a private
0700 directory (each 0600, total 12,661,265 bytes). These copies and the page
remain local and ignored; they have not been opened by a human reviewer.

Updated the private handoff auditor to bind lineage through its separate
hash-checked manifest and worksheet, rather than duplicating identifying
lineage values in every reviewer record. Targeted builder/renderer/auditor
tests pass **25/25**; complete repository CPU suite passes **893/893** in
8.633 seconds; Python compilation and `git diff --check` pass. Browser media
playback, secure-context WebCrypto, and download behavior remain untested
(there is no installed browser). The 50 ms human-player seconds versus decoder
PTS mapping remains uncalibrated, and the v13 scoring gate is still no-go.

### Faithfulness progress at 14:59 JST

- **Direct faithfulness evidence: still 0/3 human-reviewed, 0/3 scored, and 0
  effects.** GPU work in this window is fixed-prompt claim preparation only;
  no answer-margin/faithfulness score was computed.
- **Protocol/reproducibility progress:** the reviewer page no longer exposes
  benchmark IDs/paths; policy is versioned and the private media handoff is
  prepared. This mitigates a blind-review design flaw, but is not evidence of
  claim support or model dependence.
- **Still blocking scoring:** actual independent human review; exact player to
  decoder-clock calibration; a justified estimand/inference target and SESOI;
  valid equal-count disjoint controls; source-account clusters unresolved; and
  the separate v13 execution gate. Rights status remains unverified, while
  the authorized work and its artifacts remain local-only.
- **Next action:** independently verify the new v2 page/artifacts and run all
  scoped tests, then continue outcome-blind protocol work. Do not compute scores
  before the unresolved review, clock, control, and inference decisions close.

## 15:29 JST checkpoint — V5 custody repair and claim-quality stop

This fresh six-hour window is accounted from **2026-10-16 06:24:47 UTC / 15:24:47 JST** to **12:24:47 UTC / 21:24:47 JST**. At this checkpoint, **4m20s** have elapsed and **5h55m40s** remain. The prior PDCA's deadline is not reused.

### Verified V5 handoff repair

An independent source audit found two residual boundaries: stale `loadedmetadata` events were not generation-bound, and the renderer API accepted missing custody. A follow-up audit confirms the fixes in `scripts/render_nextgqa_claim_review_html.py`: metadata is accepted only when the selection generation and current blob URL match; changing a file clears state, removes the source, and reloads the player; render requires custody and validates exact row keys, ordinal, stable source/staged hashes, byte size, token, salt, and digest. Staging now constructs custody before applying the production renderer's validation. The package copier is marked private and its audit precondition is explicit; the CLI independently runs `audit_custody`, binds the HTML to frozen worksheet/lineage/custody inputs, then copies.

Rebuilt a new V5 private worksheet, 3-media custody set, page, and isolated package from the frozen source packet under the exact final policy digest. Direct custody audit reports **0 failures / 3 records**. The package contains only one HTML file and 3 anonymous MP4 files, directory mode 0700 and files 0600. A separate Node crypto implementation recomputed all 3 salted digests successfully; extracted page JavaScript passes Node 24.18 syntax validation. Relevant tests pass **29/29**, the full repository suite passes **897/897** in 9.931s, Python compilation and `git diff --check` pass. Browser playback, WebCrypto behavior in an actual browser, and the metadata race remain untested dynamically.

### Exploratory claim-quality finding; V5 package not eligible for review

A new outcome-blind local format diagnostic found that the three frozen generated claim strings all contain a suspicious `addCriterion` marker and newline artifacts (**3/3 flagged**; claim lengths 20–35 characters). This is an exploratory quality flag, not a semantic judgment or faithfulness measurement. The existing generator checks only that a parsed claim is nonempty; its `structured` count therefore did not validate readable sentence quality. I moved the newly created package to `reviewer_package_v5_format_flagged` so it is not mistaken for a review-ready handoff. No human review began and no results were removed or rewritten.

The next scientifically sound step is to inspect the claim-generation/parser interface and add a pre-review quality gate, then use a newly frozen exploratory cohort/prompt if regeneration is warranted. Do not repair or replace these three claims after seeing them and treat them as confirmatory. Any new experiment must remain distinct from this flagged packet and preserve its provenance. Browser interaction is still a separate operational gate.

### Faithfulness progress

- **Direct faithfulness evidence: `未実施`** — 0/3 human-reviewed, 0/3 scored, 0 effects, no CI/test. Model/data/intervention/metric remain Qwen2.5-VL-7B-Instruct / local NExT-GQA / human-verified claim-interval mask versus equal-count temporal control / fixed gold-vs-best-foil log-margin; no comparison has been run.
- **Auxiliary engineering only:** GPU generation earlier in the broader work produced candidates but is not faithfulness evidence. Current V5 work establishes byte custody and safer rendering. The 3/3 marker scan is a format diagnostic only.
- **Unresolved/no-go gates:** human review, usable claim text, actual browser validation, annotation-player-to-decoder clock calibration, disjoint equal-count control, and a defensible inferential target/SESOI. Rights status remains unverified; all materials remain local.
- **Next action:** diagnose the generator output/claim parser without exposing protected text; repair validation before any new handoff; continue independent clock/control/inference audit. Keep scoring disabled.

## 15:35 JST checkpoint — fail-closed claim syntax and GPU prompt smoke

The previous checkpoint said 4m20s elapsed. Rechecking the clock at **15:35:02 JST** gives **10m15s elapsed / 5h49m45s remaining** for this fresh window (deadline **21:24:47 JST**).

### Claim format gate and real-data smoke

Added shared format-only validation in `scripts/claim_text_quality.py`, applied in the generation parser, worksheet builder, and HTML renderer. It rejects empty/outer-whitespace/multiline claims and obvious template/special-token artifacts; it does not evaluate whether a claim is true or supported. Independent audit found test cases initially conflated marker and newline failures; I separated them and now assert exact codes for pure multiline and marker-only input. The builder directly rejects the existing frozen packet as multiline, so its old reviewer package remains quarantined. Test methods do not print the protected text.

The first one-item smoke was blocked at pre-model cohort selection because the exclusion manifest accidentally included all 118 videos used by unrelated historical experiments; **zero model calls occurred**. I corrected the manifest to exclude only prior claim-generation media (4 unique hashes among 85 candidate videos) and reran a bounded GPU smoke on one different candidate, 16 frames, pinned Qwen2.5-VL-7B-Instruct 4-bit BF16. It completed one model call in 13.05 s after weight load, peak allocated VRAM 7.48 GiB on RTX 5070 Ti. The stricter prompt still failed the quality gate (`claim_multiline`), and no frame-index list parsed (0); the answer label parsed. This is a generation-format diagnostic only, with no human review or faithfulness score. Per smoke-first policy, do **not** scale to three videos or keep changing prompts on this sample. The format generation branch is currently no-go pending an outcome-blind diagnosis of why model output violates the fixed schema.

The calibrated-clock independent audit confirms the existing PTS residual check assumes the annotation seconds and decoder timestamps share a zero-relative origin; it does not observe player seeks/displayed frames or drift. The synthetic affine helper has only arithmetic tests and no anchor provenance, coverage, or uncertainty validation. The blank calibration worksheet already requires separate frozen fit/held-out anchors, source/pixel bindings, a justified tolerance, and cohort-wide pass. No real calibration has passed. These are independently reviewed design facts; no protected media was read for that audit.

The separate control/inference audit confirms the nearest-control utility can assign equal-size disjoint ranks when enough non-evidence ranks exist, but the current 3-item geometry audit is not production assignment from human-reviewed claims. No source-cluster target, sampling mechanism, justified superpopulation inference, or SESOI is established; the old head threshold must not be reused. The exact finite cohort could support only descriptive estimates after the remaining gates; a population CI requires a defensible sampling and source-cluster model.

### Faithfulness progress

- **Direct:** `未実施`, still **0/3 human-reviewed, 0/3 scored, 0 effects, no CI/test**. The one GPU run was a prompt/format smoke, not a faithfulness experiment.
- **Auxiliary:** latest generation 1/1 GPU call, but claim-format gate failed and frame indices parsed 0/1. Previous three candidates are 3/3 format-flagged and quarantined. No model output or score is accepted as efficacy evidence.
- **Gates:** independent human review, usable claim format, browser execution, player-to-decoder calibration, all-item production control, inferential target/SESOI, source independence, and authenticated scoring authorization remain unresolved. Rights remain unverified; all artifacts remain local.
- **Next action:** audit how Qwen output is decoded/parsed and whether generation text includes serialization artifacts, without interpreting semantics; verify format-check tests/full suite; continue independent production-timing and inference protocol work. Do not run a full generation batch or scoring until a smoke genuinely satisfies the prespecified format gate and downstream gates are defined.

## 15:42 JST checkpoint — claim-only cohort rebuilt and packaged

At **15:42:11 JST**, **17m24s** of the fresh six-hour window have elapsed; **5h42m36s** remain to the fixed **21:24:47 JST** deadline.

After two exploratory prompt-format smokes, I froze a simpler claim-only contract for a *new* exploratory cohort: one standalone sentence, with no answer-letter or model-selected-frame output. The first simple-sentence smoke produced one clean single-line candidate and passed the structural gate (**1/1**). Because the parser/contract had been changed after seeing that smoke, I excluded that video and selected a further cohort under the final prompt/parser.

The final exploratory GPU packet used Qwen2.5-VL-7B-Instruct at the pinned revision, 4-bit BF16, 16 frames, deterministic decoding and the local RTX 5070 Ti. Requested/effective **3/3**, format-valid **3/3**, three distinct media; 15.22 s generation after model load and **7.67 GiB** peak allocated VRAM. This is still only format feasibility. It provides no semantic support judgment, temporal localization, answer-margin score, treatment/control comparison, or faithfulness effect. Answer/frame output is intentionally absent under this claim-only contract.

I independently reran the deterministic CPU cohort selector on the same frozen exclusion manifest and verified exact equality of cohort hash and selected media set with the GPU packet. From 85 candidate videos, it scanned 10; 7 were prior claim-media exclusions, 3 passed the current timing/control geometry prefilter, 0 timing rejects, 0 decoder failures, and 0 candidates without an equal-count gold-interval geometry control. This is a pre-review feasibility filter on annotation geometry; it is not the production assignment from reviewed claim intervals and narrows the target cohort. Seventy-five candidates were not scanned after reaching the requested three.

The new packet was metadata-lineage checked 3/3 and built under frozen V5 review policy. A fresh private custody manifest binds all three exact source/staged media files; the isolated reviewer package contains only the HTML and 3 token-named files (0700 directory, 0600 files), has no stable source-value hits, and custody audit reports **0 failures**. Reviewer-page JavaScript passes Node syntax validation. It remains not browser-tested and not human-reviewed. The older malformed V5 package remains quarantined.

The complete repository suite now passes **900/900** in 8.843s; focused generator/builder/renderer tests pass **22/22**; changed Python files compile and `git diff --check` passes. An earlier attempted one-video run with the overbroad historic exclusion list correctly failed before model loading (0 calls); the later corrected runs used a narrower claim-generation-only exclusion list, preserving provenance and avoiding reuse of the prior claim candidates.

### Faithfulness progress

- **Direct evaluation: `未実施`** — still **0/3 reviewed, 0/3 scored, 0 effects/CI/tests** for any claim-specific mask experiment. No mask or choice-margin intervention ran in these GPU calls.
- **Auxiliary evidence:** generation-format smoke: 1/1 passed for the initial sentence-output trial; separate final three-item batch 3/3 passed format validation and independent source lineage. Cohort geometric prefilter reproduced exactly. These are engineering and sample-feasibility diagnostics only.
- **Remaining gates:** independent human review of the *new* package; actual browser playback; annotation-player-to-decoder clock calibration; reviewed-interval production mask and valid disjoint equal-count control for every item; defensible inference target/SESOI/source-cluster assumptions; authenticated scoring authorization. Rights status remains unverified; work remains local-only.
- **Next action:** keep the new package explicitly `draft_not_reviewed`; continue outcome-blind production-clock/control and statistical-target audits, strengthen quality/reproducibility manifests, and do not invoke the scorer. A syntax pass is not a review or scoring gate.

## 15:57 JST checkpoint — independent format repeat, full geometry census, CLI integration test

At **15:57:36 JST**, **32m49s** of the fresh six-hour window have elapsed; **5h27m11s** remain to **21:24:47 JST**. This checkpoint distinguishes the complete scan from the earlier quota-stopped three-item feasibility scan.

### Fixed-prompt format repeat on real video

Ran one bounded local GPU batch on three previously unused local NExT-GQA videos, 16 frames each, with the frozen claim-only prompt and pinned Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit quantization/BF16 compute, deterministic generation, and RTX 5070 Ti. Requested/effective was **3/3**; the format-only parser accepted **2/3** and flagged **1/3** as `claim_sentence_shape_invalid`; the runner status is `partial`. Post-load generation took **16.236 s** and peak allocated VRAM was **7.331 GiB**. The new private packet is `data/private_nextgqa/claim_generation_sentence_repro_20261016_01.json`, SHA-256 `4eafb037ae9fedf4d98ff1f3c3dc5faaaccebb9702ba55e508e4c6105a743d98`, mode 0600. Cohort hash: `b62a1989bbd71bbbf819c2a8a1829fe71f9c857d7e73d7d3feddf91dbcc3dc33`. The exclusion list contains ten previous claim-media hashes; a CPU check confirms all three new media hashes are distinct from those earlier records.

Compared only as an engineering observation, the prior separate frozen batch passed the same syntax/shape gate 3/3, while this new batch passed 2/3. The first batch had informed the decision to perform this repeat, so this is exploratory and is not a calibrated generalization estimate. The gate tests single-line/basic sentence shape only; it does not prove grammatical single-sentence status, truth, question relevance, or temporal support. No retries, prompt tuning, score computation, or replacement of the failing record occurred. This result is a format-reliability limitation, not faithfulness evidence.

### Full outcome-blind candidate geometry census

Completed and independently rechecked a full scan of the **85/85** deterministic candidate videos, rather than stopping after a requested sample. The current exclusion manifest covers ten previously used claim-generation media; all ten occur in the 85-candidate pool, leaving **75/85** unexcluded. Those 75 all passed the fixed 16-frame/50 ms uniform-grid timing check and had at least one question with a nonempty gold-interval geometry and an equal-count disjoint temporal control (**75/75**); timing rejects, decoder failures, and no-control videos were all **0**. The selector tested **91 questions** to find those assignments and scanned all candidates. All 85 candidate media hashes are unique. This is an annotation-geometry prefilter based on the benchmark's gold intervals, not review of model claims and not a claim-mask cohort authorization.

The complete census is recorded only in the private mode-0600 aggregate `data/private_nextgqa/geometry_census_full_20261016_01.json` (SHA-256 `8af86aedd9a53f3bb8a7285249c0a9963d74e6bcabe0cdcd438bfed3d05c26b0`). Cohort fingerprint `d23efa91a04c86bff06195034812278d56b01930e3165195392ae9d8e0733407`. The aggregate binds **17/17** current dataset/sample-manifest input hashes, selector SHA-256 `6f2ef8e6534294790c958b2d5dfc4c40a766c1a202b655390d1336a2e2dad865`, and exclusion-manifest SHA-256 `18d3651a2de35f10a4e4df7fe1e84e74cfc67d8b9538b03c8530a6adf654efd4`; my independent reread recomputed all 17 input hashes successfully. Elapsed scan time was **334.613 s**, peak RSS **1,320,756 KiB**. The earlier **78/85** count used the seven-hash exclusion manifest before these three newly generated videos were added; this complete ten-hash scan yields 75. The three-record claim-linked review set and the 75-record gold-geometry pool are different objects.

### Package regression test

Added synthetic CLI-path integration tests for `scripts/package_nextgqa_claim_review_handoff.py`. The success fixture exercises local-only flag, input modes, custody audit, lineage hash, exact renderer re-generation, and isolated packaging of HTML plus three synthetic videos. A tampered HTML page is rejected before output creation. Focused tests pass **4/4**; the entire repository Python suite now passes **902/902** in **9.457 s**; changed Python files compile and `git diff --check` passes. These synthetic tests do not validate browser playback or a human's actual viewing.

### Literature overlap check

A primary-source check surfaced the June 2026 preprint *Chains That See, Answers That Don't* (arXiv:2606.22862, later listed for the July 2026 EvalMG workshop). It studies Qwen2.5-VL on Video-MME using forced CoT, video swaps, and a visual-degradation ladder; it reports that video-conditioned chains need not improve answer accuracy, and explicitly limits its conclusions to that model/benchmark setup. This is adjacent to the present project's question of whether a visual explanation is useful, but it does not test human-verified claim-specific temporal intervals against equal-count temporal controls on NExT-GQA or the fixed answer log-margin. Therefore it narrows broad novelty language around “video faithfulness under intervention” but does not answer this project's narrower proposed measurement. Its scoring choices and thresholds are not transferred here. Source: https://arxiv.org/abs/2606.22862 (submitted 2026-06-22; scope/protocol at https://arxiv.org/html/2606.22862v1).

### Faithfulness progress and next action

- **Direct faithfulness evidence remains `未実施`: 0/3 human-reviewed, 0/3 scored, 0 claim-specific effects, no CI/test.** No mask/control inference was run in either GPU batch or the CPU geometry census.
- **Established only:** one new real-video GPU format repeat with 2/3 syntax/shape pass; exhaustive outcome-blind gold-interval geometry census 75/85 eligible under the ten-hash exclusion rule; a synthetic package CLI regression path; and a scoped literature comparison.
- **Independent review is in progress** for the GPU/census provenance and package CLI test boundary. Pending that audit, retain the 75 count as a geometry-screening fact only. The exact reviewed-claim population, reviewer mapping, browser execution, annotation-player-to-decoder calibration, source-event dependence, inference target, SESOI interpretation, and scoring authorization remain unresolved. Rights status remains unverified; all protected artifacts stay local.
- **Next:** close the independent audit; preserve the failed format record and stop prompt/model-output tuning on these cohorts; continue outcome-blind design work toward a finite descriptive target only if its population, attrition, and uncertainty policy are frozen before scores. Do not score until human review and temporal mapping are resolved.

## 16:07 JST checkpoint — benchmark evidence annotations and novelty boundary

At **16:07:30 JST**, **42m43s** have elapsed and **5h17m17s** remain.

### Verified literature and local measurement implication

The original NExT-GQA paper explicitly extends NExT-QA with temporal intervals tied to QA pairs; it reports 10,531 valid intervals across 8,911 QA pairs, manually annotated by at least two annotators and checked/refined by another pass. Its main metrics include QA accuracy, interval IoP/IoU, and grounded-QA accuracy. This confirms that the local benchmark has human temporal support spans that can directly support an **answer-evidence dependence** experiment. The benchmark's span supports the QA answer; it is not automatically a localization of a separately generated explanation claim. Primary source: Xiao et al., *Can I Trust Your Answer? Visually Grounded Video Question Answering*, arXiv:2309.01327v2 (2024): https://arxiv.org/abs/2309.01327.

A newer adjacent result, *INFACT* (arXiv:2603.11481, 12 March 2026), evaluates Qwen2.5-VL-7B among other Video-LLMs with visual degradation, evidence corruption, and shuffle/reversal temporal interventions, using answer stability/sensitivity metrics. Another 2026 preprint, *Chains That See, Answers That Don't* (arXiv:2606.22862, 22 June 2026), evaluates forced reasoning chains on Qwen2.5-VL/Video-MME using video swaps and a visual-degradation ladder. These papers mean broad novelty claims about “Video-LLM faithfulness under visual or temporal intervention” would be too wide. Their reported protocols do not test whether a human-verified interval for a model-generated natural-language claim is more influential than an equal-count temporal control on the fixed gold-vs-best-foil next-token log-margin. That narrower design remains a possible contribution, but it is not yet measured here. Sources: https://arxiv.org/abs/2603.11481 and https://arxiv.org/abs/2606.22862.

This evidence justifies one distinct, bounded GPU branch already present in the repository: use NExT-GQA's QA support interval to compare answer margins under gold-interval masking and equal-count temporal-control masking. This would measure benchmark-answer evidence dependence only, not claim-specific explanation faithfulness. It is eligible to proceed as an exploratory finite-cohort measurement after its one-video smoke and the repository's cohort/condition preflight pass. The runner explicitly uses five conditions (full video, black-fill gold interval, black-fill equal-count control, blur-fill gold interval, blur-fill control) and reports no p-value or confidence interval. Masking can create out-of-distribution inputs, and the annotation-player-to-decoder PTS mapping remains uncalibrated; any result must retain those limits.

### Consequence for this window

The user was right that NExT-GQA provides evidence intervals. The current answer-margin pilot can use them without pretending they validate free-form generated claims. I made the runner's deterministic cohort size configurable at **1/3/10** for a proper one-video GPU smoke and added CPU unit tests for the fixed gold-vs-best-foil margin, contrast sign, and non-finite rejection. The fixed Qwen/model/precision/intervention definitions remain unchanged. Focused checks pass; the complete suite at this checkpoint passes **907/907**. No answer-margin GPU score has been run yet in this branch.

## 16:19 JST checkpoint — answer-margin GPU smoke and frozen 10-video exploratory batch

At **16:19:16 JST**, **54m29s** of the six-hour window have elapsed; **5h05m31s** remain to **21:24:47 JST**. Faithfulness progress below continues to distinguish generated-claim evaluation from this auxiliary benchmark-answer diagnostic.

### Cardinality blocker repair and independent review

The prior independent audit found that `--n` was not enforced as an effective count. The runner now fails closed unless the deterministic eligible cohort is exactly the requested size, checks requested and effective counts in the preflight binding, and writes separate requested/effective fields in score output. CPU tests cover accepted and short counts for n=1/3/10 and mismatched preflight declarations. The exact pre-change selector source was preserved locally as `data/private_nextgqa/run_nextgqa_gold_interval_margin_pilot_census_source_20261016_01.py` (0600, SHA-256 `bbaefd263289f592695c5051f48b63e7c1896fb731d1ec1f87a25faf3a6bd326`), matching the selector fingerprint used by the complete geometry census. The independent code audit confirmed the gate closes before condition preparation and model load. I also bound the timing/control utility source hashes into the preflight→score contract.

Focused tests pass **6/6**, complete Python suite **910/910** in **9.027 s**, Python compilation and `git diff --check` pass. No preflight cardinality test uses GPU or protected text.

### Real-video GPU smoke (n=1)

The one-video 16-frame preflight passed. On the RTX 5070 Ti with Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit quantization and BF16 compute, the score completed all **5/5** fixed condition forwards in **19.968 s** after the runner timer began; peak allocated VRAM was **7.279 GiB**. Status `passed`, no failures. The primary black-fill control-minus-gold-interval effect was **+7.000 nat**; blur sensitivity effect **+5.875 nat**. These are one selected video only, with no SD/CI/test. The item was excluded from the subsequent batch based only on its media hash, to avoid reusing the smoke item. Private score SHA-256 `bcc45422d18470a488b855f29932fe638e318904b7ba1884e3ff49ceb284d879`; preflight SHA-256 `7d53bf4fa85a1d412e3c66d95ccf6f239ab2fb7e4d8338f5370c7422e94f91c7`; both mode 0600.

### Exploratory 10-video GPU batch

The frozen preflight selected **10** different timing/control-eligible videos after excluding the 10 prior claim-generation videos and the smoke video. It prepared 16 frames/video and all five conditions. The score completed **10/10** full condition sets (**50/50** requested model forwards), zero failures, in **135.846 s**; RTX 5070 Ti peak allocated VRAM **9.059 GiB**. Preflight→score bound the runner, decoder, timing/control helper source hashes, model revision/config, dataset inputs, exclusion manifest, and cohort. Full restricted-choice accuracy was **7/10**.

Primary endpoint is fixed gold-vs-best-foil next-token log-margin under black-fill equal-count temporal control minus the same margin under black-fill gold QA interval masking; positive means the gold-interval mask reduced margin more. Observed selected-cohort mean **−0.1375 nat**, sample SD **1.5969 nat**; positive for **4/10**, negative for **6/10**, zero **0/10**. Gold-mask mean margin was **2.2250 nat** and equal-count-control mean was **2.0875 nat**. Blur control-minus-gold mean was **−0.3625 nat**. No CI, p-value, population inference, threshold, or multiplicity-adjusted claim is reported. The one-item smoke result is a separate excluded cohort and is not pooled into n=10.

Private score SHA-256 `4502f05248a61786fe13ed162e3a053c584480bcf79db652892e097a9e68d7f2`, preflight SHA-256 `beb698b731dd033d8517d0ac439b842914f072ed38f59e34461c0cc05181c08a`, mode 0600. Cohort SHA-256 `da0b12f0fc6143404175d352a67075743921d4f1af53ba4aff551dfe46048ac9`; runner SHA-256 `313742549b3979dacfb458f75bbfe5b846dc65599d98da5c2cfe3f10e7d8535e`. Independent result audit is pending.

This remains an **exploratory, deterministic, non-probability selected benchmark subset**. The gold spans support benchmark QA answers, not generated explanation claims. The annotation-player-to-decoder PTS origin is not independently calibrated; black and blur masks can be out of distribution; equal frame count does not remove all visual distribution shift. The result is mixed and small. It motivates no claim of faithfulness and no tuning/reselection on these scores.

### Faithfulness progress

- **Direct generated-claim faithfulness:** still **0/3 human-reviewed, 0/3 scored, 0 claim-specific effects, no CI/test**. No direct claim-mask experiment has run.
- **Auxiliary answer evidence-dependence:** one-video engineering/GPU smoke (5 forwards) and separate n=10 descriptive experiment (50 forwards), effect mean −0.1375 nat, sample SD 1.5969, 4/10 positive. This does not satisfy the claim-specific review gate.
- **Remaining direct gates:** human review of the isolated three-video package, browser playback, calibrated annotation/player↔decoder clock, claim-specific masks and equal-count controls, source-cluster/inference target and SESOI, and scoring authorization. Rights remain unverified; artifacts remain local-only.
- **Next action:** independently verify the n=10 aggregate/hash/condition computations. Then pursue a distinct outcome-blind calibration or benchmark-control audit; do not repeat or tune the same n=10 outcome. Continue the six-hour window while justified work remains.

## 16:41 JST checkpoint — paired 16/32-frame sensitivity and memory repair

At **16:41:49 JST**, **1h17m02s** of the six-hour work window have elapsed; **4h42m58s** remain to **21:24:47 JST**.

### Frame-grid audit and assignment control

On the exact n=10 video/question assignments used in the 16-frame score, the local frame2time audit finds **11/12** support spans intersected by the 16-frame uniform grid; the 32- and 64-frame grids intersect **12/12**. One anonymous item (index 8) has only **1/2** support spans sampled at 16 frames. All 10 grids meet the fixed 50 ms uniformity bound. Across 320 `frame2time` points, nearest decoded-relative-PTS residual median is **8.67 ms**, p95 **16.2 ms**, max **20.83 ms**. This assumes a shared zero origin and is not a browser/player-clock calibration. The aggregate is local-only, mode 0600; the expanded per-video output contains only anonymous index and span coverage counts. Its runner was an uncommitted worktree version: SHA-256 `409dbd5fdc182ae0d8554712c9ffa32f4944e59463ab5d8261c0248f534b9f72` (the artifact's recorded source commit does not alone identify that dirty source; exact bytes are fixed by this hash).

An unlocked 32-frame preflight selected a different video/question assignment, so it was not scored as a paired comparison. A private mode-0600 hash-only cohort lock was instead created from the exact 16-frame score selection. It binds the original preflight SHA `beb698b731dd033d8517d0ac439b842914f072ed38f59e34461c0cc05181c08a`, score SHA `4502f05248a61786fe13ed162e3a053c584480bcf79db652892e097a9e68d7f2`, and cohort SHA `da0b12f0fc6143404175d352a67075743921d4f1af53ba4aff551dfe46048ac9`; lock SHA-256 `a40e75489650720e49f98812771f36d3518d25f20e094566a94a60f870e02fcd`. The lock contains media and question-ID hashes, not source IDs or text.

### 32-frame GPU sensitivity run

The first locked 32-frame attempt stopped after **1/50** model forwards and **0/10** complete condition sets with a generic `RuntimeError`; it emitted no effects. This incomplete artifact is retained separately and excluded. Independent code review then found that score setup kept every video's processor tensor batches live together. The runner was changed to prepare grid checks without retaining batches and to build/free one video's five condition batches inside the scoring loop. It now records the failed condition and pre-forward host/VRAM snapshots. A synthetic test checks the grid-only and retained-batch return contracts; it does not directly measure process memory. Focused tests pass **8/8**, the complete Python suite **912/912** in **9.210 s**, compilation and `git diff --check` pass. The source reviewer confirmed the code path and remaining test limitation.

With a fresh current-source preflight and the same exact hash-only assignment lock, the repeated 32-frame score completed **10/10** condition sets, **50/50** forwards, zero failures, in **235.513 s**, peak allocated VRAM **11.428 GiB**. The current runner SHA-256 is `d6b64ef622b93dfe83694ea0dd0e093e776e0a4186f8362cd56dab8aa78db1f2`; preflight SHA is `a520a30eaf14e6a8e2254daa629add649f714ad2a33043ca3206b6b64c5f4bcd`; score SHA is `d4f6dc2f6411efe5809e67ae9ea281ccf7acb38b3dc883e6331e57f495587749`. These files remain mode 0600. Independent output audit is pending.

### Fixed-assignment descriptive comparison

For the same ordered video/question assignments, the primary black-fill control-minus-gold-interval margin was:

- **16 frames:** mean **−0.1375 nat**, sample SD **1.5969**, range **[−3.000, +3.000]**, positive **4/10**, negative **6/10**.
- **32 frames:** mean **−0.3375 nat**, sample SD **1.9085**, range **[−4.000, +1.625]**, positive **6/10**, negative **4/10**.
- **Paired change (32−16):** mean **−0.2000 nat**, sample SD **1.5323**, positive **6/10**, negative **4/10**.

Full restricted-choice accuracy was **7/10** at 16 frames and **9/10** at 32 frames. Blur control-minus-gold means were **−0.3625** and **−0.3500 nat**, respectively. No CI, p-value, or population inference is made. This post-score temporal-resolution sensitivity is exploratory; changing frame count changes the visual input and can affect both evidence coverage and model processing, so it does not isolate interval coverage as a causal mechanism. No threshold or metric was changed in response to these results.

The study still measures **NExT-GQA benchmark gold-interval answer-margin dependence**, not reviewed generated-claim faithfulness. The score artifacts correctly keep `claim_review_completed=false` and `claim_specific_faithfulness_estimate=null`. The nearest-PTS residual calculation does not calibrate the review player's clock; masking can be out of distribution; controls match count/rank, not visual semantics. Rights remain unverified; all protected data and derived results are local-only.

### Faithfulness progress

- Direct claim-specific evaluation remains **0/3 reviewed, 0/3 scored, 0 effects/CI/tests**.
- Auxiliary answer-evidence branch now has one separate n=10 16-frame run and one n=10 32-frame same-assignment sensitivity (50 forwards each); both selected-cohort primary means are negative and mixed across items. They provide feasibility and a sampling-resolution warning, not direct faithfulness evidence.
- The independent design audit advises that the smallest defensible later claim-specific study is a finite descriptive 3-video/3-claim feasibility run only after all reviews, actual browser playback, player↔PTS calibration, and all-item claim mask/control assignment pass. Any failure in any one item should produce zero score calls for the whole frozen cohort; no row replacement. A t CI or the previous head threshold is not justified for that deterministic n=3 set.
- Next: independently verify the paired 32-frame artifact and its lock linkage; then continue preparing the CPU-only claim-ledger→PTS assignment and fail-closed whole-cohort scoring foundations. Do not run direct claim scores before the independent review and clock gates.

## 16:55 JST checkpoint — reviewer-to-PTS planning seam

At **16:55:41 JST**, **1h30m54s** of the six-hour window have elapsed; **4h29m06s** remain to **21:24:47 JST**.

### Independent audits and corrected counts

Three independent roles reviewed distinct parts of the work. The code/protocol audit confirmed that the anonymous V5 review JSON, PTS assignment helper, and all-cohort synthetic timing gate existed separately, with no production connection from reviewed claim intervals to PTS masks. The protocol audit found that V5 valid structural review records may still contain a non-support decision, so any later scorer must explicitly require `supports_claim` for every frozen row. The player audit found `ffplay`/FFmpeg/PyAV available but no browser; the V5 page uses HTML5 `currentTime`, and neither `ffplay` nor the prior zero-origin frame2time residual calibrates that clock. A later calibration must use the same HTML5 path, frozen fit/held-out anchors, the exact media bytes, and an uncertainty band that cannot change sampled-frame classification at any interval boundary.

Recomputed directly from the two private score JSON files, the answer-margin sign counts are **16 frames: 4 positive / 6 negative**, **32 frames: 6 positive / 4 negative**, and paired `32−16`: **6 positive / 4 negative**. The 16:41 section above had the 32-frame signs reversed; it is corrected here in the log. These remain auxiliary NExT-GQA QA-gold-interval answer-margin dependence results, not claim faithfulness.

### CPU planning adapter

Added `scripts/claim_review_assignment_plan.py` and synthetic tests. It joins opaque review tokens through the worksheet, checks the exact review JSON bytes and policy binding, checks a supplied clock-calibration JSON's bytes/content/hash, maps player intervals through its affine transform to exact PTS seconds, rejects sampled frames whose classification could change under the declared PTS uncertainty bound, and creates equal-count disjoint rank-matched controls for the full frozen three-row cohort. It emits no stable IDs, claim text, interval values, or rationales. Any row failure preserves denominator **3**, returns no assignments, and reports **0 score calls**.

The output is deliberately named `assignment_plan_structurally_passed_unvalidated`. The adapter is **not** an execution gate: review/custody audit reports, calibration pass/uncertainty claims, and timing/pixel digests are caller-supplied and are not independently authenticated by this adapter. It never loads a model or media and always returns `execution_authorization=not_granted`. The returned geometry is therefore a planning seam only; the next scorer requires independent exact-byte audit integration and a real browser calibration artifact.

Dedicated tests passed **12/12** for the adapter and **16/16** for PTS utilities, including affine offset/drift, exact artifact binding, bool-vs-int JSON tampering, boundary uncertainty, unsupported late-row atomic failure, and pixel hash mismatch. The full suite passed **925/925** in **8.847 s**; compileall and `git diff --check` passed. An independent auditor also confirmed that the adapter only authenticates each calibration file's internal bytes/hash consistency: it does not independently establish the trustworthiness of the held-out pass, review/custody report, or supplied timing/pixel digests. The output state is therefore `assignment_plan_structurally_passed_unvalidated`, never an execution-ready status. New adapter SHA-256: `f3891c807ffa9c3f43989214ebb0e141a953b6f62c647725254613cf9ae3760b`; adapter test SHA-256: `29fc86500a35c1ea4538213d8f50a7a8477ac963d31d32b0739316353a7e5f54`; PTS utility SHA-256: `515b26ecca9a7e917b1c050ba0cede8dce3d9c837955dec28def31bbc48669a5`; PTS test SHA-256: `9944707d4bcc510ff0f58d5bdbef9891609fa1cc8118a0bd476a8bfd31f9759c`.

### Faithfulness progress

- Direct generated-claim faithfulness remains **0/3 reviewed, 0/3 scored, 0 effects, 0 confidence intervals, and 0 tests of a claim-specific effect**. The planning adapter adds no observations to this count.
- Auxiliary benchmark QA answer-margin dependence remains **n=10 at 16 frames** and **the same n=10 assignments at 32 frames**, each **50/50 forwards, 0 failures**. Means are **−0.1375 nat** (SD 1.5969; 4/10 positive) and **−0.3375 nat** (SD 1.9085; 6/10 positive), respectively. No CI or test; these do not measure generated claims.
- Historical closed head confirmation remains **79/79**, estimate **−0.0756 nat**, 95% t CI **[−0.1578, +0.0065]**, below the prespecified +0.25 nat practical threshold. Head/layer/alpha search remains closed.

Next action: prepare and independently audit a local-only HTML5 clock-capture/calibration path that can be used when an offline browser is available, while keeping reviewer semantics and QA outcomes hidden. Do not treat `ffplay` as a V5 substitute. Keep the planning adapter disconnected from model scoring until the review, custody, clock calibration, and exact pixel/provenance audits are genuinely verified. Keep protected inputs and derived results local; rights remain unverified.

## 17:09 JST checkpoint — planner audit repair and QA-target contract gap

At **17:09:27 JST**, **1h44m40s** of the six-hour window have elapsed; **4h15m20s** remain to **21:24:47 JST**.

### Independent adversarial audit and repairs

After the initial planner audit, an independent auditor found three additional
structural pass cases: the worksheet's top-level occurrence policy could be
rebound away from V5; empty claim IDs or model revisions could pass; and the
lone-surrogate test used six literal characters rather than an actual invalid
Unicode scalar. The planner now requires the frozen worksheet's V5 policy ID,
hash, occurrence policy, and frozen-pre-review state; validates nonempty
trimmed UTF-8 claim IDs and the builder's pinned-revision format; and fails
closed on the actual `chr(0xD800)` surrogate. The earlier displayed-claim text
tampering and claim-manifest policy cases remain covered too.

The independent auditor reran adversarial checks and found those cases
rejected, with no additional pass in the reviewed boundary. Focused suites:
planner **18/18**, handoff **11/11**, renderer **7/7**, PTS utility **16/16**.
The full repository suite had passed **929/929** immediately before this latest
planner tightening; it is being rerun against the new snapshot. `git diff
--check` passes. Planner SHA-256 is
`cde80d4f284d3254b3c156b3cd3fb69a6c70b23fc748054d45fd1fa24b4cff41`; tests SHA-256
`b4dfea2b5c725673e63d3e689eda41cbd61afbb681288c0b9f079f5e30578bb5`.

This remains an unauthenticated planning component: the caller-supplied review,
custody and calibration reports can still be fabricated, and no browser clock
or media pixel provenance is independently established here. It still returns
zero score calls and `execution_authorization=not_granted`.

### Faithfulness progress

- Direct generated-claim faithfulness: **0/3 human-reviewed, 0/3 scored, 0
  claim-specific effect estimates, 0 claim-specific CIs or tests**.
- Auxiliary NExT-GQA QA-gold-interval answer-margin: n=10 at 16 frames and the
  same n=10 assignments at 32 frames, each 50/50 forwards and 0 failures;
  means **−0.1375 nat** and **−0.3375 nat**, respectively, with no inferential
  interval or test. This is not claim-faithfulness evidence.
- Historical closed head confirmation: **79/79**, **−0.0756 nat**, 95% t CI
  **[−0.1578, +0.0065]**, below the +0.25 nat practical threshold; no head,
  layer, or alpha search is reopened.

### Next foundation

Independent protocol review found that the future three-item descriptive
estimand is clearly signed and bounded to the exact cohort, but the current
claim input manifest does not bind the benchmark question, answer-choice
order, gold index, one-token A–E IDs, or scoring prompt/tokenizer/model config.
Those need a separate frozen QA-target manifest before any scorer can safely
consume the planner's row assignments. The next task is to define and test
that private hash-only target contract, while preserving the distinction
between structural checks and authenticated review/calibration. No claim score
or GPU run is justified until the human review and real HTML5 clock gates pass.

Rights remain unverified; any protected data and derived artifacts remain
local. The worktree remains dirty with unrelated/pre-existing user files, so
no staging, commit, or publication is performed.

Post-checkpoint verification completed at **17:11 JST**: the full repository
Python suite passed **931/931 in 9.108 s** against the repaired planner; the
intentional negative-path diagnostics above are expected test output.

## 17:12 JST checkpoint — frozen QA-target contract

At **17:12:38 JST**, **1h47m51s** of the six-hour window have elapsed; **4h12m09s** remain to **21:24:47 JST**.

Following the independent estimand audit, added
`scripts/claim_answer_target_contract.py` and synthetic tests. The hash-only
contract checks exact three-row video-to-claim membership, per-row QID and
question digests, ordered five-choice digests, fixed gold index, A–E label
order and distinct token IDs, one-token-count declarations, each rendered
scoring-prefix digest, and hashes for the claim-input manifest, protocol,
checkpoint, model config, tokenizer, processor config, and prompt builder.
It returns aggregate status only and always keeps `score_calls=0` and
`execution_authorization=not_granted`. Six synthetic cases pass, including
changed config/claim bindings, choice/gold/label order, malformed duplicate
JSON keys, and non-one-token declarations.

The protocol draft now explicitly requires this QA target binding and states
that the one-token fields remain untrusted declarations until a real builder
reruns tokenization with the exact tokenizer and prefix. The helper does not
construct targets from NExT-GQA, validate those declarations against a model,
authenticate supplied hashes, or authorize inference. Independent auditors
are reviewing the new contract; no score or protected input was used to create
it. Python compilation, focused tests (**6/6**), and `git diff --check` passed.
Code SHA-256: `a3bc6d1ae962baceced424427e8045480cdc491d49e34add29de11e44cfd78f4`;
test SHA-256: `4e4019b859c3e6ffbeee570b486e691fb129a923dc066a4165932ef7147c940c`;
protocol draft SHA-256: `84c6af6eafacd41d142b82b4176c937a6830cb16b68f6d8b7d8b2f5c312c8fea`.
The full suite passed **937/937 in 9.209 s** after these additions.

Faithfulness progress is unchanged: direct generated-claim evaluation is
**0/3 reviewed, 0/3 scored, 0 effects/CI/tests**. The n=10 QA-gold-interval
results (16f −0.1375 nat; same assignments at 32f −0.3375 nat; each 50/50
forwards, no inference) remain auxiliary and do not evaluate generated claims.
The contract makes a future estimand's target auditable but does not advance
the evidence count. Next, resolve independent audit findings and assess whether
the private review handoff can export the exact target hashes without exposing
gold text to reviewers; keep the scorer disconnected until human review and
real HTML5 clock calibration pass.

Follow-up adversarial review found two parser issues and a provenance omission:
Python would treat boolean/float values as equal to integer one in the declared
token-piece-count list; repeated choice hashes were accepted; and no explicit
dataset-source manifest or row digest was required. The validator now checks
each count with exact `int` type, rejects repeated ordered-choice digests, and
requires `dataset_source_manifest_sha256` plus a per-row
`dataset_source_row_sha256`. Seven focused tests now pass, including both bool
and float spoof values; two independent auditors verified rejection and found
no additional material bypass in this structural scope. The full suite passed
**938/938 in 9.322 s**; compileall and `git diff --check` passed. Final hashes:
validator `c180bdb175109d104317e30da7c77e24b2addf83463454767880ffc1a5e548d4`,
tests `bae087e808b0dc87b5a368f50a422cebddc96e4f7cbbbd0b58f6f78b0b835bcb`,
protocol draft `c93bfde59110ae492c0595134581c72605787606f66e132549d97ebf1838a477`.
These hashes remain syntax-checked caller declarations; no source-row data,
tokenizer, or prompt bytes are read or authenticated by this helper.

## 17:28 JST checkpoint — integrated preflight authenticity boundary

The active six-hour work window is accounted from **15:24:47 JST** to
**21:24:47 JST**. At **17:28:55 JST**, **2h04m08s elapsed** and **3h55m52s
remain**. No work is counted beyond timestamps actually observed.

Independent adversarial review demonstrated that a synthetic caller can
replace QA choices and the answer, then update the target choice/gold/source
row digests, and still pass the combined structural preflight. A second attack
rebound `dataset_video_id` in both QA and lineage rows and updated the declared
hashes; that also passes structural checks. These are expected limits of the
synthetic-only adapter, but explicit regression tests now prevent later
readers from mistaking internal consistency for source authenticity. Both
assert `combined_preflight_structurally_passed_unvalidated`,
`score_calls=0`, `execution_authorization=not_granted`, and
`scope=synthetic_preflight_rehearsal_only`.

The protocol's clock-gate wording now calls the held-out maximum residual an
empirical guard over captured anchors, explicitly not an uncertainty bound,
confidence interval, or future-error guarantee. The planner boundary test was
renamed to match this interpretation.

Verification before the two independent attack cases were combined: focused
planner/target/preflight suite **36/36**, whole repository **949/949 in
9.210 s**, `compileall` and `git diff --check` passed. The independent agent's
new identity-rebinding test separately passed with the preflight suite at
**12 tests**. Re-run both suites against the combined snapshot before treating
these counts as final. Combined test SHA-256 is currently
`5182b10e00628df787f2c8e208d41e8a1a32322a6ac978779eded9843d6e70bc`; the
other checkpoint hashes are recorded after final verification.

### Faithfulness progress

- **Direct faithfulness: `未実施` — 0/3 claims human-reviewed, 0/3 scored,
  0 direct effects/intervals/tests** for Qwen2.5-VL-7B-Instruct on NExT-GQA,
  using the human-validated claim interval mask versus equal-count temporal
  control and fixed gold-versus-best-foil log-margin. Multiplicity handling
  has not been applied because there are no direct outcomes.
- **Auxiliary diagnostic only:** QA-gold-interval margin n=10 at 16 frames
  averaged −0.1375 nat (SD 1.5969; 4 positive / 6 negative); the same n=10
  at 32 frames averaged −0.3375 nat (SD 1.9085; 6 positive / 4 negative).
  Each had 50/50 forwards and zero failures, but no confidence interval or
  test; these runs do not evaluate generated-claim faithfulness.
- **Historical closed confirmation:** 79/79, −0.0756 nat, 95% t CI
  [−0.1578, +0.0065], below its +0.25 nat practical threshold; no head,
  layer, or alpha search is reopened.
- **Next evidence gate:** build a source-byte-backed target adapter and actual
  browser-player clock capture groundwork; then still require three
  independent human claim reviews and a pre-score inferential design gate. No
  GPU scoring run is justified by this synthetic integration.

Worktree remains dirty with unrelated and protected user files. Nothing was
staged, committed, or pushed; protected data and derived artifacts remain
local, and rights status remains unverified.

## 17:33 JST checkpoint — source-byte adapter and clock capture validator

At **17:33:14 JST**, the six-hour window has elapsed **2h08m27s** and has
**3h51m33s remaining** to **21:24:47 JST**.

Added a CPU-only actual-source adapter at
`scripts/nextgqa_actual_qa_source_adapter.py`. It reads raw local `test.csv`
bytes, checks the exact CSV digest against the fresh cohort's lineage, requires
the selected video/QID/claim tuples to match the three frozen lineage records,
recomputes the exact question digest, resolves each source row uniquely, keeps
choice order, computes the gold index with the score runner's NFKC/whitespace/
casefold rule, and binds a digest of the full parsed CSV row. Adversarial
synthetic tests cover tampering, duplicate pairs/headers, ambiguous gold,
lineage mismatch, short rows, wrong types, and Unicode normalization.

Independently exercised it in memory against the actual local NExT-GQA
`test.csv` and the protected fresh-cohort lineage/worksheet: **3/3** frozen
video/QID pairs resolved; **3/3** question hashes agreed with both lineage and
worksheet; **3/3** rows had five ordered choices and a unique gold; source-row
hashes were produced for **3/3**. No file was written and no model/scorer was
called. A separate full-file count using the runner's normalization found
**5,553** CSV rows: **5,552** uniquely matched answers, **1** ambiguous match,
and **0** missing matches. Any future target using the ambiguous row must
block or select another row before scoring.

The independent clock audit also added
`scripts/html5_clock_capture_calibration.py`. It validates a supplied
HTML5-clock capture structure, binds it to exact media bytes, fits only
predeclared fit anchors using exact fractions, reports the held-out maximum
residual in decoder PTS seconds, and marks the result unvalidated with no
scoring authorization. Independent audit confirmed the unit conversion and
fit/held-out separation; no browser or actual clock capture was available.

Focused actual-source and clock suites: **12/12**. Whole repository suite:
**962/962 in 9.289 s**. Python compileall and `git diff --check` passed.
Hashes: source adapter `7920d82eac45fefcb6d6dd2024f43a454644295fcbbb9fb926504570b8729cb7`,
source tests `f740f6cbb7bbed0c47211c24eddc90127c04cca44a33bc90366429a73bf5adcb`,
clock helper `f7d0e76ce08828b672bf6837ba48ed2c3d4529987c2a96b24aa315c42c843f87`,
clock tests `646a2684b8ddf35d1c976524c6fe3122e4cdc591d708beeb16b6da817eeaf57b`.

### Faithfulness progress

- **Direct claim faithfulness remains `未実施`: 0/3 reviewed, 0/3 scored,
  0 effects/CIs/tests.** The source adapter checks benchmark-row plumbing; it
  says nothing about human claim support or score changes.
- **Auxiliary diagnostics remain unchanged:** n=10 QA-gold-interval answer
  margin −0.1375 nat at 16 frames and −0.3375 nat at 32 frames; both 50/50
  forwards, no failures, and no inferential interval/test. These are not
  generated-claim faithfulness results.
- **Still unresolved:** direct human review 0/3; actual HTML5 playback clock
  capture not done; inferential target/SESOI not fixed; rights status remains
  unverified. Browser capture adapter declarations do not establish that the
  supplied telemetry came from a real browser.
- **Next:** integrate actual CSV-derived QA bindings with a complete target
  contract while keeping raw QA private, then audit exact tokenizer/prompt
  declarations and the remaining pre-score design gates. Do not launch GPU
  scoring until review and clock gates pass.

No files were staged or published. The worktree contains user data and other
unrelated changes; all protected QA-derived values remain local.

## 17:40 JST checkpoint — reproducible source audit and finite target draft

At **17:40:40 JST**, **2h15m53s** of the requested six-hour window have elapsed;
**3h44m07s** remain to **21:24:47 JST**.

Added `scripts/audit_nextgqa_actual_qa_source_binding.py`, an aggregate-only
CPU CLI that can reproduce the actual source-row audit without writing an
artifact. Against the local CSV/lineage/worksheet it reports **5,553** rows,
**5,552** uniquely matched gold answers, **1** ambiguous gold match, **0**
missing matches, and frozen-cohort joins/questions/source-row bindings all
**3/3**. The single ambiguous source row is rejected by target construction;
it is not part of this three-item cohort. Output is limited to counts/status,
with no QA strings, stable record IDs, or hashes.

Drafted `docs/nextgqa_finite_claim_descriptive_protocol_v2.md` after an
independent design audit. It fixes the target to a finite three-item
description only, marks SESOI `not_applicable` rather than zero, prohibits
population/CI/p-value/threshold claims, writes the prompt and A–E tokenization
path precisely, and states that the cohort digest freezes membership but not
the usable claim packet or sampling plan. It retains all review-policy-v5,
source, real HTML5 clock, tokenizer, and atomic-preflight gates. The protocol
is still a draft and scoring remains no-go; an independent re-review of the
revised wording is pending.

The experimental CSV-to-target wrapper is synthetic-tested at **3/3**; its
combined source adapter tests pass, but it consumes declared token IDs and
prefix hashes rather than computing them and is not a production launch gate.
The independent wrapper/source audit found no mapping bypass. It confirmed
that the row digest covers every parsed CSV column and the lineage separately
binds exact CSV bytes; all results remain unvalidated and private.

Current verification: source/target/aggregate-audit/clock focused suites
**17/17**; full repository suite **967/967 in 9.370 s**; `compileall` and
`git diff --check` passed. The suite's printed gate failures are expected
negative-path fixtures. Hashes: source adapter
`f56d05ad69c302098834e96e6c2247b12b4a0db5352b2a263decb563cd5d38e4`, tests
`51bb62d64cf933f54dafd2d08d549c985dca895f77e2d431764dc8b5d1ef8c2c`, target
wrapper `20e47d7b70f18eb604fa6d741a3a6f7e54becde48c865468ced8bf42eb132bf8`,
wrapper tests `b3b20593a770d12e09c33777f375e6b305a24a4e5cbf15bf70e1742f32f3367f`,
aggregate CLI `d3111cfb3a06d8d5df60c93fd9a0b9995fb8f94f861302e028f79999b7f85bcc`,
CLI tests `c04896b42ebf3041821914a8996260431f3112466ba3ca91f413c383809feca9`,
clock helper `f7d0e76ce08828b672bf6837ba48ed2c3d4529982c7a96b24aa315c42c843f87`,
protocol v2 `2a650ed41d3ee79e57d00c38f3ab3d461cae11efbb94f1af09d720ed05ca03c6`.

### Faithfulness progress

- **Direct claim evaluation: `未実施` — 0/3 reviewed, 0/3 scored, and 0
  direct effects/CIs/tests** for the frozen NExT-GQA / Qwen2.5-VL-7B-Instruct
  claim-mask versus matched-control answer-margin contrast. This cycle does
  not change that evidence count.
- **Engineering/QA diagnostics only:** source-row mapping 3/3; full CSV gold
  matching 5,552 unique / 1 ambiguous / 0 missing; no tokenization or score
  has yet been computed for the actual target. The previous auxiliary
  QA-gold-interval estimates remain n=10, −0.1375 nat (16f) and −0.3375 nat
  (32f), with no interval/test and no claim-faithfulness interpretation.
- **Unresolved gates:** a valid complete claim packet and 3/3 independent
  reviews, actual player-clock capture, actual tokenizer/rendered-prefix
  verification, and production all-item atomic preflight. Local runtime
  currently has the processor snapshot and 13 GiB available RAM; sandbox
  reports CUDA unavailable and has no browser installed. GPU is not needed for
  source/tokenizer foundation work, and no direct-score GPU run is justified
  before these gates.
- **Next action:** audit the actual prompt/tokenizer boundary on CPU without
  emitting QA text or IDs, and investigate safe HTML5 clock-capture event
  collection for the existing review page. Continue to keep scoring no-go.

The large shared worktree still contains unrelated data and user changes. No
files were staged, committed, or published. Rights status remains unverified.

## 17:44 JST correction — active candidate cohort

At **17:43:46 JST**, **2h18m59s** elapsed and **3h41m01s** remain to
**21:24:47 JST**. While tracing the newest artifacts for tokenizer work, I
found that the preceding 17:40 source-adapter smoke used the earlier
`freshcohort` lineage, not the later valid claim-only cohort. Its full-dataset
CSV statistics remain valid, but its three-row result is not the active
candidate-cohort result and must not be cited as such.

Metadata-only reconciliation found the latest format-valid packet to be
`claim_generation_sentence_exploratory_20261016_01.json`: status passed,
requested/effective 3/3, format gate 3/3. Its cohort digest
`162fda51377768ae4e488fbfa1a8377f903ead3af7b277a17780b53e67b79a95` matches
the three-row sentence-exploratory lineage and worksheet; worksheet remains
`draft_not_reviewed`, `review_completed=false`, authorization not granted. The
separate later `sentence_repro` cohort was 2/3 format-valid; that is a distinct
format-reliability diagnostic and is not used to replace a row in this frozen
candidate. Neither packet has human review or any score outcome.

Reran the actual-source adapter CLI using the correct sentence-exploratory
lineage and worksheet. It passed internal source binding for **3/3** rows and
question/worksheet matches; the full test.csv counts remain **5,552 unique,
1 ambiguous, 0 missing** out of **5,553**. No output file, model call, or score
was produced. Updated protocol v2 to name this cohort and explicitly state
that its digest freezes membership only. It remains a draft/no-go until the
usable claim-input packet, human reviews and all other gates are validated.

This correction does not alter the faithfulness evidence count: **0/3 human
reviews, 0/3 direct scores, 0 effects/CI/tests**. Hashes for the corrected
protocol and adapter CLI were recalculated after this correction; the
full-suite result at the prior checkpoint remains **967/967**, and only
documentation changed after that test run.

## 17:49 JST checkpoint — production QA prompt and answer-token audit

At **17:49 JST**, **2h24m** of the authorized six-hour window have elapsed;
**3h36m** remain to **21:24:47 JST**. The previous 17:44 cohort correction is
the active data binding for all results below.

An independent audit found a frame-count ambiguity: the active QA-margin
runner defaults to 16 frames, while protocol v2 proposes 32. The CPU audit now
accepts an explicit frame count, defaults to the runner's current 16-frame
configuration, and was run against both **16** and the proposed **32** frames
for the same three source-bound QA rows. Both runs decoded all requested
frames, matched exact local media bytes to lineage, and completed the pinned
processor's full unmasked video path. They called no model weights, GPU,
network, or scoring path. At 16 frames, input token counts were 2,196, 2,360,
and 2,100 (6,656 total); at 32 they were 4,308, 4,640, and 4,116 (13,064
total). These are processor-prefix lengths only, not faithfulness outcomes.

To close the parity finding, the audit now calls the exact `build_qa_prompt`
and `process_qa_video` functions used by
`run_nextgqa_gold_interval_margin_pilot.py`; it does not maintain a duplicate
prompt or processor-call implementation. It also checks the pinned local
tokenizer on labels A–E with `add_special_tokens=False`, requiring exactly one
distinct token ID per label. The emitted summary contains only the count and
hash of the ordered token IDs. Regression tests include multi-token and
duplicate-ID failures. Focused audit plus QA-margin-runner tests pass
**13/13**. Both 16f and 32f actual CPU audits reported 3/3 source bindings,
3/3 media rows, and five valid distinct one-token labels.

The protocol readiness text now distinguishes this successful QA-margin
diagnostic from the still-unverified final claim-bearing prompt and intervention
inputs. This does not validate any of the five masked conditions, the generated
claim, evidence intervals, the actual player clock, or the atomic score gate.
The QA-margin prompt is not the sentence-generation prompt. A separate
independent auditor confirmed the cohort/media binding and privacy behavior,
and identified the missing label-token check; that check has now been added.

Current source hashes: prompt audit
`3e3006d197847f1cfab5970e8743e7fd1c8f988a69e5ec0352d342f4e5265043`,
production runner `7cc7fec7a5ddf610a0e7ab5d337e177512dda5934066b897a9d2164e0e1a9b3d`,
cohort `162fda51377768ae4e488fbfa1a8377f903ead3af7b277a17780b53e67b79a95`.
The 16f and 32f ordered label-token digest is the same; the ordered video token
digests differ as expected with frame count. No output artifact was written;
stdout was aggregate diagnostics only. Rights remain unverified, so all local
data and derived results stay private.

Direct claim faithfulness remains **0/3 human-reviewed, 0/3 scored, 0 direct
effects/intervals/tests**. The auxiliary QA-gold interval estimates remain
outside this evidence count. GPU scoring remains no-go: review and clock gates
are unresolved, and masked target inputs have not passed production preflight.
Next, test the production source/target integration and claim packet for exact
one-to-one mapping and determine whether the current HTML review handoff can
record an actual player clock capture without exposing answers or intervals.

Post-check: the first full-suite pass after adding the new audit found **2
classification-test failures** because its static scanner treats every
`from_pretrained` method as a possible weight loader. Classified the new entry
explicitly as `PROCESSOR_ONLY_LOADERS`, added a guard that it loads only
`AutoProcessor` and never a conditional-generation model, and reran the full
suite: **973/973 passed in 9.561 s**. `compileall`, `git diff --check`, and the
focused classifier tests **7/7** also pass. The review worksheet still
explicitly reports `review_completed=false`; no authorization is inferred.

## 17:54 JST checkpoint — review handoff and raw clock capture

At **17:54 JST**, **2h29m** elapsed and **3h31m** remained to the six-hour
deadline **21:24:47 JST**. A local independent audit found the existing v5
sentence-exploratory reviewer handoff ready for a person to begin: three opaque
videos, 4 files total, package directory mode 0700, files 0600, policy hash
matched, media bytes matched custody, and no audit failures. It remains an
unvalidated human handoff: worksheet `review_completed=false`, authorization
not granted, and zero completed human reviews. Reviewer identity,
independence, full viewing, and blinding are self-attested rather than
technically authenticated.

Primary-agent spot verification independently confirmed worksheet/lineage/
custody counts **3/3**, cohort consistency, all three staged-media byte
bindings, and the private mode/file-count constraints. The claim packet's
metadata lineage CLI also independently re-bound **3/3** selected records
against local benchmark metadata and video bytes, producing status
`metadata_lineage_crosschecked`; it did not inspect semantic support or
authorize a score. A newly rendered and packaged local handoff now includes
the added raw HTML5 clock observation UI; the output remains private under
`data/private_nextgqa/` and cannot authorize scoring.

The clock UI stores player `currentTime`, `performance.now()`, playback rate,
duration, pause/readiness state, visibility, staged-source boolean, and UTC
timestamp as raw observations. It exports a byte-integrity index separately.
Crucially, its payload states that entered anchor IDs are not bound to a
predeclared plan, the fit/held-out split is unassigned, PTS pairing is absent,
and authorization is not granted. Independent review confirmed this cannot be
mistaken by its own schema for a calibration result. It does not establish an
actual browser measurement; no supported browser executable is available in
this environment, and no browser capture occurred. Focused renderer, clock
validator and handoff tests pass **23/23** (the renderer tests were invoked
twice through module aliases); syntax and diff checks pass.

Direct faithfulness is still **0/3 human-reviewed, 0/3 scored, no effect
estimate, interval, or inferential test**. The 3/3 source, media, and prompt
bindings are readiness diagnostics only. Next action is to incorporate the raw
clock events into a separate, plan-bound custodian calibration audit after a
human captures them; until then, actual HTML5 calibration and the three human
reviews remain hard pre-score gates. Protected materials and all derivative
artifacts remain local; rights are unverified.

Post-check: extended each raw clock event with the actual muted flag, volume,
and disabled-text-track state, regenerated a fresh private reviewer page and
package so the artifact matches the current renderer, then reran independent
and primary audits. Renderer reproduction, offline CSP, absence of stable
source IDs and gold/seed fields in embedded data, event reset after media
change, private modes, and **3/3** staged-media byte bindings were verified.
These remain static/browser-source checks only; no browser execution or human
review occurred. Renderer/clock/handoff focused suites passed **23/23**, then
the full suite passed **973/973 in 9.376 s**. `compileall`, `node --check` on
the rendered JavaScript, and `git diff --check` passed.

## 18:00 JST interaction checkpoint — clock collection after review freeze

At **18:00 JST**, **2h35m** elapsed and **3h25m** remained to **21:24:47 JST**.
After review export, the UI now disables claim-review and media-selection
controls while keeping only `.clock-control` inputs/buttons active. This allows
the reviewer to lock and export claim judgments first, then record player-clock
observations against the already custody-verified, loaded media and export the
separate raw observation/index pair. An independent reviewer confirmed the
selector leaves exactly those controls enabled; the new HTML schema continues
to declare plan-binding unchecked, fit/heldout unassigned, PTS pairing absent,
and authorization not granted. No browser was actually run.

Created final package
`data/private_nextgqa/reviewer_package_sentence_clockcapture_20261016_03`
and safely removed the two superseded packages generated during this same
iteration after checking their exact paths, permissions, and contents. The
final package is 0700 with exactly four 0600 files; its HTML reproduces from
the current renderer and its three opaque media files match custody bytes.
The post-interaction full suite passes **973/973 in 9.491 s**; renderer,
clock-calibration, and handoff focused tests pass **23/23**; compilation,
rendered-JavaScript `node --check`, and whitespace checks pass.

Faithfulness remains **0/3 human-reviewed, 0/3 scored, no direct estimate or
uncertainty/test**. The current UI/package is operationally ready for a human
review plus raw clock capture, but it cannot validate the user-attestation or
convert clock observations into calibration without a predeclared plan and
PTS pairing. Next independent task is to draft the custodian-side plan-bound
converter for future raw observations; do not call it on synthetic evidence
as if that were a calibration result.

## 18:04 JST checkpoint — future clock-capture conversion path

At **18:04:25 JST**, **2h39m38s** elapsed and **3h20m22s** remained to the
recorded six-hour deadline. The reviewer handoff's clock download is
intentionally raw and does not pair event IDs with decoder PTS or assign
fit/held-out roles. Added `scripts/prepare_html5_clock_calibration_inputs.py`
to accept that exact raw payload, its integrity index, and a separately
hash-pinned PTS anchor plan. It requires three exact opaque-token rows, exact
planned event coverage, disjoint fit/held-out IDs, exact source/player state,
valid time ranges and contiguous event sequence numbers; then it emits the
existing calibrator's per-video capture schema into exclusive 0600 private
output. It does not load media or a model, score, authenticate browser
telemetry, or grant execution authority. Output explicitly states the plan's
pre-capture chronology is not authenticated.

Focused converter/calibrator tests pass **12/12** before the final atomic-write
hardening (rerun pending). Synthetic integration shows the conversion result
is structurally consumable by the calibrator, and synthetic data can produce
`heldout_pass=true`; this is not a browser or clock measurement. Independent
code audit found schema compatibility and no scoring/privacy bypass, while
reiterating that real capture provenance and plan chronology cannot be proven
from the files alone. The converter's output must only be called a candidate
calibration input bundle until a person records actual player events and an
independent custodian verifies the frozen PTS plan and media bytes.

The independent research-design audit also concluded that the 3-item finite
pilot can assess review/clock/input feasibility only; it cannot support an
effect claim or population generalization. Any later scale-up must fix its
estimand, video-level clustering, denominator/failure rules, uncertainty
method, sample-size/precision rationale, multiplicity, and pilot inclusion
before direct scores are accessed. Direct faithfulness remains **0/3 reviewed,
0/3 scored, no direct estimate/test**. The current hard blockers remain human
review, real clock capture, and production atomic target preflight.

After the independent converter audit, added contiguous event-sequence
validation and changed output creation to temp-file plus exclusive hard-link
so interrupted writes cannot leave a partial bundle at the final path.
Focused bundle/calibrator tests now pass **12/12**. Full repository suite after
the converter and clock-control changes passes **980/980 in 9.336 s**;
`compileall` and `git diff --check` pass. Re-rendered the current private
review package to `reviewer_package_sentence_clockcapture_20261016_04` after
the control-state fix, then removed the superseded package from this iteration
with exact path/mode/content checks. The current 04 package remains private:
four files, directory 0700/files 0600. No actual observations exist to
convert or calibrate; synthetic `heldout_pass` remains strictly a software
contract test.

CLI-level validation was then added: inputs must be private, owned 0600 regular
files, output must be an exclusive private path, and the saved bundle is
verified 0600 with explicit non-authorization. The focused converter suite is
**8/8**. The full repository suite after this addition passes **981/981 in
9.419 s**; `compileall` and `git diff --check` pass. An independent code audit
verified converter schema compatibility and its raw/untrusted status; it also
confirmed synthetic values can satisfy the empirical held-out calibrator, so
no synthetic result is reported as clock evidence.

## 18:11 JST checkpoint — scale-up preregistration foundation

At **18:11:11 JST**, **2h46m24s** had elapsed and **3h13m36s** remained to
the six-hour deadline. Three independent agents were used in successive waves
(never more than three concurrent child agents); their independent reviews
converged on the same boundary: the current three-item set is only a finite
feasibility pilot, while claim review, real HTML5 clock capture, and production
atomic target preflight remain open. No direct faithfulness outcomes were
consulted to choose future scale-up values.

Added `docs/nextgqa_scaleup_preregistration_skeleton.md` as an explicitly
unfrozen planning worksheet. It separates the current claim-mask estimand from
the closed head-intervention study and leaves population, estimand, clustering,
failure denominator, uncertainty target, sample-size rationale, multiplicity,
and pilot inclusion as explicit pre-score decisions rather than inventing
values from n=3. It adds an outcome-blind prior-exposure audit requirement.
Independently recomputed one relevant overlap check: current three candidate
video-byte hashes have **0/3** overlap with the prior 10-row answer-margin
cohort lock. This is not the complete prior-manifest exposure audit. QID alone
is unsafe as an exclusion key because the benchmark reuses QIDs across videos.

The entire CPU suite passes **981/981 in 9.559 s** after the source and review
handoff changes; compileall and `git diff --check` pass after the planning
document edit. Direct claim faithfulness remains **0/3 reviewed, 0/3 scored,
no direct effect estimate, interval, or test**. The new worksheet is planning
infrastructure only and does not close any scoring gate or authorize a model
forward. Keep protected artifacts local; no commit or publication is made.

An independent critique found an outcome-selection ambiguity in the first
worksheet draft. Corrected it: the three-item feasibility set is excluded by
default from any later confirmatory primary, and any reuse is a separate
exploratory finite description whose disposition must be fixed before those
pilot scores are generated/opened. The freeze is now explicitly before the
first scale-up score forward, with a custodian output-access log. Clarified
finite census vs probability-sample vs external-superpopulation targets,
pre-score supportability attrition vs post-freeze technical failures, and
that a finite census does not need an invented power target. A full local
prior-exposure audit across all older manifests has been commissioned as a
separate outcome-blind task; the 0/3 comparison against one prior 10-row lock
is only partial evidence.

## 18:22 JST checkpoint — prior exposure and diagnostic target candidate

At **18:22:28 JST**, **2h57m41s** had elapsed and **3h02m19s** remained to the
deadline. The exposure audit scanned 11 tracked plus four untracked NExT-GQA
sample manifests, 48 private JSON manifests, 381 available result JSON/CSV
files, and 40 experiment text/manifest files. Its row-level identity scan
found no match among sample/result rows with identities, and the current
candidates have 0/3 overlap with the one available 10-row answer-margin lock.
However, four private exclusion lists contain candidate hashes; one
118-entry list `exclude_prior_all_20261016_02.json` contains **3/3**. I
independently recomputed that overlap. The contemporaneous 15:35 log describes
the 118-entry list as all videos used by unrelated historical experiments and
says it was replaced with a claim-generation-only list before the smoke. Its
15:33:37 mtime is consistent with that chronology, but the file has no Git
history or embedded provenance, and no construction source was found.
Therefore prior exposure is plausible but not fully authenticated or
exhaustive; keep these three quarantined from any scale-up/confirmatory cohort.
The 118-list is not promoted to a certified complete denylist.

The actual-source target-candidate builder has been implemented and its first
private run produced 3/3 32-frame unmasked QA target rows. Independent review
found three hardening items (future tokenizer-file coverage, source-to-Qwen
pixel-order check, and supplied-protocol identity validation). Candidate `_01`
is diagnostic-only and will be superseded by a unique `_02` after repair;
neither is score-ready. Focused tests pass **5/5**, loader guards pass **8/8**,
and the full suite passes **987/987 in 11.171 s**. Direct faithfulness remains
0/3 reviewed, 0/3 scored, with no effect estimate, interval, or test. No GPU
score was run.

## 18:24 JST checkpoint — diagnostic target binding hardening

Repaired three gaps in the private, CPU-only diagnostic candidate builder.
The tokenizer fingerprint now includes optional `added_tokens.json`. Before
processing frames, the builder requires the exact byte identity of the current
protocol v2 draft and explicitly checks its draft status and fixed 32-frame
specification. It now compares each decoded RGB frame's channel-mean signature
against the corresponding Qwen utility tensor signature, including order, using
the same tolerance as the prompt/tokenizer audit. The artifact explicitly
states that `processor_input_ids_sha256` covers only `input_ids`, not video
pixels or metadata; no full tensor digest is claimed.

Focused tests pass **16/16**; the full repository suite passes **990/990 in
9.562 s**. `py_compile` and `git diff --check` pass. New negative tests cover
protocol identity/status/frame-count edits, Qwen pixel signature mismatch or
frame reordering, and tokenizer-fingerprint sensitivity to `added_tokens.json`.

Rebuilt the current 3-row source-bound candidate into a new exclusive artifact,
preserving `_01`: `data/private_nextgqa/diagnostic_target_candidate_20261016_02.json`.
It is **0600**, 9,010 bytes, SHA-256
`83f5df56cc664339b691ee38a5f076d3b3a79ae30b0b8062f59807eee6b4a5a6`, status
`diagnostic_candidate_unvalidated`, and binds 3/3 source rows. It reports zero
weight loads, CUDA, network, model forwards, or score calls, and authorization
`not_granted`; the claim-input manifest is explicitly absent. This remains a
diagnostic of the **unmasked QA prompt/processor path**, not a reviewed claim
target, masked intervention tensor, clock calibration, or launch gate. Human
review and direct scoring remain **0/3**. Keep this protected artifact local;
it is not score-ready.

At **18:26:40 JST**, **3h01m53s** had elapsed and **2h58m07s** remained to
the six-hour deadline. Primary independently rechecked `_02` aggregate
metadata and digest: mode **0600**, **9,010 bytes**, 3/3 records, 32 frames
each, explicit `input_ids_only_not_pixels_or_video_metadata` hash scope,
review false, no claim-input manifest, and zero weights/CUDA/network/
forwards/scores with authorization not granted. A separate auditor re-reviewed
the repaired source and tests, confirmed all three fixes, and reiterated that
the channel-mean check (tolerance 2) is a coarse pixel/order diagnostic rather
than full tensor equality. Primary reran the full suite: **990/990 in 9.494 s**;
`compileall` and `git diff --check` pass. The research conclusion is unchanged:
this does not test claim-specific faithfulness.

## 18:33 JST checkpoint — synthetic mask conditions and independent boundary review

Implemented `scripts/nextgqa_claim_mask_conditions.py`, a pure CPU helper that
constructs full, reviewed-evidence black-mask, and equal-count temporal-control
RGB frame lists from caller-supplied 32-frame RGB samples, exact full-video PTS,
frozen sampled indices, and intervals. It validates the endpoint-inclusive
sampler, uses the shared inclusive interval assignment and rank-distance control
tie-break, preserves input pixels, and black-fills only assigned whole frames.
Its optional metadata is aggregate/numeric and pixel-hash based. It explicitly
does not authenticate review, media/time alignment, clock calibration, media
duration, score authorization, or execution authorization.

An independent auditor found that an attempted generic `end <= final PTS`
validation would wrongly reject valid annotations in the final frame's display
duration, because final decoded PTS need not equal playable media duration.
That change was withdrawn before integration. The caller must validate the
review interval against verified media duration and calibrated clock bounds;
the PTS map alone is insufficient. The audited shared assignment remains
unchanged. The helper test now records this limitation by accepting a span that
extends beyond final PTS when no separate media-duration bound is supplied.

Focused helper tests pass **6/6**, PTS utility tests **16/16**, and the full CPU
suite passes **996/996 in 9.805 s**. `py_compile` and `git diff --check` pass.
The first full-suite run exposed one expected-status assertion coupled to the
withdrawn check; after correcting the test to use an in-duration unsampled span,
the full suite passed. These are synthetic code-path checks, not real-media or
faithfulness results.

Independent primary-source review of perturbation/masking literature led to a
worksheet clarification: the proposed probe is removal-based necessity, not
sufficiency; black masking can create OOD/discontinuity artifacts; equal-count
controls do not match content/information value; and exact scale-up score and
contrast remain TBD. The existing gold-answer runner's next-token gold-vs-best-
foil log margin is described as a candidate only, not silently imported as the
generated-claim estimand. These are scope/interpretation constraints, not
evidence about model behavior.

An outcome-blind audit found enough surviving identity-bearing sources to build
a bounded private prior-exposure manifest, but not to certify complete history:
aggregate-only results, missing mappings, untracked historical inputs, and the
118-entry exclusion list's unknown construction cannot be recovered. A child
agent is implementing a synthetic-tested, explicit-input HMAC builder with
unresolved-source accounting; it will not process real/private rows in this
implementation task. Direct faithfulness remains **0/3 reviewed, 0/3 scored**,
with no effect estimate, interval, or test. No GPU score or publication was
made.

At **18:33:01 JST**, **3h08m14s** had elapsed and **2h51m46s** remained in the
six-hour window. Continue by auditing the exposure-builder implementation and
its privacy behavior, then validate whether remaining human-review/clock gates
can be resolved locally without opening model scores.

## 18:39 JST checkpoint — exposure index hardening and clock gate audit

The user asked whether only three subagents can be used and requested continued
groundwork. The platform provides **four concurrent slots including the primary**,
so up to **three child agents at once**; this is a concurrency cap, not a total
agent limit. Work has been rotated through independent literature, design,
implementation, and audit tasks in successive waves, with no more than three
children active concurrently.

The first synthetic prior-exposure-index implementation was independently
audited. Repairs now fail closed on ambiguous/missing result-row containers and
malformed CSV row widths; unknown row counts remain null instead of being
reported as zero. Repeated byte-identical source content is rejected, source
fingerprints are HMACed, and per-source ID namespaces are required and included
in video-ID and video/QID HMACs. The manifest separates row coverage for any
identity, video ID, media SHA-256, and video/QID pair. Opaque legacy values are
not interpreted and their identity-row coverage remains null. The caller must
still validate that a supplied video ID belongs to the declared namespace;
there is no dataset crosswalk inside the builder. It cannot recover aggregate-
only, deleted, or provenance-free history and makes no completeness claim.

Synthetic tests now pass **10/10**, including duplicate content, malformed CSV,
ambiguous JSON, keyed fingerprints, private 0600 exclusive output, path-only
unresolved rows, opaque lock-QID treatment, and namespace separation. The full
CPU suite passes **1006/1006 in 9.800 s**. These results validate code paths,
not the actual historical exposure inventory; no real/private source files
were passed to the new tool.

An independent review of the human-review/browser-clock path found no actual
human review or browser capture. To proceed, an independent reviewer must
review all three staged videos, record every support decision and interval,
and export the private JSON/index. Clock calibration separately requires an
independently recorded pre-capture anchor plan tied to exact staged media,
same-event HTML5 observations, and held-out residual review. Existing software
only asserts some freeze/status fields and accepts a plan hash; it does not
prove that plan was independently recorded before observations. Synthetic
clock tests do not substitute for this human/browser work. The final PTS also
cannot by itself validate the browser interval's media-duration endpoint.

At **18:39:55 JST**, **3h15m08s** elapsed and **2h44m52s** remained. Continue
with an independent code audit of the exposure builder and, if it passes,
decide whether a strictly scoped local inventory can be safely run from known
inputs. Direct faithfulness remains **0/3 reviewed, 0/3 scored**, with no effect
estimate, interval, or test; no GPU score has been run.

## 18:45 JST checkpoint — bounded local prior-exposure index

After the builder passed independent synthetic review, I ran it locally on 42
explicit source paths (41 unique byte contents; one exact duplicate lineage
file was omitted and recorded in the private input registry). Inputs were
limited to seven NExT-GQA sample manifests, 19 claim/target lineage-family
artifacts, one answer-margin lock, three identity-bearing result files, and 11
legacy exclusion artifacts. The builder emitted only keyed HMAC identities,
source fingerprints, counts, and static unresolved reasons. The raw source
paths are stored only in a separate mode-0600 local input registry; the HMAC
key, registry, and output are all mode **0600** under a mode-0700 directory.

The private artifact is
`data/private_nextgqa/prior_exposure_partial_20261016_01.json` (60,492 bytes,
SHA-256 `e44b34bfc2d7374a2271621982986fd1ecf402039f94c7d245e7e59e937d9fca`).
Its source registry is `prior_exposure_inputs_20261016_01.json`; the key is
`prior_exposure_hmac_key_20261016_01.bin`. These remain local and ignored by
Git. Primary independently recomputed all 41 source-file HMAC fingerprints
against that registry, confirmed the exact output schema contains no path
fields, and reproduced source/status/identity counts and artifact digest.

Among parsed sources, the row-count sums are **332 known rows**, **330 rows
with some usable identity**, **277 rows with explicit video ID**, **59 rows
with media SHA-256**, and **290 rows with a video/QID pair**. Those are
source-row counts across overlapping manifests, not unique videos or an
exposure rate. The global HMAC sets contain **212 distinct explicit video-ID
tokens**, **21 distinct media-byte tokens**, and **225 distinct video/QID pair
tokens**. The 11 legacy exclusion files remain `opaque_unresolved`; their row
counts and identity contributions are unknown in this adapter. Therefore this
index is a partial inventory of the 41 supplied source contents, **not** a
complete historical exposure certificate. It does not authenticate the
118-entry exclusion list's origin or make its candidates safe for reuse.

The experiment remains exploratory infrastructure work: direct faithfulness
is still **0/3 reviewed, 0/3 scored**, and there is no direct effect estimate,
interval, or test. No GPU score was run. An explicit `media_sha256` set adapter
is now being added with a separate unverified-provenance status so legacy
exclusion overlaps can be checked without upgrading their authority.

At **18:45:14 JST**, **3h20m27s** had elapsed and **2h39m33s** remained.

## 18:50 JST checkpoint — parse legacy media sets without promoting provenance

Added the `media-set-json` adapter to the bounded HMAC index. It accepts only
an explicit `media_sha256` array with narrow optional metadata fields, never
emits those metadata values, and labels each such source
`parsed_provenance_unverified`. A new local `_02` manifest and private input
registry were created with exclusive writes; `_01` remains unchanged. The
41-source registry contains 30 explicit-schema sources and 11 media-set
sources. The latter contain 177 rows / 177 distinct media identities. Across
the partial inventory, known rows sum to 509, and global HMAC sets contain 212
video-ID tokens, 118 media-byte tokens, and 225 video/QID-pair tokens. Row sums
cover overlapping artifacts; these totals are not unique-item exposure rates.

I independently rebuilt the `_02` manifest from its exact registry and key;
the regenerated object and serialized bytes matched the private artifact. I
recomputed all 11 media-set HMAC contributions, confirmed no raw media SHA or
optional purpose value is emitted, and recomputed the candidate comparison:
the three sentence-lineage candidates overlap **3/3** with the exact 118-entry
media list. That list is still provenance-unverified, so this is an identity
intersection only; it neither proves a viewing event nor establishes history
completeness. A separate scan found four review-custody manifests omitted from
the registry. They carry media hashes but do not prove anyone viewed the
videos; a narrowly scoped adapter is being independently audited before any
further local inventory.

The complete Python script suite passed **1011/1011** in 9.830 s; focused
exposure/mask/time tests passed **37/37**, compilation and `git diff --check`
passed. The first focused test invocation used an incorrect import root and
failed to import two modules; rerunning with the repository's `PYTHONPATH=.:scripts`
passed. No GPU inference or score ran. Current direct generated-claim
faithfulness remains **0/3 reviewed, 0/3 scored, 0 effect estimates, 0
confidence intervals, 0 inferential tests**. Human review and real browser
clock calibration remain open external tasks.

At **18:50 JST**, **3h25m** had elapsed and **2h35m** remained in the authorized
six-hour window. The next useful step is to verify whether the omitted custody
records can add media-identity linkage without treating package inclusion as
review completion, then continue resolving the pre-score estimand and coverage
gaps. All generated private manifests/configuration/key material remain local
and ignored by Git.

## 18:52 JST checkpoint — custody-package hash linkage

The narrow custody adapter passed its independent code audit and synthetic
17/17 tests. It accepts only `source_media_sha256` from custody record rows,
ignores review tokens and all other metadata, and marks outputs
`parsed_provenance_unverified`; package membership cannot establish that a
reviewer opened or watched a video. Four surviving custody manifests were
then added to a new `_03` private inventory (45 explicit paths, 45 unique file
contents). It reports 30 explicit-schema sources and 15 provenance-unverified
media/custody sources. The global identity sets remain 212 video-ID tokens,
118 media-byte tokens, and 225 video/QID-pair tokens. The four custody records
contain six distinct source-media hashes; all six are already within the
partial media identity union. Candidate media intersects the custody hashes
3/3 and the 118-entry legacy list 3/3. Neither overlap is evidence of actual
human viewing, score exposure, or complete prior-use history. The inventory
still has `historical_completeness_authenticated=false`.

The complete suite now passes **1013/1013** in 9.628 s; `git diff --check`
passes. No inference or GPU score ran. Direct generated-claim faithfulness is
still **0/3 reviewed, 0/3 scored, 0 effect estimates, 0 confidence intervals,
and 0 inferential tests**. The worksheet now links the candidate contrast to
the existing contract (control minus claim-linked-mask fixed-choice margin)
and makes explicit the unresolved population, supportability, source-cluster,
uncertainty, and threshold decisions. These changes are planning only and
authorize no score.

At **18:52 JST**, approximately **3h27m** had elapsed and **2h33m** remained
to the authorized deadline. Next, audit the protected source families for
remaining identity-bearing omissions and resolve the pre-score estimand and
denominator language against the existing protocol contract. Keep all
protected artifacts and HMAC keys local.

## 18:54 JST checkpoint — reconcile protocol denominator and production gap

Independent review found and corrected a worksheet/contract mismatch. The
contract's current all-eligible estimand cannot treat unsupported, ambiguous,
unreviewed, or zero-overlap records as removable screening attrition: any such
record in the frozen eligible cohort blocks the primary score run. The
worksheet now separates screened/sampled-stage completion counts from the
eligible-manifest analysis denominator and states that a supportable-claims-
only estimand would require a separately frozen protocol. The proposed score
and contrast agree with the existing contract; population, sampling
provenance, source clustering, uncertainty target, and SESOI remain unresolved.

An independent gate audit inspected 56 protected JSON artifacts without
printing row values: **0/8** status-bearing review worksheets report completion,
and there are **0** browser observations and **0** held-out clock-calibration
artifacts. The three-video claim-mask helper is currently a standalone
condition constructor, not wired into a production scorer. Existing review
assignment, target, timing, and synthetic gate components do not authenticate
the absent human review/browser capture, nor perform the production all-record
atomic preflight. The NExT-GQA gold-interval margin runner measures a different
claim-independent task and cannot substitute. Thus no scoring command is
ready or justified; current direct generated-claim faithfulness remains
**0/3 reviewed, 0/3 scored, 0 effect/CI/test**.

The next concrete execution gates are: a blinded independent human completes
all three reviews and exports the private review record; a custodian freezes
the media-bound decoder-PTS anchor plan before observations; same-player
HTML5 anchors are captured and held-out residuals checked; then source/media,
duration, review, clock, frame/control assignment, and proposed three-record
membership must pass one production atomic preflight before any model forward.
The mask
helper is only a piece of this path. These external observations cannot be
replaced by more synthetic tests. At **18:54 JST**, about **3h30m** elapsed
and **2h30m** remained in the requested six-hour window. Continue with
outcome-blind implementation/protocol audits that prepare this exact path,
without claiming direct experimental progress.

## 18:56 JST checkpoint — v2 pilot is still a draft

Independent review of the implementation and protocol found an important
scope distinction: the three-item `nextgqa_finite_claim_descriptive_protocol_v2`
is explicitly a design draft, while the schema-v13 contract describes a
separate all-eligible candidate estimand. Neither implies that the candidate
membership alone freezes the scoring cohort. The worksheet now calls the
three rows candidate membership, requires joint freeze of protocol, valid
claim packet, target, cohort and code before any score, and conditions the
9-forward count on selecting and freezing v2. It also distinguishes the v2
no-replacement proposal from the schema-v13 denominator rule. This closes the
wording ambiguity without changing the metric or reading outcomes.

The prior-exposure row now reflects `_03` (45 distinct contents; 30 explicit-
schema sources and 15 provenance-unverified media/custody sources) and links
to the 18:52 checkpoint. The independent artifact audit confirms the 45-source
registry reproduces `_03` exactly; four custody sources contribute 12 rows
collapsing to six media identities, with no raw-value leakage. The candidate
intersection remains 3/3, but custody membership does not establish human
viewing. The full suite remains **1013/1013** from 18:52; after these latest
documentation-only edits, `git diff --check` passes. Direct faithfulness is
unchanged at **0/3 human review, 0/3 score, 0 effect/CI/test**.

At **18:56 JST**, **3h31m** elapsed and **2h29m** remained to 21:24:47. The
next check is an independent audit of the final worksheet/log wording and a
scoped source publication review. Do not score until the external review and
actual clock gates are completed and the complete protocol is frozen.

## 19:01 JST checkpoint — updated related-work boundary

An outcome-blind primary-source scan found two close 2026 image-VLM
counterfactual methods: vCT/vCCT links removed visual objects to changes in
prediction distributions and whether CoT mentions them; EDCT extracts concepts
cited in an explanation, applies verified image edits, and checks answer and
explanation consistency. These make any broad “first explanation-driven
counterfactual faithfulness test” claim untenable. The nearest video work in
this scan evaluates evidence-corruption/temporal-intervention reliability or
video-swap/visual-degradation effects, rather than a human-reviewed temporal
span aligned to a generated textual claim. STEAR also uses spatiotemporal
counterfactual evidence intervention for mitigation, a distinct goal and
mechanism. Sources: [vCT/vCCT (arXiv:2609.06704)](https://arxiv.org/abs/2609.06704),
[EDCT-Bench (arXiv:2609.17953)](https://arxiv.org/abs/2609.17953),
[INFACT (arXiv:2603.11481)](https://arxiv.org/abs/2603.11481),
[Chains That See, Answers That Don't (arXiv:2606.22862)](https://arxiv.org/abs/2606.22862),
and [STEAR (arXiv:2604.03045)](https://arxiv.org/abs/2604.03045).

Bounded inference: this targeted scan suggests a possible narrower difference
in the video/temporal interval and claim-level review combination, but it is
not a systematic novelty search and cannot support a priority claim. The
design should be framed as an adaptation/test of interval-specific evidence
dependence, with black-mask and rank-matched temporal-control limitations
explicit, until further literature review and real-data results exist. No
experimental evidence was added; direct faithfulness remains **0/3 reviewed,
0/3 scored, no effect, CI, or test**.

At **19:01:21 JST**, **3h36m34s** elapsed and **2h23m26s** remained to the
21:24:47 deadline. Continue with the primary-source dataset/clock review and
final verification; keep remote publication disabled for protected-derived
artifacts under the local-only directive.

## 19:04 JST checkpoint — independently recomputed auxiliary GPU pilot

Reopened the two local auxiliary score JSONs directly and recomputed the
record-level aggregates without exposing record identifiers or text. Both runs
report `passed`, 10/10 completed records, 50/50 requested model forwards, and
zero failures. The 16-frame run reports primary black-control-minus-gold-
interval margin mean −0.137500 nat (sample SD 1.596926; range [−3.000000,
+3.000000]); the 32-frame run reports −0.337500 nat (SD 1.908543; range
[−4.000000, +1.625000]). Their ordered record assignments match, and the
paired change in this same primary contrast from 16 to 32 frames is −0.200000
nat (sample SD 1.532291; 6 positive, 4 negative, 0 ties). The 16-frame file
records a cohort digest; the 32-frame file records a cohort digest and a lock
digest. The metadata do not establish that both settings used one shared frozen
lock. Neither artifact supplies a CI or p-value;
both explicitly state `claim_review_completed=false` and a null
`claim_specific_faithfulness_estimate`. This is QA dependence on dataset gold
intervals, not generated-claim faithfulness, and it supports no population
inference.

Runtime artifacts identify an RTX 5070 Ti, peak allocated VRAM 9.06 GiB for
16 frames and 11.43 GiB for 32 frames, and elapsed times 135.85 s and 235.51 s.
These are verified auxiliary GPU workloads, not progress on the blocked direct
claim-faithfulness estimand. A first audit command used `python` which is not
on PATH; rerunning with repository `.venv/bin/python` succeeded. The full test
suite remains 1013/1013 from the independent 18:52 run; this checkpoint changes
only the experiment log. At **19:04:22 JST**, 3h39m35s had elapsed and 2h20m25s
remained to the 21:24:47 deadline. Continue with independent protocol and
artifact audits; do not start direct scoring without the human and browser-clock
gates.

## 19:07 JST checkpoint — capture UI now records presented-frame metadata

The independent clock audit found that the browser button sampled `currentTime`
without identifying the actual compositor-presented frame. Implemented a bounded
capture improvement in the local review HTML: where
`requestVideoFrameCallback` is available, anchor observations now carry the most
recent frame `mediaTime`, `presentedFrames`, presentation/expected-display
timestamps, dimensions, and callback time. Missing metadata is explicitly marked;
export metadata state that decoder-frame identity and PTS pairing remain
unverified. UI instructions and the v2 protocol now state that `currentTime`
alone cannot pass calibration, and unresolved browser-frame/decoder-frame
correspondence fails the clock gate. The scale-up worksheet also requires the
browser/decoder versions, stream/timebase, PTS-origin rule, temporal coverage,
and treats held-out max residual only as an empirical diagnostic, not an
uncertainty guarantee.

Validation: the focused HTML renderer, clock-input, and clock-converter tests
pass **20/20**; synthetic rendered-page JavaScript passes Node syntax checking;
`git diff --check` passes. This does not exercise an actual browser or validate
a frame-to-decoder correspondence. The prior audit exposed a limited design
not yet implemented/authenticated in the converter: no frozen beginning/interior/end coverage check or browser/decoder identity verification exists. The direct score gate remains closed. At **19:07:15 JST**, 3h42m28s had elapsed and 2h17m32s remained to the requested deadline; continue independent review and integration checks.

## 19:12 JST checkpoint — clock schema integration and digest repair

A second independent UI audit found that the first metadata patch still allowed
anchor clicks during playback/seeking, could report last-callback metadata as if
fresh, and produced JSON rejected by the strict calibration-input converter.
The renderer now requires paused, non-seeking, frame-ready state; labels the
metadata only as `last_presented_frame_metadata_unverified`; emits observation
schema v2; and records `seeking`. The converter accepts only v2 observations
with the callback metadata, paused state, and no active seek. It feeds rVFC
`mediaTime` (not click-time `currentTime`) to the empirical fit, preserves
`currentTime` only as a raw diagnostic, and keeps stale-frame/decoder-identity
limitations in the bundle. Missing metadata is rejected. These checks make the
format compatible but do not authenticate callback freshness or same-frame
identity, so the bundle remains explicitly unvalidated and cannot authorize
scoring. No browser binary or Playwright/Selenium package is installed here;
actual browser behavior remains untested.

The first full-suite run after editing the protocol text failed four diagnostic
candidate tests because the builder intentionally pins the exact protocol SHA.
This was caused by my own protocol edit, not fixture mutation or a pre-existing
failure. I recomputed the exact file digest and updated the builder's expected
constant; all four focused candidate tests then passed. Final focused UI/converter/
candidate tests pass **28/28**, and the full suite passes **1013/1013** in 9.75 s.
Rendered synthetic HTML also passes Node syntax validation; `git diff --check`
passes. Only source/schema compatibility and synthetic checks were exercised;
there is still no real browser clock capture and direct faithfulness remains
**0/3 reviewed, 0/3 scored, no direct estimate/CI/test**. At **19:12:05 JST**,
3h47m18s had elapsed and 2h12m42s remained. Continue with a whole-diff audit and
research-planning review, preserving all protected data locally.

## 19:13 JST checkpoint — guard the clock conversion input

Added a regression test where click-time `currentTime` intentionally differs
from the associated rVFC `mediaTime`. The conversion bundle follows `mediaTime`,
retains no click-time value in the calibration anchor, and preserves the stale-
metadata limitation. The focused renderer/converter/calibrator suite passes
**21/21**, including rejection tests for absent frame metadata and invalid
paused/seeking state. This verifies transformation behavior only; it does not
validate the event/frame match itself. At **19:13:41 JST**, 3h48m54s had elapsed
and 2h11m06s remained. Next: independent audit of final clock claims and exact
allowlist of any local commit; no push while protected-derived material is under
the local-only directive.

## 19:14 JST checkpoint — related-work boundary widened

A primary-source check added *Journey Before Destination* (EACL 2026) to the
near-neighbor review. It evaluates generated image-reasoning chains at the
perception-step level with VLM judges, validates judge ratings against human
annotations, and reports image benchmarks; it directly rules out a broad
novelty claim about visual faithfulness of generated intermediate claims, but
is not the same video temporal-mask intervention. INFACT evaluates answer
reliability under evidence corruption and temporal interventions in video QA;
VideoSEAL studies temporal/semantic evidence alignment in agentic long-video
QA. They set a high bar for describing this project as a narrow temporal
claim-interval intervention test rather than a general new faithfulness
framework. The scale-up worksheet now records that boundary and labels the
remaining difference as a hypothesis, not novelty evidence. Sources:
[Journey Before Destination (EACL 2026)](https://aclanthology.org/2026.eacl-long.194/),
[INFACT](https://arxiv.org/abs/2603.11481),
[VideoSEAL (ICML 2026)](https://proceedings.mlr.press/v306/qiu26v.html).

This was a targeted scan, not systematic review and not proof of novelty.
No real-data or direct-faithfulness evidence changed. At **19:14:40 JST**,
3h49m53s had elapsed and 2h10m07s remained. Continue with audit and safe
integration; the hard review/clock gate still prevents direct scoring.

## 19:16 JST checkpoint — align novelty and intervention claims to prior work

A separate literature audit confirmed the relevant scope: *Journey Before
Destination* evaluates image-reasoning perception steps with model judges and
human meta-evaluation; INFACT applies video evidence corruption and temporal
interventions at the QA-answer level; VideoSEAL is a close prior for video
answer/evidence alignment, combining temporal-overlap grounding and semantic
support audits for an agent's inspected evidence. Thus any possible distinction
here is specifically the proposed fixed-answer-margin contrast between a
human-reviewed claim-support interval and an equal-count temporal control—not
temporal grounding or video faithfulness generally. This remains an untested
novelty hypothesis, not a priority claim. Sources: [Journey Before Destination
(EACL 2026)](https://aclanthology.org/2026.eacl-long.194/),
[INFACT](https://arxiv.org/abs/2603.11481),
[VideoSEAL (ICML 2026)](https://proceedings.mlr.press/v306/qiu26v.html).

Also added *Faithfulness Under the Distribution* (ICLR 2026) to the limitation
section: its authors state that zeroing pixels in common deletion/insertion
metrics can move inputs out of distribution and distort model behavior. This
supports reporting the current proposal strictly as sensitivity to black-fill
masking; a matched temporal control does not remove visual/semantic mismatch or
OOD concerns. Any future in-distribution replacement is a separate intervention
requiring a new frozen protocol and confirmation. [FUD (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/409fcc9d24b549969b8b9be68b56a7be-Abstract-Conference.html).

The literature edits changed only the scale-up worksheet/log. No experiments or
direct claims changed; direct score gate is still **0/3 reviewed, 0/3 scored**.
At **19:16:37 JST**, 3h51m50s elapsed and 2h08m10s remain. Continue final
independent source/diff audit and proportionate testing.

## 19:17 JST checkpoint — full suite after literature and clock changes

The complete local test suite passes **1014/1014** in 10.04 s after the latest
converter test and literature-note changes. Node syntax validation, compileall,
and `git diff --check` also pass. The suite's logged child-process failure
messages are expected negative-path fixtures; the unittest runner exits OK.
No GPU/model calls occurred during this checkpoint. At **19:17:04 JST**,
3h52m17s elapsed and 2h07m43s remain to the 21:24:47 deadline. Continue with
 scoped diff and commit readiness review; protected outputs remain unpushed.

## 19:20 JST checkpoint — close empirical-residual/clock-validation confusion

Independent audit found a naming and integration weakness: the HTML5
calibrator's `heldout_pass` meant only that the observed maximum residual was
below a tolerance. The assignment planner accepted that bit with an
`empirical_heldout_residual_only` scope and could therefore emit a structurally
passed, still-unvalidated interval plan without any separately reviewed
capture/frame-correspondence evidence. It never called the model
(`score_calls=0`, authorization not granted), but this could mislead planning.

Renamed the calibrator output to `empirical_residual_within_tolerance` and made
the planner additionally require a caller attestation that independent review
is claimed plus two evidence hashes (capture and decoder-frame correspondence).
Tests show the empirical-only artifact and each missing declaration/hash fail
closed. A second independent audit confirmed that fabricated declarations and
syntactically valid arbitrary hashes can still pass the planning seam. I
therefore named the field `independent_review_claimed_unverified` and clarified
that this is only an accidental-misuse guard: the planner does not read or
authenticate evidence, and a signed evidence-reading audit manifest remains
necessary. The synthetic success fixture is expressly synthetic. Real
browser/frame calibration remains absent, so the production gate remains
closed.

Focused calibrator, assignment, and converter suites pass **33/33**. The first
attempt used an incorrect unittest import path and failed before loading any
tests; rerunning with `PYTHONPATH=scripts` passed. No GPU or protected dataset
was used. Direct faithfulness remains **0/3 reviewed, 0/3 scored, no estimate,
CI, or test**. At **19:20 JST**, 3h55m13s elapsed and 2h04m47s remain to the
21:24:47 deadline. Next: full suite, independent review of the changed gate and
publication/local-commit safety; do not push protected-derived logs/artifacts.

## 19:23 JST checkpoint — repair protocol digest pin and rerun full suite

The first full-suite attempt after clarifying that caller-supplied verification
fields are unauthenticated exposed four candidate-builder test failures: the
protocol text had changed but its exact-byte SHA pin had not. Updated the pin
to the file's current SHA-256 (`1b0017be…03a0`); an independent check confirmed
the bytes and constant match. The relevant candidate/calibration tests pass
**41/41**, and the full suite now passes **1015/1015** in 9.65 s. Compilation
and `git diff --check` pass as well. The negative-path error messages printed
by subprocess fixtures are expected and the unittest command exits successfully.

The clock-planning repair remains a structural safeguard only: caller-made
attestations and hash-shaped strings can still pass to an
`unvalidated_planning_only` artifact, which has zero score calls and no
authorization. Actual evidence authentication remains a hard open gate.

Also hardened the standalone calibrator against duplicate JSON object keys,
so direct callers cannot rely on Python's last-key-wins parsing even if they
bypass the stricter observation converter. A regression test covers duplicate
`schema_version`; the focused clock/assignment/converter suite passes **34/34**
and the full suite passes **1016/1016** in 9.92 s. No data or model was loaded.
At **19:24:40 JST**, **3h59m53s** elapsed and **2h00m07s** remain to the
21:24:47 deadline. Continue with independent UI/protocol/reproducibility audits
and review the explicit local-only change set; do not publish protected-derived
material.

## 19:26 JST checkpoint — output the full margin decomposition

An independent estimand audit confirmed the draft is coherent only as a
three-item finite descriptive masking-sensitivity contrast, not claim truth,
internal reasoning, or population evidence. It identified a useful
interpretability risk: the best foil in `gold logP − best-foil logP` can change
between conditions. The draft now requires retaining all five choice
log-probabilities and best-foil identity per item/condition in the private
0600 artifact, while keeping the fixed primary contrast unchanged and
excluding answer labels from public-safe aggregates. The pinned protocol SHA
was refreshed and will be checked with the candidate-builder tests.

A separate source-only provenance audit found that the current chain is not an
integrated reproducible handoff: renderer observations do not carry the full
worksheet/custody/media-check references, the calibrator preserves only a
capture digest, and the planner needs fields the calibrator does not emit.
Converter→calibrator and planner are tested separately with synthetic fixtures,
not as one connected artifact chain. This does not create scoring authority;
it means actual clock evidence could not yet be used honestly end-to-end. The
next useful foundation is a versioned local provenance bundle binding renderer
worksheet, custody, observation/index, frozen anchor plan, converted capture,
calibration and assignment; a synthetic end-to-end test should reject swapped
tokens/media and any missing edge. Keep the chain unvalidated until a real
browser and independent frame/PTS evidence exist.

No protected results, GPU, or direct claim scores were used for these audits.
At **19:25:42 JST**, **4h00m55s** elapsed and **1h59m05s** remain to the
21:24:47 deadline. Continue the bounded provenance/design work and retain the
zero-direct-score status.

## 19:28 JST checkpoint — audit auxiliary pilot pairing provenance

Reopened the private 16- and 32-frame score JSONs locally and verified both
have 10 records, 50/50 requested forwards, zero failures, and the logged
record-level effect fields. Their means/SDs/ranges/sign counts reproduce the
existing aggregate. The 32-frame preflight digest matches its score and its
cohort-lock digest; the lock's source score SHA and source cohort digest match
the 16-frame score bytes/metadata. However, both score files expose only
`video_index`, not per-record video/question hashes, and the two preflight
cohort digests differ. The stored lock contains 10 hashed assignments but no
reproducible builder/source linking each 16-frame score row to those
assignments. Thus the artifact chain verifies that a lock references the 16
score blob, but does not independently reconstruct the claimed 16↔32 row-level
pairing. Treat the −0.200 nat difference as positional/exploratory only, not a
verified paired effect; do not use it for inference or planning thresholds.

The standalone means remain descriptive for their observed cohorts and are not
claim faithfulness. Direct claim-specific evidence is still **0/3 reviewed,
0/3 scored, no effect/CI/test**. No raw row identities were emitted in this
audit. At **19:27:50 JST**, **4h03m03s** elapsed and **1h56m57s** remain to the
deadline. Next: keep the local outputs private, re-run affected tests after
the protocol detail change, and close the artifact/protocol audit before any
commit decision.

The revised protocol digest is `d25544ed…d10f` and matches the candidate
builder's exact-byte pin. The affected candidate/calibration set passes
**42/42**; the complete suite passes **1016/1016** in **9.57 s** with exit 0,
and `git diff --check` passes. At **19:29:31 JST**, **4h04m44s** elapsed and
**1h55m16s** remain. Continue provenance-chain review and prepare a strictly
local commit only after checking the full explicit path allowlist.

## 19:32 JST checkpoint — align the research overview to verified gates

Updated `docs/research_overview.md` to stop calling the converter “future” work
and to reflect its current schema-v2 state, while explicitly retaining the
missing real-browser/PTS/frame-match and integrated-provenance gates. Also
added the audit finding that the auxiliary 16/32-frame paired difference is
not independently reconstructible by item from stored score rows. The two
per-setting means remain reproducible; the paired contrast remains exploratory
and unverified. This is a documentation correction, not a new experiment.

An independent privacy scan of local commit `4f75c41` found no raw QA/claim
text, video IDs, or credentials. It did identify one host-local `/tmp` artifact
locator and digest in the private restart log. Since the commit remains local
and must not be pushed, no history rewrite is warranted; future publication
remains blocked. The commit contains only docs/scripts/tests, not raw data or
result files. At **19:31:36 JST**, **4h06m49s** elapsed and **1h53m11s** remain.

## 19:37 JST checkpoint — distinguish per-item review chronology

The offline review page previously stamped every record with one export-time
`reviewed_at_utc`, which hid whether the three rows were completed separately.
Updated it to record each row's first video `play` event and the time its
required row fields became complete. The reviewer audit and assignment planner
now require `review_started_at_utc ≤ reviewed_at_utc ≤ index.created_at_utc`,
and the export schema is versioned as v2. The UI explicitly calls these
browser self-report timestamps; they do not prove actual viewing or chronology.
This improves audit detail without clearing any human-review gate.

Focused packet/audit/assignment/renderer tests pass **50/50**; the full suite
passes **1018/1018** in **9.79 s**. A rendered synthetic page's inline JavaScript
passes `node --check`; Python compilation and `git diff --check` pass. The
independent reviewer found no blocking schema defect and verified that the
assignment remains unauthenticated and not granted. One naming caveat: the
start field is the first verified-media play event, not the beginning of reading;
the UI describes that limitation. No actual browser, dataset, GPU, or model was
used. At **19:37:55 JST**, **4h13m08s** elapsed and **1h46m52s** remain. Next:
stage only these eight source/test paths plus this log and commit locally.
