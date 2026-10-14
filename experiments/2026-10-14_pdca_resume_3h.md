# PDCA resume: three-hour research window (2026-10-14)

## Window and scope

Requested work window: 3 hours. Start recorded at 2026-10-14 17:23:12 JST;
deadline 2026-10-14 20:23:12 JST. At the 17:25 JST checkpoint, 2m17s had
elapsed. Work must continue until that deadline unless the user cancels or a
real external blocker prevents all useful work. The local tree began on `main`
at `116aefa7d0a531520768801a00c42f86c94d5061`, tracking the same cached
`origin/main` ref. No live fetch was performed. Existing untracked NExT-GQA and
sample-manifest files were preserved and not staged.

## Plan

The preregistered head confirmation is closed after missing its +0.25 nat
practical criterion; NOAA human review and NExT-GQA item-level rights remain
unresolved, so no new direct faithfulness GPU experiment is authorized. The
existing plan also contains one already-cleared fixed-clip GPU implementation
diagnostic, but its same-clip batch-shape branch had already reached a stop
before this resumed window. The initial work therefore focused on independent
review-workflow, protocol, provenance, and CPU validation, keeping the direct
estimand status explicit.

## Environment checkpoint

An individually elevated, read-only per-command check at 17:24 JST observed an
RTX 5070 Ti with 14,816 MiB free of 16,303 MiB, instantaneous GPU utilization
0% and memory busy 8%. The repository `.venv` has PyTorch 2.11.0+cu128, CUDA
12.8, `torch.cuda.is_available() == True`, and BF16 support. Linux reported
15 GiB RAM total, 13 GiB available, and cgroup current use about 2.61 GiB with
no configured cgroup maximum. These are point-in-time readings. No GPU job was
started at this initial checkpoint because direct-evaluation branches were
closed or gated; an artificial load would have no discriminating scientific
purpose.

## Independent review-workflow check

The NOAA private first-pass export and gold-reference page were reviewed at the
code level without opening canonical annotations, IDs, or review packets. The
gold page compared `source_sha256` objects using serialized JSON strings, so
equivalent objects with different key order could fail. The comparison now
checks exactly the five expected hash fields and values independent of object
key order, while rejecting missing or extra keys. First-pass then separate gold
reveal remains the order. This is only an interoperability correction; it does
not prove blinding, human review, rights, or execution readiness.

The primary agent independently reviewed the diff and reran the focused NOAA
packet/contract suite: 16 tests passed. Python byte-compilation and
`git diff --check` passed. A further runtime check extracted the comparator
from a generated synthetic gold-reference page and executed it under Node:
reordered keys were accepted and missing, extra, or mismatched fields were
rejected. The dated detail is in
`experiments/2026-10-14_noaa_first_pass_hash_match.md`.

## Independent protocol audit

The claim-mask schema-v5 validator was audited directly along with its
documentation and tests. It fails closed to exploratory stage, requires the
SESOI unresolved fields to remain null unless a positive, estimand-bound
pre-outcome declaration is supplied, fixes one unadjusted primary contrast and
two Holm-adjusted secondary family names, recomputes the primary estimand hash,
and always reports `execution_authorization=not_granted`. Its structural checks
do not authenticate evidence, reviewer identity, rights, or timing truth.

An independent reviewer identified two future reproducibility improvements:
record review timestamp and outcome-score access status/timing alongside a
fixed SESOI declaration; and bind the individual secondary hypotheses/tests
within each Holm family, rather than only the family names. These are
recommendations for a future schema revision, not current execution blockers
beyond the already unresolved SESOI/human/rights gates. The primary agent
independently ran the claim-mask validator, normal-power, and Student-t power
suites: 40 tests passed.

## Existing direct-result provenance recheck

The published head-confirmation artifact and run manifest were reanalyzed
locally. The analyzer verified record accounting and recomputed the stored
summary: 79/79 records, zero failures, mean selective effect −0.0756 nat,
SD 0.3667, 95% t CI [−0.1578,+0.0065], descriptive p versus zero 0.0706,
and p versus the +0.25 nat threshold 1.52e−11. The threshold success rule is
false. This remains the previously completed confirmation, not a new sample or
estimate. The result JSON SHA-256 is
`993acd16e62c1e8cf8aafe075674cc1e649761f2e1ca5bb05be83814a2e6ac36`; the run
manifest SHA-256 is
`d3c59d49c3609a03d868592d188507f3df5fc1cd1408612bfb98c315d63e1b16`.

The current local held-out sample JSON still does not match the historical
manifest hash in the run manifest (`2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`), so the original analyzer cannot directly join the local file to the old result. A separate tracked video-identity audit already records this exact manifest hash and reports 79/79 unique normalized IDs, 79 successfully hashed files, zero duplicate IDs, zero repeated paths, and zero exact-content duplicates. Its recorded runner hash matches both the local script and the source-commit Git blob. The head-result has all 79 requested ordinal keys, with no missing or duplicated key. Added `scripts/reconcile_head_confirmation_identity.py` to check these relations without reading or emitting private IDs; its privacy-safe output is `results/head_confirmation_identity_reconciliation_20261014_05.json` (SHA-256 `a6ee6dae985678a51b2e6bcc52eceef12be80293ae90b54087e66829992f6067`). It marks `cohort_identity_verified` and explicitly sets `result_run_manifest_crosshash_binding=false` and `independence_established=false`. The original private manifest remains mismatched locally, but the existing hash-linked audit now supports the 79-record uniqueness/content summary; shared scenes, campaigns, transcodes not byte-identical, and dependence remain untested. The private manifest was not printed, changed, staged, or published.

## Potential per-item license metadata lead

A read-only public-source search found the NExT-CF 2026 dataset card, which
describes 570 clips derived from NExT-QA and says its `metadata.json`,
`tier_assignment.csv`, and `ATTRIBUTION.csv` include per-clip original license
and photographer attribution. The card marks the release as an anonymous
submission under review and applies CC-BY-SA 4.0 to metadata/QA and
CC-BY-NC-SA 4.0 to tier-1 edited videos. This is a potentially useful locator
for source-license leads, not an authoritative rights determination: the
individual records and underlying Flickr license pages have not been
independently checked, the release is anonymous, applicability to this
repository's eligible cohort is unknown, and its terms must be evaluated for
the intended local inference, retention, and publication uses.

The official Multimedia Commons page still says the YFCC metadata is the
source of per-media license information and that Webscope is currently not
offering the YFCC100M metadata route. No NExT-CF data files were opened or
downloaded, no private ID intersection was attempted, and no legal conclusion
was made. Links: [NExT-CF dataset card](https://huggingface.co/datasets/nextcf-anon-2026/nextcf),
[NExT-CF v1.0.0 release metadata](https://huggingface.co/datasets/nextcf-anon-2026/nextcf/commit/f5975d95e8a920bd7531db12f41e3e8d7e5656d2),
[official YFCC100M page](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).

The official [NExT-QA README](https://github.com/doc-doc/NExT-QA/blob/main/README.md)
says its videos are sourced from VidOR and points to separately downloaded raw
videos; it asks users to cite/attach the source for use and distribution but
does not provide item-level media-license decisions there. The official
[NExT-GQA README](https://github.com/doc-doc/NExT-GQA/blob/main/README.md)
similarly separates raw-video download from annotation/code, and its root
[MIT license](https://github.com/doc-doc/NExT-GQA/blob/main/LICENSE) only
licenses the software package. This recheck supports keeping the two rights
streams separate; it does not establish whether any particular clip is
permitted for this study.

## Rights-ledger structural audit and publication checkpoint

To make the unresolved rights gate easier to assess without exposing private
crosswalks, added `scripts/audit_item_rights_manifest.py`. It checks the exact
eligible-manifest byte hash, unique eligible IDs, one ledger row per eligible
ID, no outside/duplicate/missing rows, separate annotation and source-media
evidence references, review and conditions declarations, and one explicit
human decision per declared use. Output contains counts, file hashes, and
failure codes only; it always says `execution_authorization=not_granted` and
does not label declaration hashes as authenticated rights. The protocol
contract documents the private draft shape and the command. Synthetic tests
cover complete declarations, stale cohort hash, missing/duplicate/extra rows,
unresolved or prohibited uses, unresolved conditions, and private-ID
non-disclosure. The full scripts test suite passes 489 tests, including the
seven new synthetic/CLI audit tests. Python byte-compilation and `git diff
--check` pass. An independent auditor found
no fail-open path or identifier echo and prompted clearer terminology for
declaration-completeness versus actual rights verification; that clarification
is now applied. No real rights ledger or dataset manifest was passed to the
new checker.

### Follow-up refinement: scope and source-dataset lineage

An additional independent review found that a free-text purpose scope and
dataset lineage should be explicit required fields, rather than implicit in the
allowed use categories and source identifiers. The ledger now requires a
nonempty `intended_use_scope`, annotation `dataset_id`, and per-video
`source_dataset_id`; the contract describes these as human declarations, not
verified license conclusions. Their values remain excluded from the summary.
The seven focused synthetic/CLI tests pass after the refinement, and the full
suite was rerun: 489 tests passed. Byte-compilation and `git diff --check` also
pass. This is structural evidence only; no actual permissions were inferred.

The scoped NOAA workflow change was committed as
`a5033f0d208780c5b452508cdf92635a91e8a1f7`; the rights-ledger structural audit,
tests, contract, and log were committed as
`24ef1b4d116d5f7bf7f1c4dfe8f301cb2376e05c`. Both were pushed in sequence and
then live-fetched. The latest `origin/main` resolves to the same second commit;
the published audit-script and log SHA-256 values match their local files.

### Source-lineage follow-up

Primary dataset documentation confirms a plausible multi-step lineage:
NExT-GQA's dataset README says its training annotations match NExT-QA and that
its `map_vid_vidorID` maps QA video IDs to video paths; NExT-QA states that its
videos come from VidOR. VidOR describes its videos as unprocessed YFCC100M items
and exposes a YFCC video identifier/hash in its annotation format. The
Multimedia Commons page
says YFCC metadata contains the per-media creator-selected license, while its
current access instructions say Webscope is not offering the metadata download
and that dataset access is subject to the Webscope agreement. NExT-GQA's README
separately instructs users to download raw videos. These sources identify a
possible mapping path; they do not verify any private sample's links or license.
No item identifiers were queried externally, no source metadata was downloaded,
and no rights conclusion was made. Sources: [NExT-GQA dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md), [NExT-QA README](https://github.com/doc-doc/NExT-QA/blob/main/README.md), [VidOR dataset documentation](https://xdshang.github.io/docs/vidor.html), [YFCC100M Core Dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/), and [NExT-GQA preparation README](https://github.com/doc-doc/NExT-GQA/blob/main/README.md).

This review showed that a final `source_dataset_id` and `source_video_id` alone
cannot preserve evidence for each intermediate mapping. The private rights-ledger
schema has been advanced to v2: it binds an `eligible_dataset_id` namespace and
requires each video row to contain an ordered chain of mapping transitions,
each with private evidence locator/hash. The checker validates the start at the
eligible ID, continuity between transitions, and the final source-media ID.
This remains a structural declaration check; it neither fetches nor
authenticates mapping evidence. Ten focused tests pass, and an independent audit
found no fail-open path or privacy leak in the chain check. The full 502-test
suite passes; the change was published as `f75397ca91a6218683bb8d6b1b1165aaed478e2e`,
live-fetched, and its validator SHA-256 matched the local file.

### Held-out video-name mapping check

The confirmation manifest stores a `video` field that is not itself keyed the
same way as `map_vid_vidorID.json`. The official NExT-GQA README describes that
file as mapping QA annotation video IDs to video paths, so an exact raw-string
join is the wrong check. Added a privacy-safe CLI that joins only by the
filename stem, rejects duplicate JSON keys and sample stems, validates file-like
inputs and nonempty map values, normalizes path separators and dot segments when
detecting duplicate mapped paths, and emits counts/hashes without IDs or paths.
Against the local 25-row held-out manifest
(SHA-256 `7bf254a944ec1f05017b01e365a0145dac762fb6d08b69ea736a06864b02cb32`)
and local map (SHA-256
`1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685`), the
reproducible audit found 25/25 distinct stems mapped to 25 distinct normalized
path values, with zero unmapped rows. Six synthetic tests pass, including
privacy and malformed-input cases. The independent audit caught duplicate-key,
directory-like input, whitespace-target, and path-normalization issues before
publication; these are now addressed. This verifies only a local
annotation-ID-to-video-path join; it does not verify path availability, the
earlier dataset transfer history, a YFCC source record, a per-video license, or
permission for inference. The result artifact is
`results/nextgqa_heldout_video_mapping_20261014_03.json`; it contains only
aggregate counts, input hashes, and fixed status fields. Unsalted input hashes
can still link identical known inputs, so the artifact is derived/minimal rather
than fully anonymous.
Its SHA-256 is `c517255124e952b35c98910b302d7b0a9d8a1dd3223147af188e20d1968b5038`.

The same audited join was run on the separate 80-row necessity-choice candidate
manifest (SHA-256
`adbda59dbe8886e1f1eddde08a1a28ea976a0499f04b446eb45a6147e9b06dc6`):
80/80 distinct stems mapped to 80 distinct normalized path values, with zero
unmapped rows. Its derived summary is
`results/nextgqa_necessity_candidates_video_mapping_20261014_01.json`. This
cohort-level path mapping does not establish candidate eligibility or rights.
Its SHA-256 is `b65c1a8b104f4dd5254bd2c5f8addc7cf153e70fe85fe65c8a254684979f20dc`.

After the fixes, the independent auditor found no remaining issue in the
implementation, synthetic tests, summaries, or claims, and confirmed that both
summaries contain no IDs or paths. The auditor did not reopen the private input
manifest/map; the primary agent independently ran both commands and reproduced
the 25/25 and 80/80 aggregate counts. The full CPU suite passes 508 tests,
`py_compile` passes, and `git diff --check` is clean. The old local artifacts
`nextgqa_heldout_video_mapping_20261014_01.json` and `_02.json` predate the final
schema and are excluded from publication; only the `_03` held-out result and
the 80-row candidate result are in scope.

### NExT-CF license-index lead review

The public NExT-CF dataset card describes 570 clips derived from NExT-QA and
states that it publishes `metadata.json`, `tier_assignment.csv`, and
`ATTRIBUTION.csv` with per-clip original license and Flickr attribution. Its
v1.0.0 release `CITATION.cff` marks the authors as anonymous and the submission
as under review ([release commit](https://huggingface.co/datasets/nextcf-anon-2026/nextcf/commit/f5975d9),
[CITATION.cff](https://huggingface.co/datasets/nextcf-anon-2026/nextcf/blob/main/CITATION.cff)).
The README limits use to noncommercial research and distinguishes
CC-BY-NC-SA edited tier-1 videos
from CC-BY-SA metadata; tier-2 NoDerivatives entries are not redistributed as
edited videos. The official [NExT-CF dataset card](https://huggingface.co/datasets/nextcf-anon-2026/nextcf/blob/main/README.md)
and release metadata are therefore a potentially useful source-license index,
but they do not verify
that any of this repository's held-out/candidate items are present or that its
assertions are authoritative for our intended uses. No item IDs were queried
externally, no gated records were downloaded, and no license conclusion was
made. The held-out mapping result only maps local QA IDs to local video-path
strings; it does not link the paths to NExT-CF records. This lead cannot close
the rights gate absent an independently reviewable, cohort-matched per-item
record and evidence for the actual source media.

To test the split applicability without sending IDs externally, a second local
audit compares only filename stems from the two private cohorts to the local
NExT-QA `train.csv`, `val.csv`, and `test.csv`. The NExT-GQA README says its
`train.csv` annotations match NExT-QA train and its `val.csv`/`test.csv` are
subsets of the corresponding NExT-QA splits ([official dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)).
All 25 held-out and all 80 necessity-choice candidate IDs match the NExT-QA
test video set; neither cohort matches train or val. The local split sets have
0 pairwise video-ID intersections (3,870 train, 570 val, 1,000 test distinct
video IDs). NExT-CF's Croissant metadata says its source videos were drawn from
NExT-QA validation ([Croissant metadata at commit `1b03938`](https://huggingface.co/datasets/nextcf-anon-2026/nextcf/blob/1b03938/croissant.json)).
Under the declared split IDs, its license index is therefore not a cohort match
for these test samples. This is still not proof against aliases or duplicate
media under different IDs, nor an independent validation of either dataset's
split claims. The final privacy-safe outputs are
`results/nextgqa_heldout_split_membership_20261014_03.json` and
`results/nextgqa_necessity_candidates_split_membership_20261014_03.json`; they
contain aggregate split counts, hashes, and fixed statuses only. Unsalted hashes
retain the same linkability caveat above, and permissions remain not granted.
The independent audit found that Python's CSV `DictReader` could silently
overwrite duplicate `video_id` headers or ignore extra fields. The parser now
rejects duplicate/missing headers and rows with incorrect field counts; JSON
also rejects non-finite constants. Regression tests cover these malformed
inputs. Five focused split-audit tests and the full 513-test scripts suite pass;
`py_compile` and `git diff --check` pass. The final strict CSV-parser rerun uses
the `_03` outputs; `_01` and `_02` are excluded from publication. The
independent auditor reviewed the hardened parser and `_02` summaries; the
primary agent reran with strict CSV parsing and confirmed byte-identical `_03`
summaries, with hashes below. The SHA-256
values are `da5d7094278a5e4dd55123be6bc3b83b174b403f27f9eeadccecb5002e903d47`
(held-out) and
`27cafd48100003908dd1c2521818490078e336a5d6361512bf4ce95d58017d9e`
(necessity-choice candidates).

### Private review-template preparation

To make the remaining human rights review actionable, added
`scripts/create_item_rights_review_template.py`. It accepts a private cohort
manifest (`video_id` records or file-like `video` values normalized to filename
stems), creates a canonical eligible manifest and matching schema-v2 ledger in
one new private directory, and writes only unresolved decisions/review fields.
The bundle directory is mode 0700 and its two files are mode 0600; existing
output directories are refused. An independent audit found and prompted a hash
binding repair: the ledger now hashes the exact canonical eligible-manifest
bytes that are written beside it. Five focused tests pass, including parsing,
permissions, no overwrite, stdout privacy, hash binding, and the requirement
that the generated ledger remains a rights-audit failure. The final independent
audit found no remaining critical issue; unsalted summary hashes remain
linkable and are not treated as anonymous.

Private, local-only bundles were generated for 25 held-out and 80
necessity-choice candidate videos; their IDs and paths were not committed. The
existing rights checker confirms the files bind correctly but both templates
fail closed: complete permitted declarations are 0/25 and 0/80, respectively,
with annotation/source evidence, source lineage, conditions, and use decisions
unresolved. No reviewer attestation was created, and execution authorization
remains `not_granted`. This prepares a human worksheet only; it does not advance
faithfulness evidence or clear data rights.

### Candidate-use and exact-media overlap audit

Before treating the 80-row necessity-choice candidate manifest as a fresh
cohort, a local CPU audit compared opaque `video_id` values and video-path
filename stems against all 14
other `data/nextgqa_samples*.json` manifests. The 80 candidates contain 80
unique video identities and QA pairs; 63 already occur in historical sample
manifests: 60 in the necessity-choice replication heldout set and 3 in its
smoke set. Both historical black and blur outputs contain those 60 held-out
records and 3 smoke records, all with zero failures. The local preflight
screened all 80 under the older 16-frame grounding/control rule: 72 were
marked `ok`, 7 had insufficient controls, and 1 had no evidence. Of the 17
candidates absent from previous sample manifests, 9 were `ok` but not selected
for that old run and 8 were preflight-ineligible under its rule. Thus these 17
are not wholly unseen: all were preflight-screened, and only 9 were not
previously outcome-scored. The 80-row pool cannot be called a fresh independent
cohort relative to historical necessity-choice outcomes. Whether reuse is
appropriate for the distinct claim-mask intervention must be fixed before
scoring; the remaining 9 are not a fresh random sample, and power for a
restricted set remains unassessed. These counts do not redefine the claim-mask
estimand or establish eligibility under its future review protocol.

The 25-row held-out and 80-row candidate media files are all locally present
and readable (requested/effective 25/25 and 80/80; total bytes hashed
102,221,199 and 369,691,580). Their normalized video stems, resolved paths,
and exact SHA-256 file contents are unique within each cohort, with zero exact
file-content overlap between cohorts. This only excludes repeated byte-identical
files; it does not detect transcodes, shared scenes, aliases, or campaign-level
dependence. This output is a local-only audit: it binds unsalted hashes of
private manifests containing paths and annotation text, so that artifact was
not published. The new reusable identity-overlap auditor uses fixed manifest
labels, rejects malformed supplied identity fields, and emits counts only in
the CLI summary; its eight tests cover overlap, duplicates, privacy, and
malformed inputs. Its private JSON contains
unsalted manifest hashes and should remain local.

An independent reviewer confirmed all 80 candidate pairs, the 60 held-out plus
3 smoke overlap, prior outcome joins, and the 72/8 preflight split. The primary
reproduced these aggregate counts from the private preflight and result
artifacts. The exact-media check is verified only for the local files and
stated sets. No GPU inference was run; item-rights and human-review gates
remain unresolved.

As another bounded source-provenance check, the local `NExTVideo.zip` central
directory was inspected without extraction or video reads. It lists 5,600
members: 5,440 MP4 files and 160 directory entries, with 24,583,229,969 total
uncompressed bytes. No filename matched license/rights, attribution/credit,
metadata, manifest, README, or index terms. An independent auditor reproduced
these aggregate counts. This is filename-only evidence: it does not establish
rights, and metadata may exist outside the archive. The archive itself and
member names remain private; no media was copied or published.

The mapping audit was extended to compare each sample's local path relative to
the `NExTVideo` root with its `map_vid_vidorID` target after adding the archive's
`.mp4` suffix. All 25 held-out and 80 candidate rows join exactly to the
expected relative mapped path, and all 105 files exist. The comparison resolves
paths under the declared root, rejects POSIX/Windows absolute and traversal
targets, reports unresolved rows, and emits counts only. CLI output is created
exclusively with mode 0600 because it includes unsalted input hashes. Six
focused regression tests cover path match/mismatch, root containment, Windows
drive and UNC paths, incomplete rows, and output privacy.
These checks verify consistency among the local annotation ID, mapping table,
and local file path; they do not verify the source upload identity, original
license, or authority to infer. The private result summaries include
unsalted input hashes and remain local.

### Post-stop duplicate GPU run (not a new research result)

At 18:36 JST, I mistakenly resumed the fixed NOAA batch-shape diagnostic before
checking the latest resumed-window log, which had already concluded that this
same-clip branch was stopped after two matching runs. The four-frame smoke and
32-frame/16-decode replay therefore duplicated a closed branch; this was an
avoidable workflow error. The smoke completed 38/38 model calls and its
same-context top-1 gate passed. The 32-frame replay completed 38/38 calls
(34 measured, 4 warmup) and failed the same-context top-1 gate. Its artifact
`results/qwen_video_batch_throughput_gpu_smoke_20261014_b1_b2_scalar_32f16_01.json`
has SHA-256
`6c0cf5877db25844bc0a244892601b2d6dfa9cf21c1c83bdd7b3fb9926cfe78e`, is mode
0600, and remains local and uncommitted. The independent artifact audit
verified its source commit/hash, call accounting, processor/cache geometry,
and aggregate-only privacy.

The replay reproduced the previously observed first top-1 disagreement at
generated step 7; steps 1–6 agreed, the step-7 context prefixes were equal,
B1's top-1 margin was 0.125 logit units, B2's was 0.0, and their top-2 sets
matched. Step 8 also disagreed at a tie-adjacent margin; all comparisons used
the forced shared B1 prefix. The same two mismatch steps, margins, and
top-2-set result were already present in both earlier same-clip runs. This
third run adds no independent video or new research conclusion; its differing
timings are not throughput evidence. Do not publish it as a new confirmation
or repeat this branch. The finding remains a narrow numerical near-tie case
for one engineering clip, not an estimate of prevalence or a faithfulness
effect.

At the 18:38:04 JST checkpoint, 1h14m52s had elapsed in the requested window
and 1h45m08s remained. Direct faithfulness progress remains 0 new samples and
0 new effect estimates. The CUDA path was available (RTX 5070 Ti, 14,783 MiB
free before the long smoke, CUDA-enabled `.venv`); hardware availability did
not clear the human-review, rights, or claim-mask SESOI gates.

## Faithfulness progress at this checkpoint

- New direct faithfulness samples this window: 0.
- New direct faithfulness effect estimates this window: 0.
- Existing direct evidence: NExT-GQA / Qwen2.5-VL-7B-Instruct 4-bit,
  top-4 gradient×activation head attenuation at alpha 0.5, frozen-answer
  margin versus matched controls, 79/79 complete; −0.0756 nat, 95% t CI
  [−0.1578,+0.0065]. The preregistered +0.25 nat criterion is not supported.
- Engineering/protocol evidence: one hash-key-order interoperability defect
  corrected and 16 focused tests passed; schema/power audit suites passed 40
  tests. These results are not faithfulness evidence.
- Remaining gates: NOAA human review is incomplete; NExT-GQA per-item video
  rights are unresolved; claim-mask SESOI is unresolved; the historical local
  sample manifest does not reproduce the recorded input hash.
- Current window checkpoint (18:28:40 JST; 1h05m28s elapsed, 1h54m32s remain):
  direct faithfulness is `未実施` (0 new samples / 0 new effect estimates);
  local video-name/path mapping is `補助診断のみ` (105/105 local paths match
  normalized mapped paths and files are present); exact-file identity is
  `補助診断のみ` (105/105 read,
  105 unique byte sequences, zero exact-content overlap between the two cohorts).
  Candidate-use overlap is also an auxiliary provenance finding (63/80 already
  outcome-scored; 17/80 absent from result manifests but preflight-screened,
  of which only 9 were old-rule eligible and unscored). Per-video rights and the
  human/SESOI gates remain `no_go`. The full CPU suite passes 531 tests after
  fixing issues found by independent audits. No GPU run occurred because no
  cleared direct-evaluation branch is available.
- Current window checkpoint (18:33:39 JST; 1h10m27s elapsed, 1h49m33s remain):
  direct faithfulness is still `未実施` (0 new samples / 0 new effect estimates).
  Added an explicit UNC-path regression after independent review; the focused
  mapping/overlap suites pass 20 tests, and the full scripts suite passes 532
  tests. `py_compile` and `git diff --check` also pass. This hardens only a
  local provenance audit and does not open a rights or inference gate.

## Next action

Continue the time-box with useful, bounded, non-duplicative work. Reassess
whether any public-source rights path or protocol reproducibility issue can be
resolved without exposing private identifiers or making unsupported legal
claims. Keep closed head/batch branches closed. Run a GPU experiment only if a
distinct, pre-specified, cleared and smoke-validated experiment becomes
available; do not generate GPU load to fill the window.

## Schema-v6 protocol reproducibility hardening

Reviewing the protocol against the existing design found that schema v5 fixed
the names of the two Holm families but had no hash reference for their exact
member hypotheses/tests. It also represented a future SESOI's review timing as
a label only, with no independent review timestamp or outcome-score-access
record. Schema v6 now requires a SHA-256 reference for the frozen
secondary-hypothesis manifest. A future `preoutcome_fixed` SESOI additionally
requires the independent-review timestamp at or before freeze time, an explicit
`not_accessed_at_freeze` declaration, and a hash of the private access-audit
record. Missing/malformed declarations fail the structural check. The contract,
decision template, and protocol design were updated together.

The focused validator suite passes 32 tests, including old-schema rejection,
manifest hash shape, review timestamp ordering and invalid times, access status,
and preservation of `execution_authorization=not_granted`. The combined claim-mask
validator/power suites pass 41 tests, and the full scripts suite passes 490 tests.
Byte-compilation and `git diff --check` pass. An independent auditor directly
inspected the schema, docs, and malformed-value paths and found no defect; it
confirmed bad JSON-shaped values fail closed and execution authorization stays
ungranted. These are structural checks only: hashes are not resolved, timeline
declarations are not authenticated, and neither schema validity nor these tests
authorize scoring. Direct faithfulness progress at this checkpoint is still 0
new samples / 0 new effect estimates; the gates above remain unresolved and no
GPU job ran. The implementation and tests are published in commit
`61cc6aad3340673161c4bd35a412ebf9464200ce`, live-fetched and verified.

## Historical head-result identity reconciliation

The local sample JSON hash mismatch looked like an identity-provenance gap, so
the published result, run manifest, and existing privacy-safe video audit were
reconciled independently. The new tool requires result ordinals to cover the
entire requested prefix, the run-manifest sample hash to equal the audit's
input hash, a complete per-video identity/content audit with no ID/path/exact
content duplicates, and the historical runner source hash to match the recorded
Git source commit. It prints only aggregate counts, hashes, and status. An
independent audit caught that the legacy run manifest lacks a result-file hash
and source commit, so a broad `verified` status would overstate result-to-run
linkage. The output now says `cohort_identity_verified` and separately reports
`result_run_manifest_crosshash_binding=false`; detailed boolean fields default
false on malformed inputs. The audit also caught bool/int count equivalence and
misleading detail flags on malformed records; strict integer checks and
regression tests were added. The historical artifacts pass the cohort identity
reconciliation: 79/79 results and hashed files, 79 unique IDs, zero duplicate
IDs or exact-content files. Nine focused tests and the full 499-test suite pass.
A CLI output artifact was generated without private inputs and contains no
ID/path/question/answer fields. This supports aggregate video uniqueness for
this historical cohort, but does not establish statistical independence,
result-to-manifest crosshash linkage, or identify visually shared
scenes/campaigns. It does not change the closed head result or its effect
estimate.

## Candidate-selection provenance repair

An independent review of the untracked 80-row candidate cohort found that its
original selection command, input hashes, candidate pool, and per-video QA-row
choice cannot be recovered from tracked records. The tracked
`scripts/build_samples.py` is a seeded random QA-row sampler that stops after
finding `n` readable unique videos; it is not an all-eligible-video enumerator.
Thus the old 80-row cohort must not be described as reproducibly selected.

For future runs, the script now accepts optional `--audit-out` and records
aggregate exclusion counts, seed/RNG and Python/OpenCV build versions, hashes of
inputs and output, repository HEAD plus a boolean check that the current script
matches that HEAD. The sidecar explicitly says the local video tree and media
bytes are not hash-bound and decoder decisions are not reproduced. The sample
manifest and sidecar are prepared as mode-0600 sibling temporary files, then
published with exclusive hard links; a second-target collision rolls back the
first link, and sidecar-construction failure happens before publication. The
two names are not jointly atomic to concurrent readers, so a successful writer
can briefly expose the sample before the sidecar. This prevents a failed writer
from leaving an orphan sample that looks provenance-complete. The sidecar
remains private because its unsalted hashes can link inputs. It does not repair
historical selection provenance or authorize inference.

Verification: 5 focused provenance tests pass, including matching the sidecar
hash to the exact staged manifest, owner-only permissions, existing-output
preservation, rollback after a collision, and rejection of `n <= 0`. A
synthetic one-video CLI run selected 1/1 fake readable MP4, emitted both files
at mode 0600, and its
sidecar status/hash checks passed. The complete scripts suite passes 537 tests;
expected failure-shaped JSON lines are emitted by negative-path tests. Python
byte-compilation and `git diff --check` pass. No real sample selection, direct
faithfulness scoring, or GPU task ran. The sidecar hashes the values of a
whitelist of OpenCV backend/FFmpeg environment settings without storing the
values, following an independent design review. The independent implementation
audit found no correctness blocker; it noted that the pair is not jointly
atomic to concurrent readers and suggested additional failure-injection tests.
An additional regression round then injected audit-construction and arbitrary
second-link failures and verified temporary-file cleanup and rollback. The
focused suite passes 7 tests and the full scripts suite passes 539. These tests
do not establish joint atomic visibility or crash durability.

Commit `526d6a5c1246fcd3d5a5f7c8e751532692e7bdf5` was pushed to GitHub `main`
and verified with a live `git ls-remote`; only the script, focused tests, and
dated log were included. Existing untracked dataset and sample files remain
untouched. Follow-up commit `67a0bc764d1e490e94032d2eaa867f1166e1aa42` adds two
failure-injection tests and updates this log; it too was pushed, live-fetched,
and its source/test SHA-256 values matched the local files. A post-commit
synthetic CLI smoke additionally verified `source_matches_commit=true` with
the recorded commit, 1/1 selected synthetic video, 0600 outputs, and no
temporary files left over. Full scripts suite: 539 passed.

At 18:49:42 JST, 1h26m30s had elapsed and 1h33m30s remained. Direct faithfulness
progress remains 0 new samples and 0 new effect estimates.

## Archive-path privacy hardening

A second audit of the sample-builder CLI found that archive paths could appear
in its missing/invalid ZIP diagnostics and indexing progress. The first fix
changed intentional messages to basenames and added tests for missing,
requested-invalid, home-fallback-invalid, and indexing errors. An independent
review then noticed that `is_zipfile` and extraction exceptions also needed
path-safe handling, so the first 10-test validation was not treated as final.

The independent review then traced the exceptional `is_zipfile`, ZIP-index, and
member-extraction paths, finding that Python exception text could include the
archive path. Those expected archive failures are now caught and reduced to the
archive basename plus exception class. Regression tests inject path-bearing
errors into all three branches, in addition to missing/invalid requested and
fallback archive cases; all assert that the temporary absolute path is absent
from stdout/stderr. The focused suite passes 12 tests and the complete scripts
suite passes 544. The independent reviewer found no remaining leak in the
explicit or handled archive paths. This does not promise redaction from
unrelated interpreter or operating-system diagnostics. Direct faithfulness
progress remains 0 new samples / 0 effects; no real video was read in these
privacy tests and no GPU run occurred.

At 18:54:08 JST, 1h30m56s had elapsed and 1h29m04s remained. Commit
`63eb2c510612a007364dbc2d48b9f3fff2ede472` was pushed and live-fetched; the
remote script and focused-test SHA-256 values match local bytes. The unrelated
untracked dataset/sample files remain untouched. The logged time is the
checkpoint immediately before that commit/push, not time spent by those later
commands.
