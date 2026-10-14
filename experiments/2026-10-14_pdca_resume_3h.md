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
published with exclusive hard links; the sidecar is linked before the sample.
A second-target collision rolls back the first link, and sidecar-construction
failure happens before publication. The two names are not jointly atomic: a
concurrent reader can briefly see the sidecar before the sample, while ordinary
publication does not expose a sample before its sidecar. Abrupt process/system
termination between the links can leave an orphan sidecar; crash durability is
not guaranteed. The sidecar remains private
because its unsalted hashes can link inputs. It does not repair
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

## Synthetic archive-path end-to-end check

To verify the successful archive branch, a new synthetic ZIP CLI test creates a
dummy `folder/clip.mp4` member, maps one fake QA row to it, and substitutes only
the readability decision so no real media is decoded. The script indexes the
archive, extracts/moves the member under the temporary video root, emits a
private one-row manifest plus sidecar, hashes the exact ZIP and manifest, and
records a source/HEAD match. The test confirms extracted bytes equal the ZIP
member, sidecar/CLI output omit temporary paths and identifiers, and both files
are mode 0600. Independent review suggested asserting extracted bytes and final
manifest digest; both assertions were added. No claim about real-video decoding
is made. The focused provenance suite passes 13 tests and the complete scripts
suite passes 545. This closes archive-path plumbing coverage with synthetic
content only; it does not clear data rights or justify real candidate selection.
Independent review confirmed the test scope and that the readability stub does
not overstate media validation.

At 18:56:12 JST, 1h33m00s had elapsed and 1h27m00s remained. Direct faithfulness
progress remains 0 new samples / 0 new effect estimates.

## Sidecar-first pair publication

Reviewing the pair writer's concurrent-reader window led to one further
adjustment: it now links the provenance sidecar first and the sample manifest
second. If a reader observes the sample during normal execution, its sidecar is
already present. The test asserts this ordering and forces a pre-existing sample
collision after sidecar publication to check rollback and preservation. A
separate injected second-link error exercises cleanup. The pair is still not
jointly atomic or crash-durable: readers may briefly see a sidecar without a
sample, and abrupt termination may leave that sidecar behind. One initial test
run incorrectly expected `source_matches_commit=true` while the script was
modified; correcting the assertion to check the field's boolean type made the
test independent of dirty/clean worktree state. The focused suite passes 13 and
the complete scripts suite 545. An independent audit verified the publication
order, rollback on collision/second-link failure, and the stated concurrent
reader and crash limitations. Commit
`c5521a93dc6b862afbf33f05e38a9b3bd1983ce9` was pushed and live-fetched; remote
script/test SHA-256 values match local bytes. Untracked data/sample files were
preserved. Faithfulness progress remains 0 samples / 0 effect estimates.

At 18:59:03 JST, 1h35m51s had elapsed and 1h24m09s remained.

## Selection parity with provenance sidecar enabled

A synthetic regression test now invokes the sample builder twice on the same
four-row fixture, with fixed seed 29 and n=2, once with and once without
`--audit-out`. The complete selected manifests match and contain two unique
IDs. Readability is stubbed and media bytes are synthetic, so this establishes
that enabling the audit output does not alter seeded selection on this fixture;
it does not establish real-media readability, cohort representativeness, or
rights. Independent review confirmed the assertion scope. The provenance-focused
suite passes 14 tests and the full scripts suite passes 546 tests. Direct
faithfulness progress remains 0 new samples / 0 new effect estimates.

At 19:02:14 JST, 1h39m02s had elapsed and 1h20m58s remained. After this
checkpoint, the focused provenance suite passed 14/14, the full scripts suite
passed 546/546, `py_compile` passed for the builder and focused tests, and
`git diff --check` passed. Negative-path tests printed their expected error
and diagnostic JSON fixtures. Publication is still pending at this checkpoint.

## Public-source rights-path check

On 2026-10-14, the official NExT-GQA repository description and dataset README
were checked without opening local annotations or sample IDs. They describe
NExT-GQA QA annotations as a subset of NExT-QA and describe a mapping from QA
video IDs to video paths. The official NExT-QA repository README says raw video
is sourced from VidOR and asks dataset users to cite/attribute the project; its
`LICENSE` file is an MIT license for software/documentation. NExT-GQA also has
an MIT `LICENSE` whose grant is expressly for software/documentation. The
official
NExT-QA dataset page links to separate video, mapping, and annotation downloads,
but the page itself does not state item-level media licenses or an annotation
data-use license. The official
VidOR project page describes its 10,000 videos as drawn from YFCC100M. The
Multimedia Commons YFCC page currently says its Webscope route is unavailable,
that dataset access is subject to a Webscope agreement, and that original media
licenses are uploader-selected and represented per item in metadata. These
pages do not establish the applicable terms for any particular local video or
the annotation files; no item ID was matched and no legal conclusion is made.
The rights gate therefore remains unresolved. Needed evidence is still a
source-bound item-level media license/conditions review plus an annotation
dataset terms review, for the exact frozen eligible cohort.

Sources inspected: <https://github.com/doc-doc/NExT-GQA>;
<https://github.com/doc-doc/NExT-GQA/blob/main/LICENSE>;
<https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md>;
<https://github.com/doc-doc/NExT-QA/blob/main/README.md>;
<https://github.com/doc-doc/NExT-QA/blob/main/LICENSE>;
<https://doc-doc.github.io/docs/nextqa.html>;
<https://xdshang.github.io/docs/vidor.html>;
<https://multimediacommons.wordpress.com/yfcc100m-core-dataset/>.
This is a research-log note, not a rights determination. Direct faithfulness
progress remains 0 new samples / 0 effects.

At 19:03:21 JST, 1h40m09s had elapsed and 1h19m03s remained. This log update
has not yet been committed or published.

## Claim-selective dependence interval contract

The CPU arithmetic helper previously returned only the mean video-level
contrast although the frozen design requires an all-eligible-video 95% t
interval. It now reports sample SD, standard error, degrees-of-freedom-based
two-sided Student-t 95% CI, and explicitly returns no inferential interval for
n=1. The contrast is evaluated as `control_score - selected_score`, algebraically
canceling the shared full score before floating-point arithmetic; independent
review found that the former equivalent expression lost a -0.5 contrast under
a synthetic 1e16 common offset. Nonfinite derived effects and dispersion/CI
overflow fail closed. The independent reviewer re-ran the latest 10 focused
tests and found no material issue; added tests cover exact interval values,
n=1, constant effects, large offsets, and extreme values. The full scripts
suite passes 551 tests; py_compile and `git diff --check` pass.

This is an estimator/preparation contract only. It does not compute a p-value,
fix the unresolved SESOI, establish the independence or representativeness of a
real cohort, or authorize claim extraction, rights-sensitive scoring, human
review, GPU inference, or efficacy claims. A t interval relies on independent
video-level contrasts and its usual sampling assumptions. Direct faithfulness
progress remains 0 new samples / 0 new effect estimates.

At 19:06:08 JST, 1h42m56s had elapsed and 1h17m04s remained. The code and this
log entry are not yet committed/published.

## Freeze the claim-mask interval method in the protocol schema

The new estimator exposed a protocol gap: the SESOI rule said “95% CI lower
bound” but did not identify the interval method. Before any claim-mask outcomes
exist, schema v7 now requires
`primary_ci_method=two_sided_student_t_df_n_minus_1`, matching the tested CPU
estimator. The contract and decision template identify the 95% interval,
video-level independence assumption, and `df=n-1`; v6 inputs are rejected, and
a new test rejects an alternate method. The validator still requires an
unresolved SESOI and always reports `execution_authorization=not_granted`.
Focused validator tests pass 33/33; py_compile and `git diff --check` pass.
This freezes an analysis method, not a threshold; no outcome data or GPU were
used. Direct faithfulness progress remains 0 new samples / 0 effects.

At 19:08:12 JST, 1h45m00s had elapsed and 1h15m00s remained. Independent
review and the complete regression suite were pending. The independent auditor
then verified the v7 fields/docs, old-schema rejection, retained exploratory
and no-authorization gates, and ran 33/33 focused tests successfully. The full
scripts suite passes 552/552; expected failure-shaped diagnostics came from
negative-path fixtures. The validator, tests, template, contract, and this log
entry are not yet committed/published.

## Publication scope audit

After publication, `HEAD` and `origin/main` both resolve to `689dd9d`. The five
commits since `52c9905` contain only the intended provenance regression test,
claim estimator/tests, claim protocol validator/tests/docs, and this dated log.
The log was searched for local absolute paths, private cohort filenames, and
synthetic IDs; no matches appeared. User-local untracked files remain
unmodified and unstaged. This is a tracked-path/content hygiene check, not a
claim that every public-source legal statement was independently adjudicated.

At 19:09:13 JST, 1h46m01s had elapsed and 1h13m59s remained. This publication
scope audit note is not yet committed.

## Machine-readable CI provenance

To make the estimator output directly comparable with schema v7, its summary
now emits the same `primary_ci_method` field/value, `interval_confidence_level`
0.95, numeric `interval_degrees_of_freedom` (`n-1`, or null at n=1), and the
video-independence assumption. An independent audit confirms those fields and
the 10/10 focused estimator tests. The final full scripts suite passes 552/552;
expected diagnostic output is from negative-path fixtures. Byte-compilation and
`git diff --check` pass. This is output provenance only; it adds no result,
fixed SESOI, or authorization. Direct faithfulness remains 0 new samples / 0
new estimates. The small update is not yet published.

At 19:11:33 JST, 1h48m21s had elapsed and 1h11m39s remained.

After that checkpoint, the estimator field-name alignment and test were pushed
in `2002e1a`, then the official NExT-GQA LICENSE scope citation was added in
`77820cd`. Both were live-fetched; `HEAD` and `origin/main` agree at
`77820cdc93a1777f0ccc4d622c0dfd182750669e`. The estimator still passes 10/10
focused checks, and the complete suite passed 552/552 after the final code
change. No direct evaluation samples or effects have been added.

At 19:12:30 JST, 1h49m18s had elapsed and 1h10m42s remained.

## Remaining gates and next authorized research step

The latest GPU-cache log and the prior one-hour resume log were cross-checked:
the same-clip B=2 near-tie scalar replay already ran and the engineering branch
was explicitly stopped. Do not repeat it or the closed head search. No GPU job
is justified in this window: hardware is available, but every remaining model
evaluation path is gated or scientifically stopped. Direct faithfulness
progress remains 0 new samples / 0 new effect estimates.

Next, unresolved prerequisites are (1) the actual human review and private
attestation for NOAA's 3 videos/6 questions if that separate branch is resumed;
(2) item-specific NExT-GQA annotation/media rights evidence for an exact frozen
cohort; (3) an independently reviewed, answer-relevant claim-to-frame mapping
for every eligible NExT-GQA video; and (4) an estimand-bound, independently
reviewed SESOI with outcome-access record fixed before scoring. After the
NExT-GQA inputs are complete, run the existing privacy-safe validators; only a
passing protocol, rights, claim-review, and SESOI audit can unlock a small smoke
followed by the preregistered all-eligible-video primary GPU run. Keep all local
IDs, raw media, annotation text, and rights evidence private. This is the gate
sequence, not a claim that those inputs exist or are complete.

At 19:13:23 JST, 1h50m11s had elapsed and 1h09m49s remained. The gate update is
not yet committed.

## Control-frame tie-break regression

An independent audit compared the existing `nearest_control_indices` function
with the claim-mask control contract. It enumerates every eligible equal-size
control subset, minimizes total absolute temporal-rank distance, and uses the
ordered rank tuple as the tie-break key. A regression test now exercises a
two-sided equal-distance tie and confirms the lexicographically lower rank is
selected. The focused utility suite passes 11/11; the full scripts suite passes
553/553. No implementation change or real-video read occurred. This confirms a
deterministic synthetic control-selection property; it does not establish
screen-content matching, human claim validity, rights, or faithfulness effect.
Direct faithfulness progress remains 0 new samples / 0 effects.

At 19:14:57 JST, 1h51m45s had elapsed and 1h08m15s remained. Publication is
pending.

## Fail closed on numerically invalid conditional power

An independent numerical stress check of the existing planning-only
`analyze_claim_mask_power_sensitivity.py` found that finite inputs
`n=10000, excess_over_sesoi_sd=1e100` cause SciPy's noncentral-t survival
function to return NaN. The helper previously returned that value as if it
were power, and `build_grid` could then serialize it only by failing later at
JSON output. The helper now raises `ValueError` whenever computed power is
non-finite or outside [0,1]; regression coverage checks both the direct helper
and grid path. This is a numerical safety repair; it selects no SESOI and uses
no empirical outcomes. Focused tests pass 6/6, `py_compile` and
`git diff --check` pass, and the complete scripts suite passes 554/554.

Independent audit reproduced SciPy's NaN at the same extreme input and
confirmed both fail-closed paths; it found no material issue. As a separate
formula check, a seeded simulation of 300,000 independent normal video-level
paired differences (`n=40`, true mean 0.3 paired SD above the hypothetical
SESOI, seed 20261014) estimated success probability 0.45634 (binomial SE
0.00091), versus noncentral-t calculation 0.45645; difference was -0.13
Monte Carlo SE. This checks the conditional normal-model arithmetic only; it
does not validate normality for real videos, choose an SESOI, or establish
faithfulness. Direct faithfulness progress remains 0 new samples / 0 effects.

At 19:16:55 JST, 1h53m43s had elapsed and 1h06m17s remained.

A second boundary check found that an arbitrarily large but positive integer
sample size (`n=10**100`) made SciPy's t quantile raise `TypeError` before the
power-result guard. The numerical calculation block now normalizes arithmetic,
type-conversion, and value errors into the same explicit `ValueError`; the
regression input is included in the invalid-input test. Independent review
reproduced the `stdtrit` type failure and confirmed normalization. The focused
suite remains 6/6 and the full scripts suite passes 554/554; syntax and diff
checks pass. This protects the planning CLI from an internal library exception
on infeasible inputs; no valid-sample design or scientific conclusion changes.
At 19:18:13 JST, 1h55m01s had elapsed and 1h04m59s remained. The addition and
log await scoped publication.

## Current execution-resource check

At 19:19 JST, an individually elevated read-only check saw the RTX 5070 Ti
with 14,624/16,303 MiB free, instantaneous GPU utilization 0% and memory-busy
14%; `.venv` reports torch `2.11.0+cu128`, CUDA build 12.8, CUDA available,
and BF16 supported. Linux reports 15 GiB RAM, 13 GiB available, 4 GiB swap
almost free, and cgroup `memory.max=max`. These are point-in-time readings, not
sustained bandwidth or utilization measurements. No model job ran: the GPU is
available, but the only efficacy candidate remains blocked by human review,
item-level rights, claim mapping review, and unresolved pre-outcome SESOI; the
closed head and batch-diagnostic branches are not reopened. This separates a
research-protocol no-go from a hardware/runtime blocker. Faithfulness progress
remains 0 new samples / 0 effects.

At 19:19:17 JST, 1h56m05s had elapsed and 1h03m55s remained. Resource snapshot
and gate assessment are recorded; no raw or private data were accessed.

The numerical guard is also covered at the CLI boundary: an invalid extreme
grid must raise before the output path is opened, leaving no empty/partial JSON
artifact. The temporary-directory regression passes; independent audit
confirmed `main()` calculates the grid before opening with exclusive-create
mode and reran focused tests 7/7. The full scripts suite passes 555/555 after
this test addition. No result file was published from the invalid input. At
19:20:37 JST, 1h57m25s had elapsed and 1h02m35s remained. This protects
planning artifact integrity; direct faithfulness remains 0 new samples / 0
effects.

## Scoped publication audit

The full change range from the starting checkpoint `52c9905` through live
`origin/main` `1ff0569` contains 12 tracked files limited to protocol/template
docs, this dated log, analysis/validator source, and tests. No result artifact,
media, model weights, question/answer records, or private sample manifest was
added. A diff scan for the local cohort filenames/paths and synthetic IDs
found only the two intended public NExT-GQA source links. `git diff --check`
passes. Current user-local untracked dataset/metadata entries remain untracked;
they were not opened, staged, or modified. This is a scoped path/content audit,
not a claim that all upstream rights are cleared. At 19:21:21 JST, 1h58m09s
had elapsed and 1h01m51s remained.

## Valid-range output reproducibility after numerical guard

Regenerated the existing hypothetical 16-cell normal noncentral-t grid and
80%-target minimum-n sensitivity with the original command, writing only to a
unique `/tmp` path. `cmp` confirms byte-for-byte equality with
`results/claim_mask_power_sensitivity_20261014_03.json`; both SHA-256 values
are `b431d6e465a684259fe3c1370a81f24225d69baabd3ed8b968b50933b06174e0`.
Thus the extreme-input fail-closed repair did not alter this valid-range
artifact. These are unchanged hypothetical design values, not new empirical
evidence. At 19:22:24 JST, 1h59m12s had elapsed and 1h00m48s remained; direct
faithfulness progress remains 0 new samples / 0 effects.

## Exploratory licensed-dataset alternative (not a protocol change)

Because the NExT-GQA item-rights path remains unresolved, I checked the
official Perception Test repository and paper as a possible future fallback.
The repository describes 11,609 real-world videos, 38,060 three-option
multiple-choice questions over 10,361 videos, 6,086 grounded questions over
3,063 videos, and 73,503 temporally bounded action segments. Its README labels
software Apache-2.0 and all other materials CC-BY 4.0; the paper describes
filming by roughly 100 participants, and the supplement states participants
gave informed consent. See the [official repository](https://github.com/google-deepmind/perception_test),
[paper](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf),
and [supplement](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Supplemental-Datasets_and_Benchmarks.pdf).

An independent design review confirmed it is only a candidate: the current
claim-mask protocol is a fixed five-choice answer margin, while Perception
Test MC-QA has three options; its grounded-QA answer is a list of object-track
IDs. Action intervals are not automatically answer-relevant evidence for each
QA, and object tracks are typically annotated around once per second. Thus
the source may make rights provenance clearer, but replacing NExT-GQA changes
the estimand/score distribution and still requires pre-outcome human QA-to-
claim-to-time mapping, split/license-scope review, and a new SESOI/power
assessment. This is a literature/schema check only: no dataset was downloaded,
opened, counted locally, or used for inference; the primary NExT-GQA plan is
unchanged. It does not add faithfulness evidence. At 19:26:34 JST, 2h03m22s
had elapsed and 56m38s remained; direct faithfulness progress is 0 new samples
/ 0 effects.

## QIVD alternative: source-contract fit audit

I checked the current official QIVD description and its linked 2025-03-24
Research Use Data License Agreement, without downloading the dataset. The page
describes 2,900 audio-video items with a human question/answer and a timestamp
marking when an answer is appropriate; this is structurally closer to temporal
evidence review than Perception Test action segments. The license defines
Research Use as internal processing and internal use of Results solely for
non-profit research; prohibits disclosure of the dataset, combining it with
third-party-owned data, and commercial use; and treats the dataset as
confidential, with access language limited to qualifying employees. It also
requires deletion of data and Results when the agreement terminates, and
imposes, where applicable, de-identification/security and retention
requirements. Section 15 explicitly contemplates publication of Results or a
publication based on them, with citation suggested but not required. The
defined term Results is limited to AI-produced algorithms or models, so
whether aggregate scores/analyses fall within it is unclear from the text.
Sources:
[official QIVD page](https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd)
and [linked research-use agreement](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf).

Assessment (inference, not a legal opinion): the timestamp could support
claim-to-time review, but it is not necessarily a claim-specific evidence
interval. The dataset itself cannot be redistributed; the agreement's
publication language contemplates publication based on defined Results, but
the status of aggregate scores/analyses and their retention after termination
is not clear from the text. Written clarification from the institution and
QTI would be needed before relying on public derived summaries or long-term
reproducibility. Therefore QIVD is not selected as a replacement cohort and
does not change the NExT-GQA primary plan or unlock inference. No
video/question/answer content was accessed; this was a terms/schema review
only. Direct faithfulness progress remains 0 new samples / 0 effects. At
19:27:57 JST, 2h04m45s elapsed and 55m15s remained. Independent audit found
the original publication-fit statement too strong; the wording above now
reflects §15 and the ambiguity around whether aggregate analyses are Results.

## Claim-mask estimand/control wording audit and schema v8

Following the validator-hardening log's next action, I compared the proposed
estimand with `scripts/claim_selective_dependence_utils.py`, the v7 validator,
and the frozen protocol wording. The contrast direction is internally
consistent: `d=(S_full-S_selected)-(S_full-S_control)` equals
`S_control-S_selected`, and the helper calculates that reduced form directly.
The full-condition margin is still needed for the pre-mask greedy-correctness
secondary grouping, but does not numerically enter the primary contrast. The
contract now states this explicitly to prevent confusing a two-drop contrast
with an independent contribution from the shared full score.

I found one reproducibility omission: the intervention contract said all
conditions share fill/preprocessing, but neither the required fields nor the
primary-estimand hash had an explicit reference to the actual masked-frame
transform. The current draft therefore did not force a future protocol to bind
black versus blur, blur parameters, color space, operation order around resize/
normalization, or whole-frame replacement. Schema v8 adds required
`intervention.mask_transform_sha256` and includes it in recomputation of the
primary-estimand hash. The contract states what the private versioned transform
specification must define and that selected/control masks use the same
transform; it deliberately picks no fill. Schema v7 inputs are rejected.
This is a structural provenance repair, not a human semantic review, execution
authorization, inference, or faithfulness result. The validator and contract
still report `execution_authorization: not_granted`.

Focused verification: claim-mask validator tests pass 34/34, including missing
transform-hash rejection and a check that changing that hash invalidates the
primary-estimand digest; Python compilation and `git diff --check` pass. An
independent code audit confirmed required-field validation, digest inclusion,
schema-v7 rejection, and all 34 focused tests. It caught two stale schema-v7
references in the SESOI worksheet; these are now updated to v8, and the
worksheet calls out the exact transform details to bind. At 19:30:22 JST,
2h07m10s elapsed and 52m50s remained. Faithfulness progress: no additional
samples or effect estimates; engineering/protocol diagnostics only.

The independent follow-up confirmed the worksheet has no current-facing v7
references; it suggested matching the contract's private/versioned and same
transform-for-both-masks wording, which is now included. Full repository script
suite passed 556/556; the emitted CLI/model failure summaries are deliberate
negative-path test fixtures, not suite failures. `git diff --check` passes.
Only the protocol contract, SESOI worksheet, this experiment log, and the
validator/test are modified. User-local untracked data remain uninspected and
unstaged. At 19:30:48 JST, 2h07m36s elapsed and 52m24s remained. Direct
faithfulness evidence remains 0 new samples / 0 effects; the historical 79/79
head confirmation remains closed with −0.0756 nat (95% t CI
[−0.1578,+0.0065]), below +0.25 nat.

Scoped publication: commit `d10e4bfe7adbd6588946e8b7352e40d260cb4553`
(`Bind claim mask transform in protocol schema`) was pushed to `origin/main`.
An immediate live `git fetch origin main` verified `HEAD == origin/main ==
d10e4bfe7adbd6588946e8b7352e40d260cb4553`. The published commit contains
only the five reviewed contract/template/log/validator/test files. The listed
untracked local dataset/sample files remain untouched. At 19:31:15 JST,
2h08m03s elapsed and 51m57s remained.

## Empty evidence-mask eligibility edge case

The next gate audit found that a reviewed interval can overlap zero sampled
frames. With `k=0`, the old wording allowed an empty selected mask, for which
the purported evidence intervention would be a no-op. Schema v9 therefore
adds `minimum_selected_frame_count=1` to the required intervention declaration
and canonical estimand hash. The contract requires `k>=1`; zero-overlap items
must be recorded as pre-scoring exclusions, before any outcome score is
observed. This does not make the validator inspect item manifests or enforce
runtime selection; it binds the declared rule while the private eligibility
manifest and selector implementation remain separately reviewable.

The first focused run exposed a real implementation defect: the canonical
hash helper treated every payload leaf as text, so the new integer made the
digest unavailable. I changed that helper to accept only nonempty strings and
nonnegative exact integers in the canonical payload, keeping booleans and
other types invalid. The corrected focused validator suite passes 35/35; Python
compilation and `git diff --check` pass. Independent audit is pending. At
19:32:46 JST, 2h09m34s elapsed and 50m26s remained. This is protocol
engineering only: 0 new faithfulness samples / 0 new effect estimates.

Independent gate audit confirmed the validator requires the exact integer 1
(rejecting 0, 2, bool, and null), the field enters the primary-estimand hash,
and schema text does not claim the validator measures actual runtime masks. A
follow-up audit approved the explicit boundary: future scoring code must
compute per-item overlap/cardinality, enforce the minimum before scoring, and
record zero-overlap exclusions. Full repository suite passes 557/557,
including the zero-mask regression; `git diff --check` and Python compilation
pass. The first schema-v9 focused run had five failures because the hash helper
rejected integer payload leaves; that exact cause was repaired and both
subsequent focused runs pass 35/35. This is a fixed implementation defect, not
a flaky test. At 19:33:41 JST, 2h10m29s elapsed and 49m31s remained.

Scoped publication: commit `e4008ef67ae28b7d546f041fe043ec82cb4fb5f8`
(`Reject empty claim evidence masks`) was pushed, then verified by live fetch
with `HEAD == origin/main == e4008ef67ae28b7d546f041fe043ec82cb4fb5f8`.
It updates only the same five protocol/log/source/test files. Private local
untracked dataset/sample artifacts remain untouched. At 19:34:14 JST,
2h10m58s elapsed and 49m02s remained.

## Legacy 5-choice runner compatibility audit

Static source review compared the proposed v9 contract with
`necessity_choice_margin_test.py` and its utility/tests. The implementation
scores distinct A–E single-token IDs at the prompt's next-token position,
computes gold log-probability minus the condition-specific best foil, and sets
baseline correctness by unmasked argmax among the five choice-label logits
versus gold. Its inclusive timestamp membership and equal-count minimum
temporal-rank control/tie rule agree with the proposed geometric selection. It
checks empty evidence and insufficient controls before scoring, but the
independent audit found these become generic runtime failures rather than
pre-scoring exclusion rows reconciled into the frozen cohort denominator.
The contract now specifies the restricted-choice correctness definition and
the first-label tie rule.

Other critical gaps are population/inference and claim mapping: the legacy
runner labels baseline-correct samples as primary, uses bootstrap intervals
and unadjusted sign-flip tests, has no question-type secondary family, and
uses existing grounding intervals without binding an independently reviewed
answer-relevant claim mapping. The user-selected v9 estimand requires all
eligible videos primary, a Student-t primary interval, and Holm correction for
the predeclared baseline-correctness and question-type secondary families. An
independent source/test audit confirmed these differences and the distinction
between pre-score failure and cohort exclusion; 14/14 focused utility/failure
tests pass with `PYTHONPATH=scripts`. The existing script is explicitly
marked in the contract as legacy and not v9-compatible; historical artifacts
must not be upgraded by relabeling. No runner was changed and no GPU/data was
used. At 19:35:30 JST, 2h12m18s elapsed and 47m42s remained; direct
faithfulness progress is unchanged at 0 new samples / 0 new effects.

## Fixed A–E chat-generation boundary gate

The compatibility audit exposed a tokenization assumption that the legacy
runner checked only in isolation: each label had to be one standalone token,
but no check compared that token with the first assistant-target token after
the actual chat-generation prefix. The v9 contract now requires all five A–E
labels to be distinct single tokens, the generation prefix to remain stable
when each label is completed, and the contextual first target token to equal
the standalone label token. The requirement binds to pinned tokenizer, chat
template, prompt builder, and the audit result; it does not claim the CPU audit
validates scores or any real QA item.

Extended `audit_qwen_claim_tokenization_cpu.py` with a prospective fixed
synthetic five-choice prompt. Its gate requires all three conditions and the
CLI writes a diagnostic report before returning nonzero on gate failure. The
report contains counts and token-sequence hashes, not prompt text or token IDs.
Regression tests cover the valid boundary and reject duplicate/multi-token
label mappings; focused tests pass 7/7. An independent review reran those tests
and verified the failure-gate ordering and privacy properties. The full
repository suite passes 559/559 after the gate-exit-code addition, along with
`git diff --check` and Python compilation.

Pinned local processor CPU audit: 12/12 prior synthetic boundary cases
completed without failures; A–E each map to one distinct token, the 63-token
generation prefix is stable for all five, and contextual target-start matches
are 5/5 (`gate_pass=true`). Artifact:
`results/qwen_claim_tokenization_cpu_20261014T104500Z.json`, SHA-256
`02d0eef5ace528fae41404c29d3a49608830bcf0d1a873cdb4456154d45a546c`.
Runner SHA-256 is
`f6bf4999b6f79a76b0e99bc77c7bd0d7877dfa98f584da3c6e644831c0592c43`; pinned
tokenizer revision is `cc594898137f460bfe9f0759e9844b3ce807cfb5`, with
`tokenizers 0.23.2` and `transformers 5.17.0`. This is tokenizer/template
evidence on one fixed synthetic prompt only; it cannot establish model
behavior, correctness of historical per-item scoring, or faithfulness efficacy.
No model weights, data, GPU, or user-local samples were read. Direct
faithfulness progress remains 0 new samples / 0 new effect estimates, with
the historical head result unchanged. At 19:42:52 JST, 2h19m40s elapsed and
40m20s remained.

### Multimodal video-processor boundary extension

The text-only chat-template token sequence (60 tokens in a separate probe)
does not equal the fully processed multimodal model input because the video
processor expands frame placeholders into patch tokens. Therefore template
stability alone did not verify the token sequence actually scored by the
legacy runner. I passed three generated 32×32 solid-color JPEG frames and the
same fixed synthetic choice prompt through production `build_inputs()` on CPU,
then processed each completed assistant A–E prompt through the same processor.
The expanded base prefix was 353 tokens; all five completed inputs preserved
it and began their assistant suffix with the same standalone label token
(5/5). No model forward pass or GPU was used. This adds expanded video-input
stability to the required future runner gate.

An intermediate candidate report for this extension was superseded and deleted
after review found stale `no_video` metadata; it was not published. The
3-frame CPU diagnostic at this checkpoint records both text-template and
expanded video checks in
`results/qwen_claim_tokenization_cpu_20261014T104900Z.json`, SHA-256
`6d29b855ec966fbdfc0e4c313816f406cbbc3d71cb71113b0dec7b569dd6dafe`.
Runner SHA-256:
`8b9cda5da06146aae32fa3ca79b6ed793accb769a1fb11eb37b61d38ef8d3d78`.
It binds the production video prompt-builder source with SHA-256
`ef400606a7e6b7100870990a600df71694c7e019292136e23a25e8513d1e28b3` and
the installed `process_vision_info` implementation with SHA-256
`f3710ced8ffe735da57d08cf5236cf9d326ff646ed1bdad2e62593a7801adc33`, without
publishing local paths.
All prompt/frame content is fixed synthetic fixture material; the artifact
stores counts, booleans, and hashes only. The processor emitted its warning
that its default video path does not apply qwen-vl-utils' per-frame pixel cap;
this diagnostic intentionally follows the production runner's current default
and its tiny synthetic frames make no runtime estimate for real videos. Run
time/VRAM, multimodal model behavior, historical score correctness, and
faithfulness efficacy remain untested. Independent review confirmed the final
scope and focused 7/7 tests pass. The independent reviewer recomputed the
production builder file hash and installed video-preprocessor source hash; both
match the final artifact. Full suite passes 559/559 after the final code change,
with `git diff --check` and Python compilation clean. At 19:49:39 JST,
2h26m27s elapsed and 33m33s remained; direct faithfulness progress remains
0 samples / 0 effects.

Publication: commit `965efa8adde939aec2f241e6bfca14d5d1671154` contains only
the contract, diagnostic implementation/test, this log, and the final synthetic
CPU audit artifact. Push succeeded; a live `git fetch origin main` verified
`HEAD == origin/main == 965efa8adde939aec2f241e6bfca14d5d1671154` and the
remote artifact SHA-256 equals
`6d29b855ec966fbdfc0e4c313816f406cbbc3d71cb71113b0dec7b569dd6dafe`.
Untracked private NExT-GQA data/sample files remain untouched and unstaged. At
19:50:11 JST, 2h26m59s elapsed and 33m01s remained. No new faithfulness sample
or effect estimate was obtained.

## Independent v9 no-go and legacy CLI bypass audit

An independent source audit confirmed that the schema-v9 validator is
declaration-only, rejects a confirmatory stage while SESOI is unresolved, and
always returns `execution_authorization=not_granted`. It does not authenticate
the referenced review, rights, mapping, or outcome-access evidence. The legacy
`necessity_choice_margin_test.py` CLI remains technically invokable without
calling this validator. This is a bypass-shaped implementation path, not a
v9-compliant or authorized scoring route; the contract now marks the script
legacy and forbids using it unchanged for v9. No invocation was made. Do not
run it on real data until the rights, human review, mapping, cohort, SESOI, and
runner gates are closed. Its historical results remain legacy exploratory.

I rechecked current public primary pages. The NExT-QA repository asks users
to cite its paper and attach the source for use/distribution, while its MIT
file is written as a license for software and associated documentation; the
NExT-GQA README calls its QA files identical to/subsets of NExT-QA and
describes the video-path map. YFCC's source page says original-media licenses
are creator-selected and per item in metadata, but the exact local cohort is
not linked to source metadata. These facts do not establish exact-cohort
annotation terms or media permissions. No IDs, private maps, video, or local
rights evidence were inspected. Sources: [NExT-QA README](https://github.com/doc-doc/NExT-QA/blob/main/README.md),
[NExT-QA LICENSE](https://github.com/doc-doc/NExT-QA/blob/main/LICENSE),
[NExT-GQA data README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md),
and [YFCC100M Core Dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).
The independent audit verified validator no-go behavior and the direct CLI
path; it did not read local data. Faithfulness remains at 0 new samples / 0
new effects. At 19:52:02 JST, 2h28m50s elapsed and 31m10s remained.

## Default-scale synthetic video token-boundary confirmation

To test whether the previous 3-frame/32-pixel check missed an input-length
interaction, I changed only the fixed synthetic video fixture to the legacy
runner defaults: 16 frames at 224×224, with per-frame pixel capping omitted as
in the current production builder. On the pinned local processor, the fully
expanded base prompt contains 1,217 input tokens; all five completed A–E
inputs preserve the exact expanded prefix and their suffix begins with the
standalone label token (5/5, `gate_pass=true`). The CPU-only boundary core took
0.208 seconds excluding imports, processor loading, and artifact writing; this
is not a real-video runtime estimate. The processor again warned that the
uncapped path may use more tokens than qwen-vl-utils' capped behavior. No model
weights, forward pass, GPU, real/private video, or QA data was used.

The updated diagnostic is
`results/qwen_claim_tokenization_cpu_20261014T105200Z.json`, SHA-256
`556ff4e6b26bdbc0de128de1a2f5c7e9a02d8cde51c1dec2799a91aad5b0441a`;
runner SHA-256 is
`aa073b1fc56ea7adf04e2ffb5024e70aa52984bfb825a518f231dc7e067c0dca` and
fixture SHA-256 is
`427fd55a59d86bb4e2f34e67b4a29315765e7773adbb50c690de0f864e0747fa`. The
independent reviewer confirmed the fixture matches the runner defaults,
reproduced the 1,217-token input and 5/5 boundary result, and verified the
source hashes and output privacy. Focused tests pass 7/7; the full scripts
suite passes 559/559, `py_compile` and `git diff --check` pass. Direct
faithfulness remains 0 new samples / 0 effects. At 19:53:13 JST, 2h30m01s
elapsed and 29m59s remained.

The default-scale diagnostic, runner/config fixture, tests, and this log update
were published in `fda632018bd1ada0fad3524609c4b4599e07f990`. Push and live
fetch succeeded; `HEAD == origin/main`, and the remote result artifact hash
matches the logged SHA-256. The earlier 3-frame artifact remains a separate
published diagnostic; the 16-frame default-scale result is the current
stronger boundary check, not a faithfulness result. At 19:54:18 JST, 2h31m06s
elapsed and 28m54s remained. No model/GPU/data scoring occurred.

## Synthetic mask-induced score-position stability check

Because the runner selects next-token logits at the end of the processor input,
I tested whether black masking changes the input sequence or only the pixels.
Using the same fixed 16×224 synthetic video, I masked four selected frames and
four separate control frames with the production `apply_mask_fill(...,
"black")` helper. For both conditions, `input_ids` and `video_grid_thw`
exactly match full, pixel tensor shapes match, and pixel tensor content differs
from full. Both selected/control checks pass. This addresses a possible
position-shift confound for the tested preprocessing configuration only; it
does not establish equivalence for every media shape or fill transform and
does not measure model scores or faithfulness.

The fixed mask kind and exact synthetic frame indices enter the fixture hash;
the artifact additionally hashes the production mask-fill implementation.
Updated result:
`results/qwen_claim_tokenization_cpu_20261014T105700Z.json`, SHA-256
`1bcd6b8ac6d328fb7e4f9a2301f27634fc8a2c60cce0011b8f61d195d24e0fa2`.
Runner SHA-256:
`d7c5740b8fda1914005252051d63e706323f61b77953768d410ff4651b2ed525`;
fixture SHA-256:
`8ae8a04aa1a3ec18fbfbb0c8c6e50fbc86c5f49b5a218451d8fdc54e5dc08313`;
mask-fill source SHA-256:
`38e3dcdc1e756e9a5f70ce3daa047a3a2a948e0db8f54e8dc33f8dedd8629961`.
The CPU boundary core took 0.322 seconds, excluding imports, processor load,
and artifact write. Focused tests pass 7/7; the full suite passes 559/559,
`py_compile` and `git diff --check` pass. An independent reviewer recomputed
the artifact, fixture, runner, video-builder, and mask-fill hashes; all match.
The reviewer independently reran the synthetic audit with the same counts and
gates, confirmed the output contains no prompt, option text, or local paths,
and found no check-logic issue. Direct faithfulness progress remains 0 samples
/ 0 effects. At 19:57:04 JST, 2h33m52s elapsed and 26m08s remained.

## Diagnostic scope documentation correction and rerun

Reviewing the committed runner against its execution showed that its module
docstring still described an older three-frame fixture, while the current
diagnostic uses 16 synthetic 224×224 frames and black-masks four selected and
four control frames. I corrected that scope description and reran the audit;
all reported counts and gates reproduce (base 1,217 tokens; chat A–E 5/5;
video A–E 5/5; both mask input-stability gates pass). The regenerated artifact
is `results/qwen_claim_tokenization_cpu_20261014T110000Z.json`, SHA-256
`c0c53faae4b6f6bf1c1c3de913314baeb7c11d07f83f26ca66bd5e5eac049eb6`; the
corrected runner SHA-256 is
`83583ac94cbf051edc30f7e1738cbdc1a9f14263510c7809b62ddedc750bceff`.
An independent reviewer verified the runner, fixture, production builder,
mask-helper, and artifact hashes, reproduced the reported checks, and found no
private prompt or path leakage. Focused tests pass 7/7 and the full scripts
suite passes 559/559. The first test command used an invalid unittest import
path; rerunning with `PYTHONPATH=scripts` passed. This is a documentation and
reproducibility repair only; no model forward, GPU, or real data was used.
Direct faithfulness progress remains 0 new samples / 0 new effect estimates.
At 19:58:10 JST, 2h34m58s elapsed and 25m02s remained.

## Claim-mask validator CLI authorization-boundary smoke

To ensure a structurally clean protocol cannot be mistaken for permission to
score, I generated the test suite's synthetic schema-v9 fixture in a temporary
directory and ran the validator CLI on it. The command exited 0 for structural
validity, while its JSON explicitly returned
`execution_authorization=not_granted` and
`semantic_validation=not_machine_verifiable`; the CLI does not authenticate
the fake review/rights hashes or grant inference. This distinction is
intentional per the validator contract. The focused validator suite passes
35/35, including tests that rights status and a caller-supplied true runtime
flag cannot authorize execution. A source search found no v9 scoring runner
integrated with this validator; the legacy runner remains explicitly outside
the v9 path. No private records, data, videos, or model were accessed. Direct
faithfulness progress remains 0 new samples / 0 effects. At 20:00:10 JST,
2h36m58s elapsed and 22m02s remained.
