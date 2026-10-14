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
unresolved. No new real-data GPU experiment is currently authorized by its
scientific/data gates. The selected work is therefore an independent audit of
review workflow interoperability, protocol structure, result provenance, and
CPU validation while keeping the direct-estimand status explicit.

## Environment checkpoint

An individually elevated, read-only per-command check at 17:24 JST observed an
RTX 5070 Ti with 14,816 MiB free of 16,303 MiB, instantaneous GPU utilization
0% and memory busy 8%. The repository `.venv` has PyTorch 2.11.0+cu128, CUDA
12.8, `torch.cuda.is_available() == True`, and BF16 support. Linux reported
15 GiB RAM total, 13 GiB available, and cgroup current use about 2.61 GiB with
no configured cgroup maximum. These are point-in-time readings. No GPU job was
started because available GPU branches are closed or gated, and an artificial
load would have no discriminating scientific purpose.

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
found no fail-open path or privacy leak in the chain check; full-suite rerun and
publication remain pending at this checkpoint.

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
