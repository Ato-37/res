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

The current local held-out sample JSON does not match the historical sample
manifest hash in that run manifest (`2fa141ee9a9c00e86db380b2ee2dbb3161080224c6b0eaf86ddc401cd1341e10`). Therefore the analyzer rejects the current local file and
cannot reproduce the result-to-video identity/uniqueness join from this tree.
Without that optional private manifest, the published result/manifest summary
checks pass, but unique video identities are correctly reported as
`unverified`. This is a bounded historical-provenance limitation; it does not
change the recomputed effect summary or authorize reopening the closed branch.
The private manifest was not printed, changed, staged, or published.

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

The scoped NOAA workflow change was committed as
`a5033f0d208780c5b452508cdf92635a91e8a1f7`; the rights-ledger structural audit,
tests, contract, and log were committed as
`24ef1b4d116d5f7bf7f1c4dfe8f301cb2376e05c`. Both were pushed in sequence and
then live-fetched. The latest `origin/main` resolves to the same second commit;
the published audit-script and log SHA-256 values match their local files.

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
