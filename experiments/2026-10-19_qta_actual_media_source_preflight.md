# QTA actual-media source hash preflight — 2026-10-19

## Purpose

Add a CPU-only, local preflight joining the fixed three-row real-video sampler
receipt to the actual-QA source by unique source-video SHA-256. This checks
which three source-video byte identities are shared. It does not score
faithfulness or grant authorization.

## Setup

New script: `scripts/qta_actual_media_source_preflight.py`. It reuses
`load_candidate_inventory` from `scripts/run_qta_real_video_sampler_preflight.py`,
`bind_qta_actual_qa_source` from `scripts/qta_actual_qa_source_contract.py`, and
`_build_claim_and_selection` / `_target_manifest` from
`scripts/run_qta_actual_qa_source_binding_smoke.py`.

The source bytes are checked against exact candidate-inventory roles:
`source_test_csv`, `claim_input_manifest`, `claim_mapping_worksheet`,
`diagnostic_target_candidate`, `claim_lineage`, `qta_protocol_draft`, and
`qta_treatment_selection`. Binder-ready claim/selection and target bytes are
derived from those frozen inputs through the existing adapters. Protocol and
config digest pins are read from the matching inventory roles; callers cannot
provide alternate expected pins.

CSV, protocol, and config are read only from the fixed repository paths used by
the existing source-binding smoke CLI: `NExT-GQA/datasets/nextgqa/test.csv`,
`docs/qwen_question_attention_faithfulness_protocol_v1_draft.md`, and
`experiments/2026-10-18_qta_treatment_selection.md`. Descriptor-based reads
reject symlinks along the path and check file owner, mode, type, size, and
stable-read metadata. Private review inputs remain basenames under
`data/private_nextgqa`. Candidate inventory accepts only
`inventory-NAME/pre_review_candidate_byte_inventory.json`, with a 0700
directory and 0600 file.

The join uses sampler `source_sha256` and lineage `video_sha256`, linked through
the binder crosswalk. It never uses array order or sampler `record_index` as
identity. Duplicate/missing lineage video IDs fail closed. Any failure resets
all accepted row counts to zero. Output contains aggregate counts and explicit
false authorization/scoring fields only; no raw hashes, review tokens,
questions, answers, claims, or video identifiers are emitted.

## Verified results

The focused synthetic suite passed 11/11 tests. It covers order-independent
joining, duplicate sampler hashes, exact inventory/media and source-role
binding, stale code closure, duplicate lineage IDs, atomic zeroing, and denied
authorization fields. Filesystem tests cover fixed repository paths, rejected
alternate paths and symlinks, mutation during read, constrained nested
inventory paths, and symlink rejection in code closure. `py_compile`, CLI
`--help`, and whitespace checks passed. Each of the seven inventory-bound
source roles has an individual changed-byte rejection test, and a missing
source role also fails closed. The pre-review inventory builder's focused
suite passed 21/21 after registering nine previously unmapped local QTA
phase-plan modules and the preflight itself. A direct static transitive-import
scan then found no remaining unregistered local script dependencies. A final
integration regression run covering the source contract, dataset adapter,
binding CLI, sampler, this preflight, and inventory builder passed **68/68**
tests.

A first candidate build had 82 roles and one missing slot,
`python_source_import_closure`; it was kept as a failed local candidate. After
registering the transitive modules, a later candidate had 91 roles and zero
missing slots. That candidate was superseded when the preflight implementation
itself was added to the registered code closure. The final candidate inventory
had 92 roles and zero missing slots. It remains `candidate_only`, with review
incomplete, no joint freeze, and execution authorization `not_granted`. A fresh
CPU sampler run on the exact inventory decoded 3/3 videos and sampled 32/32
frames each in 10.2 seconds. Decoded frame counts were 510, 1,140, and 1,258;
maximum grid deviations were 16.145, 16.145, and 16.355 ms. No model or GPU
was used.

The final actual local source join then returned
`source_video_hash_joined_unvalidated`, with requested/sampler/actual-QA/joined
counts all 3/3 and no failures. An
independent check recomputed all seven source-role bindings, the
inventory-to-receipt digest link, inventory media set, lineage video-hash set,
uniqueness, the preflight code hash, and private file modes: **10/10 checks
passed**. The CLI revealed only aggregate counts/status and fixed denial
fields. Sampler or source provenance is not authenticated by these local bytes.

## Interpretation and limitations

The real-data join establishes only that the three source-video SHA-256 values
in the current local sampler receipt match the three distinct video hashes in
the inventory-bound actual-QA lineage/crosswalk, independent of row order. It
does not verify that a human found any claim supported, authenticate the
candidate inventory or receipt, establish browser-clock/decoder-frame
correspondence, or permit a baseline/treatment forward. No faithfulness score,
effect estimate, confidence interval, or test was produced. Rights status
remains unverified and every artifact stays local.

Direct faithfulness remains **0/3 claims reviewed, 0/15 clock/frame
correspondences, 0/3 prediction-link reviews, and 0/3 direct scores**. The
informal nickname-to-question check still has no response JSON. Therefore no
direct `ΔF` or uncertainty estimate exists. The head confirmation remains
closed at n=79, −0.0756 nat, 95% t CI [−0.1578,+0.0065], below its preregistered
+0.25-nat practical threshold. The selected v2 QTA proxy's n=3 descriptive
`ΔA` is not faithfulness evidence and remains closed.

## Next action

The simple nickname-first impression artifact is now available at
`data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v4.html`. It asks
only three broad questions and is not a formal claim review. No response JSON is
present in the private repository tree; if one is completed, inspect it locally
and keep it separate from formal review counts. The source-identity join above
is complete, but does not unlock scoring. Direct scoring still requires the
formal Stage-1 claim review/custody, all 15 clock/frame correspondences, a
frozen baseline `p/q`, and separate Stage-2 prediction-link review. Continue
with CPU-only implementation or verification of the existing claim-mapped
scoring caller and its fail-closed gates; do not launch a real-data model
forward until those prerequisites pass. Do not repeat the completed proxy GPU
cohorts.

The current local join can be reproduced with the following command from the
repository root; it reads protected files locally and emits only aggregate
status/counts:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/qta_actual_media_source_preflight.py \
  --candidate-inventory inventory-qta-join-20261019_0635/pre_review_candidate_byte_inventory.json \
  --sampler-receipt qta_real_video_sampler_preflight_20261019_0635.json \
  --claim-input-manifest claim_input_manifest_v6_20261017_0635.json \
  --worksheet claim_mapping_sentence_v6_20261017_0635.json \
  --target-candidate diagnostic_target_v6_current_20261017_01.json \
  --lineage-manifest claim_lineage_sentence_exploratory_20261016_01.json
```

## Earlier freshness rerun (2026-10-19, 06:43 JST)

After adding the synthetic formal-workspace browser harness, the primary agent
reran the same command against the current worktree. It still returned
`source_video_hash_joined_unvalidated`, with sampler/actual-QA/joined counts
3/3/3, no failure, zero score calls, and all authorization/authentication
fields false. Thus the new browser test file did not stale the inventory's
registered code closure. This confirms only the current local byte-identity
join; it does not change the interpretation or open any later gate. No GPU or
model was used.

## Current code-closure refresh (2026-10-19, 10:46–10:48 JST)

The previous 92-role candidate inventory had become stale after changes to the
registered code. A fresh candidate was built at
`data/private_nextgqa/inventory-qta-freshness-20261019_1105/` with **93 roles**
and **zero missing required slots**. The added code role is the local module
`qta_prebaseline_live_batch_binding.py`; the static transitive-import closure
now has no unmapped registered script imports. Comparing the old candidate to
the new one, all **23 non-code source/review/media roles** retained identical
sizes and SHA-256 values. This is a local freshness check, not authentication
of the inventory or any reviewer.

A new sampler preflight was run against this exact candidate. It passed for
**3/3 videos**, decoding **510, 1,140, and 1,258 frames** and sampling **32
frames per video**. Effective decode rates were **1.825, 0.816, and 0.739
frames/s**; maximum grid deviations were **16.145, 16.145, and 16.355 ms**.
It used CPU only; no model was loaded and no faithfulness score was computed.
The refreshed actual-QA source preflight then joined **3/3** sampler videos to
the fixed actual-QA rows by source-video hash, with no failure code. Its status
is `source_video_hash_joined_unvalidated`; source/sampler authentication,
semantic review, human identity, model load, scoring, and execution
authorization all remain false. The reported later gates remain Stage-1
custody, human claim review, clock/frame correspondence and calibration,
processor-batch binding, prescore mapping, baseline `p/q` freeze, and Stage-2
prediction-link semantic review.

Independent read-only review recomputed **70/70** registered code-role sizes
and digests and checked the new inventory/sampler binding and non-authorizing
fields. The independent reviewer did not reopen the private actual-QA inputs;
the current 3/3 source-video join is therefore supported by the primary's
sanitized CLI output, not by a second independent execution of that join.

On the current checkout, the focused source-preflight, inventory-builder,
sampler, and pre-baseline diagnostic suites passed **45/45 tests**. This
revalidates the fail-closed structural checks, not the authenticity or semantic
validity of real review evidence. Direct faithfulness remains **0/3 claims
reviewed, 0/15 clock/frame correspondences, 0/3 prediction-link reviews, and
0/3 direct scores**; there is no direct `ΔF`, interval, or statistical test.
No GPU forward was justified by a source-byte freshness check. Rights status
remains unverified and all candidate artifacts stay local.

Next, inspect whether the claim-mapped baseline caller can consume the present
inputs without bypassing the missing human and clock/frame evidence. If it can,
finish its fail-closed static and synthetic validation; if not, document the
exact interface/gate gap and prepare the shortest local handoff for the
nickname-first preliminary review. Do not rerun closed proxy cohorts or count
the informal artifact as formal evidence.

## Baseline-path and preliminary-response checkpoint (2026-10-19, 10:48 JST)

Source inspection confirmed that `nextgqa_baseline_only_scoring.py` exposes a
fixed three-row full-video scoring function and has synthetic unit coverage, but
the current `qta_actual_candidate_input_bridge.py` and
`qta_direct_faithfulness_preload_gate.py` are explicitly non-authorizing; there
is no production real-data wrapper that can safely turn candidate inputs into
an authorized baseline run today. The current formal review workspace also
requires externally pinned trust and custody inputs and always reports model
loading/scoring denied. Consequently, a GPU baseline call would bypass the
current protocol and was not run.

The focused bridge, preload-gate, baseline-candidate, baseline-scoring, and
formal-workspace test command completed **64/64 test executions**. These are
synthetic structural/invariant tests (some imported test classes are discovered
more than once by the combined unittest command); they do not create human or
real-video faithfulness evidence. A private-tree response scan found **0**
`exploratory_video_review_*.json` files, so the nickname-first preliminary
artifact has not yet produced an inspectable observation. Direct faithfulness
remains **0/3 claim reviews, 0/15 clock/frame pairs, 0/3 prediction-link
reviews, and 0/3 direct scores**, with no direct estimate or uncertainty
interval. The next evidence-bearing step remains a locally saved preliminary
response or the independent formal-review inputs; no proxy result can replace
either.

## Persisted-result and transaction audit follow-up (2026-10-19, 10:52 JST)

To retain an inspectable result without exposing row-level data, the sanitized
source-preflight JSON was saved exclusively at
`data/private_nextgqa/qta_actual_media_source_preflight_20261019_1052.json`.
Its mode is **0600**, size **846 bytes**, and SHA-256 is
`7eeda971b722d54a55e1a11cc437fa8b5364e6ef8ec19659b54cc21d6789b2ea`. An
independent read-only audit parsed this aggregate and compared its fields to
the source-preflight code: it confirmed the **3/3/3/3** joined counts, no
failure code, and every denial field. It did not reopen source QA/review/video
inputs, so the joined count still has one primary execution and is not an
independent recomputation of the raw join.

Separate design and phase-gate audits confirmed the next implementation gap:
the baseline scorer primitive exists and validates three full-video batches,
but it accepts an already loaded model and caller-provided pins. The existing
phase-1 signed receipt is deliberately non-authorizing; the bridge and
clock/frame audit cannot authenticate reviewer identity, semantic review,
15/15 correspondence, or out-of-band trust-root origin. There is no
same-process pre-load transaction that validates those exact inputs, keeps the
same `FrozenBatch` objects alive, and only then permits exactly three
baseline forwards. A speculative coordinator cannot make the current local
candidate review evidence trustworthy. The next dependent action is therefore
to receive the actual independent review/trust/correspondence artifacts; once
they exist, implement and test the one-shot baseline transaction against the
frozen phase-1 protocol, before any model loader can run.

## Pre-baseline transaction design re-audit (2026-10-19, 10:55 JST)

An independent API audit verified that the baseline score primitive itself is
already present; the missing interface is the same-process authorization and
loader boundary. The scorer accepts an already-loaded model plus caller-provided
row/run pins, so calling it directly cannot guarantee zero model loads when a
real upstream gate fails. The live-batch helper checks three exact in-memory
`FrozenBatch` objects, but its signed phase-1 receipt verifier explicitly
returns `execution_authorization=not_granted` and `authorized_forwards=0`.
The 15-anchor clock/frame audit and Stage-1 audit also retain explicit
unverified identity/semantics/calibration fields. Thus there is currently no
valid route to load the real model for the planned three baseline forwards.

The predeclared protocol gives the required sequence: validate all source,
review, clock/frame, prompt/token, processor/runtime, and exact batch evidence
before loading weights; permit only 3 baseline forwards; freeze `p/q`; stop and
send the predictions for separate Stage-2 review. The current receipt code is
only a signed byte-binding candidate. A production transaction must compose
and rerun the validators, preserve the exact batch object identities into the
scorer, and verify a separately trusted phase-1 authorization; no caller
success boolean or synthetic signature can stand in for those inputs. This
audit refines the implementation boundary but adds no human or faithfulness
evidence and does not justify a GPU run.

## Informal response inventory correction (2026-10-19)

A focused local response inventory later found two syntactically valid
single-review exports with the same nickname, same ordered item IDs, and same
categorical judgments. They are duplicate exports of one informal respondent,
not independent reviews. The downloaded copy was imported and revalidated
against the v6 artifact. Its item judgments match the prior export; the saved
schema has no explanation-sufficiency field. A separate chat follow-up raises
question clarity for item 1 and explanation sufficiency for item 3. Those
comments remain qualitative and are not substituted for structured labels.

This corrects earlier lines stating no valid quick-review response was present.
Formal evidence remains unchanged at **0/3** claim reviews, **0/15**
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. The baseline transaction remains blocked on trusted formal
review, calibration/correspondence, and phase-1 authorization inputs.

## Fresh candidate and actual-media sampler recheck (2026-10-19 14:58 JST)

The formal pre-review byte inventory was rebuilt into a new local-only
candidate directory accepted by the sampler/source-preflight path. It contains
**94 unique roles, 0 missing slots**, and remains `candidate_only`, with review,
freeze, scoring, and execution-authorization flags false. Inventory SHA-256 is
`25aeb8a66324d57290e3b0c9fb6e1d786f1fc66d4d8aa5dac0c61dde9efe2673` (file
0600; directory 0700). An independent audit recomputed size and SHA-256 for
five decisive runtime roles, including the formal renderer and combined-review
envelope; all five matched. No review HTML or media content was opened by that
auditor.

The frozen CPU sampler was rerun against this exact candidate on the three
local source videos: **3/3 requested/effective**, **32 sampled frames per
video**. Decoded-frame counts were 510, 1,140, and 1,258; effective rates were
1.825287, 0.815690, and 0.739134 fps; maximum grid deviations were 16.145,
16.145, and 16.355 ms. The owner-only receipt is
`data/private_nextgqa/qta_real_video_sampler_preflight_20261019_1458_current.json`
(0600; 30,925 bytes; SHA-256
`ee9e00811f0c62b79db29c429e1af451c6d307ced5e07d59b778b64f3c418247`). The
separate CPU source-byte preflight then joined the sampler's three source
hashes to **3/3 actual-QA rows** with no failure code. An independent auditor
repeated the source preflight once and confirmed the same sanitized aggregate
and denial fields. Its local report is
`data/private_nextgqa/qta_actual_media_source_preflight_20261019_1458_current.json`
(0600; 846 bytes). The candidate, sampler receipt, and report remain local-only.

This closes a source-byte/sampler freshness check only. Sampler/source
provenance, human identity, semantic review, clock/frame correspondence,
scoring, and execution remain unauthenticated or denied. No model was loaded
and no GPU forward was run. Direct-faithfulness status remains **0/3** formal
claim reviews, **0/15** verified clock/frame correspondences, **0/3**
prediction-link reviews, and **0/3** direct scores; there is no direct `ΔF`,
interval, or test. The next direct GPU task remains exactly three untreated
baseline forwards, conditional on trusted formal review/release and all 15
clock/frame checks. Do not rerun closed proxy cohorts.

