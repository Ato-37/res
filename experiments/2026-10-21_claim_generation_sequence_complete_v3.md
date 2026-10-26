# Claim-generation v3: retain necessary event sequence (2026-10-21)

## Motivation

The legacy three-item exploratory review is now verified against its exact
local v14 question/claim artifact and the corresponding claim-mapping packet.
Across its three self-reported respondents, item 2 was judged not to fit the
question by 2/3; item 3 was judged to fit by 3/3 but its video evidence was
judged insufficient by 2/3. The user's own notes similarly identified an
incorrect response and a correct but incomplete event description. These are
development observations only: they do not label the current cohort or
establish claim-level faithfulness.

The previous `v2_short` instruction requires at most 25 words and one main
action. This may suppress a second visible step or resulting state needed to
answer a question. Version `v3_sequence_complete` is a frozen prompt variant:
one concise sentence of at most 40 words; include observable steps relevant to
all parts of the question and the resulting state when relevant; allow a short
sequence; do not invent intent, causes, or unseen details. The old `v1` and
`v2_short` prompt bytes remain unchanged. This is a prompt-design hypothesis,
not a claim that v3 is more accurate.

## Planned development check

Apply v3 once to all three exact rows in the already frozen cohort11 test lock,
without replacing rows. This cohort is outcome/exposure-known and was already
used for v2 claim generation; the comparison is an exploratory paired prompt
development check, not an independent replication. Record all three outputs,
format-valid attrition, output lengths, and binding to the same source-video and
question digests. Do not score claim truth or faithfulness from fluency,
structure, QA answers, or the existing v2 results. If the outputs are
structured, build a new fingerprinted local review artifact rather than
overwriting either existing handoff. Review is still needed to judge semantic
fit and evidence sufficiency.

Before GPU execution: run the focused generator tests and verify the frozen
cohort11 pair filter, lock, exclusion manifest, dataset binding, and new unique
output path. Expected work is one pinned 4-bit BF16 Qwen2.5-VL-7B model load,
16 frames × 3 videos, three short deterministic generations, approximately
20–30 seconds model/setup time and 8 GiB peak allocated VRAM based on the prior
cohort11 generation. Keep the packet local and mode 0600; rights remain
unverified.

## Execution and results

An initial v3 generation was run locally on 2026-10-21 using the then-current
replay helper: 3 requested/effective videos, 16 frames each, 3/3 structured
claims, 21.22 seconds, RTX 5070 Ti, 7.78 GiB peak allocated VRAM, zero reported
generation failures. Its private output is
`data/private_nextgqa/claim_generation_cohort11_gpu_v3_20261021_01.json`.
Independent audit subsequently found the replay helper did not fully bind the
packet's question/choices/gold/spans to the live dataset and did not enforce
secure private-manifest reads. **Do not treat this initial output as a verified
handoff or evidence; do not reuse it for human review.** It is preserved locally
for provenance, and the claim-review count remains 0.

The replay path has since been repaired to require exact schema and hash sets,
re-read the current split CSV/gsub/video map, reselect the locked rows, re-decode
and verify frame PTS/pixels, compare row content and pair order, and recompute
the frozen cohort digest. Private manifests are read from a directory file
descriptor with `O_NOFOLLOW`; the generator also requires replay split equality
with its CLI split. The original cohort11 packet now passes a CPU rebind:
3/3 rows, 3 unique pairs, recomputed cohort SHA matches the lock. Focused prompt
and replay tests pass 18/18, including a synthetic valid 3-row fixture, edited
question/row-swap rejection, missing dataset-hash rejection, split mismatch,
symlink/mode rejection, and a live local packet rebind. An independent audit of
these repaired files was pending at this checkpoint; it subsequently approved
the replay path for exploratory prompt development after further tests.

The follow-up audit requested three additional hardenings. The CLI now
preserves unresolved private paths so the fd-anchored reader can reject
symlinks; the generator compares initially read exclusion/filter digests with
the exact replay snapshots and records those verified digests and dataset
hashes; and the receipt records the replay-loader source SHA. The HTML review
builder now uses the same dataset/lock rebinding and verified decoded frames,
then checks its source snapshots before rendering. Combined focused tests for
replay, generator and review-builder paths pass 25/25; compilation and scoped
diff checks pass. Replay reparses each raw generation string, requires exact
equality with its parsed claim and row-level/top-level prompt version, and
rejects parsed-claim tampering. The original v2 packet passes this reparse
check. The independent auditor confirmed this final hardening and the live
packet rebind before the GPU run.

## Verified GPU run and review handoff

After the independent audit cleared the repaired path, v3 was rerun on the
same frozen cohort11 rows. The private mode-0600 receipt is
`data/private_nextgqa/claim_generation_cohort11_gpu_v3_20261021_02.json`.
It records RTX 5070 Ti, pinned Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit/BF16, 16 frames per row,
3 requested/effective rows, 3/3 parse-structured claims, 20.05 seconds, and
7.78 GiB peak allocated VRAM. No direct score or model-selected evidence frame
was produced.

On the same three locked video/question pairs, v2 produced 2/3 parser-valid
claims and v3 produced 3/3. Paired transitions were 2 valid→valid and 1
invalid→valid, with no valid→invalid transition. Valid-claim word counts were
9 and 12 for v2, and 8, 10, and 12 for v3. This is a tiny, known-exposure
prompt-development comparison; it shows only improved adherence to this
parser's sentence-shape contract on the one failed row. It does not measure
question fit, visual support, claim truth, or faithfulness, and has no
independent cohort or inferential test.

The review builder re-bound the v3 receipt to the current test CSV, gsub,
video map, frozen lock and pair filter; recomputed the cohort digest; reparsed
raw generated strings; and decoded the bound frames. It created the local
mode-0600 handoff
`data/private_nextgqa/claim_support_frame_review_cohort11_v3_20261021_08.html`,
SHA-256
`e4222baf2620fe8e04f30d4f8f7af830e30a878ed1aec436fcc53bbcfe88e150`.
The artifact contains 3 review items × 16 sampled frames, no gold fields,
and zero unreviewable rows. Structural inspection found all 48 frame image
data URLs embedded; review responses remain **0**. The handoff does not reuse
legacy answers. Inline JavaScript syntax validation passes. A Windows Chrome
browser E2E attempt could not start: WSL interop returned
`UtilBindVsockAnyPort: socket failed`; it created no synthetic or human
responses. The previous browser E2E covered the same shared renderer, but this
specific artifact's interactive flow remains unverified in Chrome.

Provenance limitation: the receipt's recorded `replay_loader_sha256` does not
match the current loader file. The loader was adjusted after generation to
accept the generator's `passed` status while building the handoff; the
historical loader byte snapshot was not preserved, and the local packet cannot
cryptographically authenticate model emission. Generator and pilot source
hashes match their current files, and the current loader independently
revalidated the exact packet, dataset rows, prompt parse, cohort and frames.
Retain this as an exploratory, locally hash-bound handoff, not fully SSS or
authenticated inference provenance. Rights status remains unverified; all
materials stay local.

Combined focused parser, replay, generator and builder tests pass 25/25;
Python compilation and scoped `git diff --check` pass. No independent
semantic human review was performed.

## Interpretation ceiling and next action

Even a complete 3/3 structured result would show only that the changed
instruction can produce parseable candidate sentences on three known
videos. It would not demonstrate better semantic fit, evidence completeness,
or faithfulness. Bind any eventual human response to the exact new artifact;
do not reuse legacy ratings for new claim text. If v3 candidates fail format
or omit required event details on review, revise the prompt only on development
data and reserve a distinct held-out cohort for any confirmatory comparison.
