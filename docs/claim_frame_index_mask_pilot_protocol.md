# Discrete frame-index claim interval pilot

This exploratory pilot replaces the closed timestamp-output localization branch
with a bounded discrete interface. For each of three unique source videos in a
private claim-generation packet, the pinned local Qwen2.5-VL-3B checkpoint sees
a deterministic 4×4 contact sheet of the exact 16 RGB frames decoded by the
repository's endpoint-inclusive sampler. Each cell carries its zero-based
frame index. Generation is constrained to the finite set of canonical JSON
strings for inclusive 1–4 frame spans, so malformed or out-of-range outputs
cannot be selected. The parser additionally accepts one unambiguous JSON
object when wrapped in markdown or brief prose, while rejecting duplicate
keys or multiple schema-matching objects. These protocol repairs followed
strict-JSON smoke/full format failures and remain exploratory.

The pinned local Qwen2.5-VL-7B checkpoint scores the frozen five-option QA
gold-versus-best-foil next-token log-probability margin under full video,
black and blur masking of the proposed contiguous interval, and matching
black and blur masks of an equal-count nearest-rank temporal control. Smoke
mode runs fixed row 0 and requires all five forwards to pass. Full mode requires a separate passed
smoke receipt bound to the same packet and checks row-0 localization
reproducibility before running all three rows (15 scorer forwards). Rows are
never replaced or survivor-averaged; any incomplete triplet invalidates the
aggregate.

The smoke receipt also pins hashes of the runner and its decoding/scoring
helpers, installed inference-package versions, BF16 precision, and 4-bit
quantization. Full mode rejects a receipt if any of these differ. Model
identity is tied to the local Hugging Face snapshot revision and `config.json`
hash; local checkpoint weight files are trusted to match that pinned cache
revision and are not individually re-hashed. Receipts record the GPU model and
memory capacity, initial free VRAM/host memory, peak allocated VRAM, and elapsed
time; full mode requires the same GPU model and capacity as smoke.

Outputs are local private receipts only, created with mode 0600. The receipt
records packet/video byte hashes, checkpoint revisions/config hashes, prompt
template hash, seed, coverage, and numeric margins, while omitting question,
choice, claim, source identifier, and path text. The measure is named
`machine_proposed_interval_dependence`. Human review collection is waived and
human review was not performed. This can at most provide an exploratory visual
sensitivity replication: a margin change after masking cannot establish that a
claim is true, that the proposed frames support it, or that an explanation is
faithful. The localizer and scorer are from the same model family, so shared
model bias remains. The 16-frame sampler's timestamp grid is checked against
the packet before localization; mismatch aborts rather than changing the
interface.

The implementation and tests are `scripts/run_claim_frame_index_mask_pilot.py`
and `scripts/test_run_claim_frame_index_mask_pilot.py`. No model run is implied
by these protocol definitions.
