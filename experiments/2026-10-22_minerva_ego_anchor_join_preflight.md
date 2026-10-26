# Minerva-Ego anchor join and local video feasibility (2026-10-22)

## Purpose

Find an executable path to direct explanation-evidence evaluation after the
local NExT-GQA validation candidates were exhausted. Audit the official
Minerva-Ego rationale/object annotations against the matching HD-EPIC track
and frame metadata before spending GPU time or downloading a video collection.
All fetched files and derived details are local-only; rights status remains
unverified.

## Sources and provenance

Fetched the official Minerva-Ego JSON (4,282,928 bytes; SHA-256
`46707b6d30b6e5d4291569f48f6c89ec89091ddf589ddc4793d5a35faf4a0639`) and the
official HD-EPIC `assoc_info.json` (2,897,077 bytes; SHA-256
`f5c6852ccf400dea945758abb168df5b789e4bb13d8387f68ec17c0b9b751204`) and
`mask_info.json` (7,611,269 bytes; SHA-256
`2ea7ce0dfa5aff5a144e29953d02c19d372fed3f8eb8ce871c09e6eed55e551f`). They
are held under `/tmp`; no annotation text, videos, or masks were added to the
repository or published.

The Minerva-Ego file contains 1,160 unique question keys over 156 unique video
IDs. All 156 video IDs exist in both HD-EPIC metadata files. There are 1,055
questions with at least one `referenced_object_masks` entry (3,305 references
total). Each reference ends in a stable object ID that joined to an HD-EPIC
association; all 3,305 joined associations have tracks. Their 22,909 track
mask IDs all joined to a `mask_info` record with a frame number and bounding
box. The object display-name text agreed with the joined association on
3,298/3,305 references; seven ID joins have a display-name mismatch and must
be retained as an explicit quality limitation or excluded by a frozen rule.

The reasoning field is text, but the object-reference IDs are embedded inline.
An independent re-count using token-boundary matching found 3,186/3,305
references (96.4%) on a reasoning line that also contains a valid `mm:ss`
anchor; 1,036/1,055 referenced questions (98.2%) have at least one such
same-line object/time pair, and 979/1,055 (92.8%) have every referenced
object on at least one timestamp-bearing line. Under the conservative
timestamp-point rule (`mm:ss` denotes its displayed second, and that point
must lie in the object's track interval), 1,828/3,305 references (55.3%)
pass; 873/1,055 questions (82.7%) have at least one passing reference, while
334/1,055 (31.7%) pass for every reference. Under the pilot's more permissive
rounded-bin rule (`[mm:ss, mm:ss+1s)` overlaps the track interval), 2,217/3,305
references, 932/1,055 questions with any passing reference, and 474/1,055
with all references passing qualify. Seven display-name mismatches remain
among the 3,305 joins; requiring every display name to match reduces the
rounded-bin all-reference pool to 473 questions over 151 videos. These are
syntactic/metadata joins, not visual validation:
they do not establish that timestamps are semantically exact, that a line is an
atomic causal claim, or that a referenced object mask supports the line. Track
segments describe annotated movement and can be broader than the specific
event; a timestamp outside a segment does not prove the object is absent.
Frame numbers are converted and visually checked for the single selected video
in the GPU pilot section below; other rows remain metadata-only.

## Video retrieval status

The official HD-EPIC downloader supports a per-video ID filter. The
model-facing pilot row is a non-counterfactual movement question in
`P01-20240202-161354.mp4`, selected before any model outcome. The row contains
30 entries in `objects_of_interest`; `referenced_object_masks` contains one
exact-name matched white-mug object, which has two movement tracks and four
mask frames. Its key
suffix is
`b9f19f203842665262cfaf978348d269adbcd9da`. It asks about a mug's movement
through the video. Its authored trace has four object-linked mug lines; the
first timestamp is outside a movement interval, while later lines overlap the
two tracks. The pilot scored one dataset-authored line with its opaque
object-ID parenthetical removed. The official checksum manifest gives video
MD5 `a13e897ebcb918b2b22b79df2b89dc02` and size 360,732,493 bytes.

The candidate's official frame-to-VRS mapping CSV was fetched locally
(431,164 bytes; official MD5 verified as
`945519ec32718366642189f9cd6f78ad`; SHA-256
`8be741d5319b0ed345b3889f703719e2586f5afb37c74d3592912ac42fc2e8f1`). It has
9,828 strictly increasing MP4-time rows spanning 0 to 327.5667 s at a median
33.333 ms/frame. For the selected mug row, four mask frames map to 7.8667,
11.0667, 87.0333, and 94.3000 s. After checksum verification, sequential
PyAV decoding confirmed all four times exactly against decoded-frame PTS. The
MP4 has 9,828 frames at 30 fps. Bounding boxes overlaid on those frames were
visually consistent with the white mug at the counter, sink-side, and
dishwasher. This is a primary-agent spot check, not independent human
annotation validation. The official schema cautions masks and boxes may be
inconsistent.

The first candidate video `P01-20240202-110250.mp4` was rejected before model
inference because none of its five questions passed the all-reference
time-overlap rule; an incomplete local transfer remains without checksum
verification. The selected replacement was fetched in eight verified HTTP
byte ranges, concatenated, and matched the official file size and MD5.
Retrieval remains restricted to selected IDs; the official video collection
is 115.5 GB.

## Interpretation and next action

This is a materially stronger data lead than the earlier schema-only audit:
Minerva-Ego's human-authored rationale text, object IDs, HD-EPIC tracks, and
frame/bbox annotations have concrete joins for the complete 1,160-question
file. It can support a prespecified exploratory test of Qwen's dependence on
fixed, human-grounded reference-rationale lines under matched object/time
perturbations. That estimand is reference-rationale evidence dependence; it
does not by itself show that Qwen generates more faithful explanations.

At this preflight stage the NExT-GQA direct faithfulness status was: current-
artifact review 0/3, verified frame-time pairs 0/15, prediction links 0/3,
direct scores 0/3, and formal review 0/25. At that point no Minerva-Ego model
inference had run. The subsequent GPU pilot is recorded below and does not
change those direct NExT-GQA counts.

CPU implementation added in `scripts/minerva_ego_anchor_utils.py` to parse
rounded timestamp bins, retain object-linked rationale lines overlapping
track intervals, map mask frame IDs to sampled frames using actual PTS, and
select deterministic equal-count control temporal groups. The preflight suite
passed 8 tests; the corrected suite passes 11. These tests validate helper
arithmetic and contracts only, not efficacy.

## One-video exploratory GPU pilot (2026-10-22)

The selected MP4 was verified before inference: 360,732,493 bytes, official
MD5 `a13e897ebcb918b2b22b79df2b89dc02`, SHA-256
`fdc8048e7dcc3bce10d06ee3b1c7fd7267a6b5cbacd76af2ba93aa1e5fc2843c`.
PyAV decoded all 9,828 frames at 30 fps; the four mask frames mapped to exact
decoded PTS at 7.8667, 11.0667, 87.0333, and 94.3000 s. Their bboxes were
visually overlaid on the local frames and corresponded to the white mug, at
the counter, sink-side, and dishwasher. This is a primary-agent spot check,
not independent human annotation validation.

Executed `scripts/run_minerva_ego_anchor_gpu_pilot.py` with the pinned
Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
4-bit quantization, BF16 compute, eager attention, 64 uniformly sampled
frames, and `video_grid_thw=[32,22,22]`. The one fixed researcher-composed
sentence was teacher-forced under three conditions: baseline, +ln(2) attention
logit bias to annotation-derived groups [1,8,9] in layers 5–16, and the same
three-group bias to an equal-dose temporal sham [0,21,23]. Each condition was
one forward. Runtime for the three scored forwards was 33.77 s; peak allocated
VRAM was 11,286,223,872 bytes (10.51 GiB). The private output is
`data/private_minerva/minerva_ego_anchor_pilot_20261022.json` (mode 0600;
SHA-256 `3d2d94ebe7b147a9f7c5537f83be226d5bcdba261b9e05bc118f01ebb653e739`).
The run used runner SHA-256
`abfc7fa87607ab2e826c84b669da02743f97ba2188701bab881a9352b47a4cac`.

Observed researcher-paraphrase target mean log probability was −1.57045 nats/token at baseline,
−1.56222 with evidence-group bias, and −1.59219 with sham bias. Descriptive
differences were +0.00822 evidence-minus-baseline and +0.02997 evidence-minus-
sham nats/token. With one video/question and one target sentence there is no
sampling uncertainty estimate, significance test, or multiplicity correction;
these are not inferential results. The target is a researcher-composed
paraphrase of the dataset trace, not the model's generated explanation or a
human-reviewed label. This is an exploratory intervention/plumbing result,
not evidence that explanation faithfulness improved.

A target-text sensitivity rerun used the first dataset-authored rationale line
whose exact object ID and rounded `mm:ss` bin overlap the selected object's
track; only its opaque ID parenthetical was removed. It used the same video,
frame selection, model, layers, bias, evidence groups, and sham, so it is a
within-row sensitivity analysis, not an independent replicate. The second
private output is
`data/private_minerva/minerva_ego_anchor_rationale_line_20261022.json` (mode
0600; SHA-256
`179509b169ec3fb4c3fa95fae50c32d36167c239cb16c1963b8b30c7b51e4af1`). Its
runner SHA-256 is
`23c9fd34c9223a66d34c803be58aab7278e6b38bff5c63cd9e17f986f6e095a7`.
Scores were −2.93473 baseline, −2.99107 evidence-bias, and −2.97991 sham
nats/token: evidence-minus-baseline −0.05634, evidence-minus-sham −0.01116.
The sign therefore differs from the researcher-paraphrase target, showing
material target-wording sensitivity in this single row. Neither target yields
a stable intervention benefit, and neither is a generated explanation.

Post-run audit found the runner's logged `annotated_track_groups_excluded_from_sham`
field was empty because its first version checked group centers only. An
independent recomputation checked every sampled frame in each temporal group:
track-overlapping groups are [1,8,9], exactly the evidence groups; the actual
sham [0,21,23] lies outside both annotated intervals, and deterministic
reselection remains [0,21,23]. Thus the score comparison used the intended
out-of-track sham, but the run's exclusion metadata field is incomplete. The
helper and runner were corrected to exclude a group if any constituent sample
falls in a track, and an 11th unit test now covers that boundary. Corrected
runner SHA-256 is `bf26533479302b2373cb1f218d6c55d34a18beaf65a2bce4ec586a391032d89c`,
helper SHA-256 is `4c32f08111755f4f22c1836744b141ed2d6a0e675d251f7c7d2b44878881cb39`,
and test SHA-256 is `b9abf72e52be47a35d2fb3efe9f334de6122593d12f29adf36d2fa6dfabbd737`;
the saved run artifact remains immutable and corresponds to the runner hash
above. For the second run the corrected runner's exclusion field records
[1,8,9] as expected. Updated CPU suite passed 11/11. Runtime environment was
PyTorch 2.11.0+cu128 / CUDA 12.8, Transformers 5.18.0, bitsandbytes 0.50.2,
qwen-vl-utils 0.0.14, PyAV 18.1.0, and Pillow 12.3.0. JSON outputs do not
embed all runtime/package hashes; this log binds the runner hashes separately.

### Generated-output grounding probe

To test whether the intervention changes generated outputs, the same frozen
single row/video and 64-frame setup was run with deterministic generation
(baseline, evidence-group bias, equal-dose temporal sham; 96-token cap). The
prompt requested movement-only descriptions and approximate `mm:ss` anchors.
All three generation calls completed in 56.92 s; peak allocated VRAM was
11,462,822,400 bytes (10.68 GiB). The output is local-only at
`data/private_minerva/minerva_ego_generated_grounding_20261022.json` (mode
0600; SHA-256
`2754a2a78f4c495ecf48f410ec8879eb9f3b118573090f8a1ceb31ac097c1346`), bound
to runner SHA-256
`e1df810f55f4954311435b79b5052f14837a49eef9a62b4d643e823ea04d2a6b`. It
records one effective row, three completed calls, evidence groups [1,8,9],
sham groups [0,21,23], and confirms [1,8,9] are the track-overlapping groups
excluded from sham selection.

The generated outputs contained zero parsable mug/cup time anchors in all
conditions. Consequently lexical anchor precision was 0/0 (defined as 0 by
the scorer) and movement-segment recall was 0/2 in each condition. This is a
prompt-following/measurement failure for this row, not evidence of equal
faithfulness or of no intervention effect. The run does not evaluate action
semantics, and there is no human review. Direct generated-explanation
faithfulness remains unmeasured. The 12 CPU helper tests pass, including the
anchor scorer contract; they do not validate model behavior.

The private output showed the concrete formatting error: all conditions
generated the literal-like token sequence `0mm:ss` and no parseable time.
The prompt was revised to explicitly require numeric `M:SS` values and forbid
the placeholder. A same-row/same-video/same-frame deterministic baseline,
support-bias and sham diagnostic then completed 3/3 generation calls in
54.58 s, with peak allocated VRAM 11,570,233,856 bytes (10.78 GiB). The
private output is
`data/private_minerva/minerva_ego_generated_grounding_format_v2_20261022.json`
(mode 0600; SHA-256
`f41dbf3508804b1afa25273adba42774530d091bfac0925e323ab0b9b8744403`), bound
to runner SHA-256
`3cae880dc6c951eb924d32f7ca5e65d3dd0c661a952077a35df1d2511f78183c`.
All conditions emitted four parseable named-object timestamp anchors; one of
four overlapped an annotated movement interval (precision 1/4), and one of
two annotated intervals had a matching anchor (recall 1/2) in every condition.
The support-bias generation text differed from baseline/sham, while all three
conditions had identical anchor metrics. Hook audits confirm the intended
support/sham bias was applied at the configured layers; this single greedy
decoding row cannot establish that internal intervention improved grounding.
The same 12 CPU helper tests and scoped `py_compile`/whitespace checks passed.

## Current faithfulness assessment and next action

Direct generated-explanation faithfulness remains **unmeasured**: the
NExT-GQA status remains review 0/3, verified frame-time pairs 0/15, prediction
links 0/3, direct scores 0/3, formal review 0/25. The Minerva pilot completed
one real-video reference-likelihood proxy row, not a direct faithfulness row.
Next, freeze an outcome-blind multi-video cohort of single-reference movement
questions with exact display-name joins and rationale timestamp bins
overlapping the object's annotated tracks. Keep this numeric timestamp prompt
and lexical/time metric fixed for a small feasibility replication, but treat
that automated score only as a grounding proxy: action semantics and causal
support still need independent validation. The current one-video results
cannot establish an intervention effect. Direct NExT-GQA support-dependent
faithfulness remains gated by the absent exact-artifact review response.

### Frozen multi-video feasibility cohort (pre-outcome for these rows)

Applied a deterministic selector to the complete local Minerva-Ego/HD-EPIC
metadata, excluding rows tagged counterfactual, requiring an Action Recognition
tag, exactly one referenced object, a single-word object label, exact display
name/association match, a same-line object-ID plus rounded-time-bin/track
overlap, and successful joins for every referenced mask ID. This yields 46
eligible rows across 38 videos. The cohort contains three distinct videos,
one row per video, selected by fixed SHA-256 ordering; this is an exploratory
feasibility sample, not a probability sample or confirmatory holdout. Its
private mode-0600 selection receipt is
`data/private_minerva/minerva_ego_generated_grounding_cohort_v1_20261022.json`
(SHA-256
`09f03c48d0a56bdfddf5fdf80aae9ac07b868b0a4a5ae9f97e1b85e37eb7ce97`). The
selector SHA-256 is
`8edf25122053fb74e1c2ea707d9cc9bd4c80acb54891cd660aa8d68429f8750f`.
Three focused selection tests, 14 anchor tests, and three range-geometry
tests pass (20/20 total); Python byte-compilation and scoped whitespace checks
pass.

The official manifest's HEAD sizes for all video-specific files were 4.237
GiB; the actual transfer now requests only the three MP4s and their three
MP4-to-VRS time maps. The official HD-EPIC downloader's unmodified sequential
transfer was tried, but after about three minutes it had delivered only 24 MiB
of the first 1.13 GiB video. It was interrupted without deleting its partial
file. An eight-way 8-MiB range smoke measured 0.73 MiB/s; a 24-way 1-MiB smoke
measured 2.71 MiB/s. A private bounded downloader was added to fetch only the
frozen MP4/map files in parallel ranges, verify each official MD5, refuse
existing mismatched outputs, and write under the private data root. Its source
SHA-256 was
`2810c4983054594b477055cdbd0a12ea160c6d459d61f1e23997df2c6500d3f5` at
launch; the source was subsequently hardened to use a unique assembled-temp
filename. Official
script and manifest hashes are `07fcb487beb7e241660042e5d2ab981bc3ac4efef9a5e857f4ea7cc05216e4cc`
and `4b2bc0f8bae6c27d3521d7f319b3a9df16652da1f0492be3ff01261fd8197c06`.
At this log update, range retrieval is still active and no inference has run
on those three frozen rows. Rights remain unverified; all downloaded material
stays local.

### Retrieval and preflight completion update (2026-10-22)

The preceding paragraph records an intermediate state and is superseded here.
The bounded range downloader completed all six requested files (three MP4s and
three time maps), verifying each against the official MD5 manifest. Total
downloaded bytes were 4,548,930,349. Files were stored under the private data
root with mode 0600; rights remain unverified and the data remain local.

The CPU preflight then passed all 3/3 frozen row/video bindings. Each row's 64
uniformly selected mapped PTS decoded successfully, with maximum measured drift
below 5e-13 seconds; all rows had both support and sham groups. This establishes
input integrity and executable sampling geometry, not model efficacy.

A GPU smoke was attempted for the first frozen video, three generation
conditions. The runner stopped before model inference because only 231 MiB of
16,303 MiB total GPU memory was free (54% utilization); its preflight contract
requires at least 12 GiB free. No new cohort generation result was produced.
This is a transient shared-GPU capacity blocker, not a CUDA or dataset failure.
Do not start another copy or relax the guard while the GPU is occupied. Retry
the same frozen row only after free VRAM meets the existing threshold, then
continue one video at a time. The run remains exploratory and its automatic
time-track scores remain grounding proxies rather than direct faithfulness.

On a subsequent checkpoint the same capacity condition persisted: 215 MiB
free, 15,781 MiB used, and 47% GPU utilization. The runner's free-VRAM guard
again prevented model loading. A full independent local re-hash then verified
all 6/6 downloaded files against the official MD5 manifest and mode-0600 file
permissions. The focused Minerva cohort, downloader, preflight, and anchor
utility test suites also passed 25/25 tests. These checks strengthen input and
runner readiness but add no model-output or faithfulness evidence.

After successful download verification, the next run is three videos × three
deterministic generation conditions (nine total generations). The previous
single-video measured peak was 10.78 GiB allocated VRAM; current free VRAM and
each video's frame count must be rechecked before inference. At the observed
24-way smoke rate, the 4.237-GiB upper estimate is approximately 27 minutes;
actual speed can vary. Report per-video anchor precision/interval recall and
within-video support-minus-baseline/sham differences descriptively only. This
automatic time-track proxy does not adjudicate action semantics, causal
support, or direct explanation faithfulness.

To reduce CPU video-decoding cost, the generation runner now can seek directly
to the 64 uniformly selected frame PTS from the official time map. On the
already checksum-verified 9,828-frame pilot video, all 64 seek-decoded RGB
frames were pixel-identical to the prior full sequential decode and every PTS
matched (maximum drift 0 s). Seek decoding took 36.32 s versus 67.02 s for
full decode under concurrent dataset transfer. This is a modest speedup on
this short video, not a guaranteed speedup for other encodings; new rows still
fail closed on frame-count or PTS-map mismatch. The multi-video run will use
this path and independently verify every selected frame's mapped PTS before
model processing.

The first CPU temporal-grid calculation showed a standard 32-frame grid misses
the second movement interval: nearest samples 84.533 and 95.100 s both fall
outside [87.043,94.31355]. A 64-frame configuration with an approximately
equal visual-token budget was frozen before scoring. Actual Qwen processing
produced 32 temporal steps and support groups [1,8,9]. The provisional
preflight sham [26,27,30] was superseded by run-time sham groups [0,21,23].

## External source references

- [Official MINERVA collection README](https://github.com/google-deepmind/neptune)
- [Official HD-EPIC annotation schema](https://github.com/hd-epic/hd-epic-annotations/blob/main/README.md)
- [Official HD-EPIC per-video downloader](https://github.com/hd-epic/hd-epic-downloader)
- [HD-EPIC dataset page and license statement](https://hd-epic.github.io/site/)

## Three-video GPU feasibility run and prompt repair (2026-10-22)

### Initial frozen-cohort run

The three-video cohort was run once in baseline, object-time-support-bias,
and equal-dose temporal-sham conditions (9 greedy generations total). The
selection receipt is
`data/private_minerva/minerva_ego_generated_grounding_cohort_v1_20261022.json`
(SHA-256
`09f03c48d0a56bdfddf5fdf80aae9ac07b868b0a4a5ae9f97e1b85e37eb7ce97`). The
initial row output hashes were `6ebf8e41536bf81e8e3494a92d5c2e723b50df56ad4071c93cadfc77e44fe3b1`,
`c4f876701dcb1260716bc2c15adc5a9bb7e96a3e78cbe0aa2b3a9bdbc7957769`, and
`1461fb50f14dfe7d34dadd04f2e87e12bf5920d488a4e89e8b52327f340d8de2`.
Each row was bound to the frozen row key and the official video MD5.

An audit after this run found that the prompt named every target as “mug,”
although the three frozen annotation names are different. These initial outputs
are retained for provenance but excluded from interpretation because the prompt
did not consistently identify the annotated target. They must not be treated as
a valid test of the intended object-conditioned generation.

### Prompt correction and exploratory rerun

Added `build_grounding_prompt` to construct the target name and accepted final-
word alias from the joined annotation for each row. A unit test asserts that a
non-mug object is named correctly and that the old fixed “mug” instruction is
absent. The focused selector, downloader, preflight, and anchor/prompt suite
passes 26/26 tests. Corrected runner SHA-256 is
`b698600fcad7222f1652369f54fc470b201ad40da3054f48f719bfd16b8bc55f`.

The same already-inspected three-video cohort was rerun under the corrected
prompt with new mode-0600 local outputs; this is an implementation-repair
exploration, not a confirmatory replication. The outputs are
`data/private_minerva/minerva_ego_generated_grounding_cohort_v1_promptfix_row01_20261022.json`
(SHA-256 `2a801d1d292c32e3cb6e274279eec1c51607330f5b24442347bc40983e1b5e6e`),
`..._row02_...` (SHA-256
`d48268d4744b6879f9ca59eb9c2e867ff20fdcc5ab9188a861329a0905478600`), and
`..._row03_...` (SHA-256
`cab40c736228b006333e1daba44bf89a70e883737744768cc6ff88ddcb40e455`). All
three runner, frozen-row, and official-video-manifest bindings passed. All 9
generation traces matched token counts, and all 6 support/sham interventions
passed the hook/cell audits; baseline had no intervention hooks. Model revision
was `cc594898137f460bfe9f0759e9844b3ce807cfb5`, Qwen2.5-VL-7B-Instruct, 4-bit
with bf16 compute, eager attention, and deterministic greedy decoding. The
effective cohort was 3 rows/3 videos, with 3 conditions each; per-row runner
times were 51.5, 67.7, and 58.5 s and peak allocated VRAM was 10.61–10.64 GiB.
Rights remain unverified; data and generated text remain local.

### Observations and limits

The fixed automatic metric remains at floor after the prompt repair. Across
the 3 videos, baseline/support/sham generated 5/5/5 timestamp anchors; 12/15
were lexically associated with the target, and **0/12** of those overlapped an
annotated movement track. All 10 annotated movement segments (1+6+3) had zero generated
anchor coverage in every condition. Per-video support-minus-baseline and
support-minus-sham differences were both exactly 0 for anchor precision and
segment recall (n=3 paired videos). No inferential test or population
confidence interval is warranted: the cohort is small, deterministically
selected, and the proxy score has a floor. The result does not establish
equivalence or absence of a faithfulness effect; it shows this generation setup
did not produce a discriminating proxy outcome on these rows.

The intended target-specific generation measurement is therefore a feasibility
failure: even after correcting the target-name instruction, the model did not
anchor its timed movement sentences to the annotated movement intervals, and
support bias did not change either proxy score relative to baseline or sham.
Do not increase sample size on this unchanged free-generation protocol or tune
prompts against these outcomes. Reconsider the measurement before any further
generation run. Direct explanation-faithfulness counts remain unchanged at
0/3 claim reviews, 0/15 verified frame-time decisions, 0/3 prediction links,
0/3 direct scores, and 0/25 formal support reviews.

## Frozen exploratory alternative: reference-rationale likelihood

Before the next GPU scoring run, use the same three pre-outcome-selected
video/row bindings, without adding or replacing rows. For each row, select the
first dataset-authored reasoning line in source order that contains the exact
referenced object ID and a rounded one-second timestamp bin overlapping that
object's annotated movement track; strip only the numbered prefix and opaque
object-ID parenthetical. Freeze that line by SHA-256 in the private output.
Teacher-force its tokens after the row's question, answer choices, and dynamic
joined object name. Score mean target-token log probability under baseline,
mask-frame-derived object/time support-key bias, and deterministic equal-dose
track-disjoint temporal sham. Use 64 uniformly sampled frames and the existing
12-layer, +ln(2)-nat attention-key intervention. Primary descriptive contrasts
are support minus baseline and support minus sham in nats/token, paired within
video; keep all three rows and all failures. Do not run a hypothesis test or
claim population effects at n=3. This is an exploratory annotation-backed
prediction-dependence proxy, not a direct generated-explanation faithfulness
score: the current rows have already been observed in the free-generation
pilot, the cohort was geometry-selected, and the rationale lines are not
independently reviewed.

### Observed GPU results (exploratory)

The generalized runner
`scripts/run_minerva_ego_anchor_gpu_pilot.py` was updated to bind arbitrary
single-reference rows to their video and expected checksum, select the first
annotation-linked rationale line with a track-overlapping rounded timestamp,
use the row's actual object name in the question prompt, and sample frames via
the official PTS map. It records per-layer scoring hook audits. The target
selector has a focused CPU test; the full Minerva/download/preflight suite now
passes 27/27 tests and the relevant scripts byte-compile. Runner SHA-256 is
`f110c526ca24b1c8851fea6bedc03ba856a4572229b116a20734a6add7a8f0de`.

Three rows completed, each with three scored forwards. Outputs are mode 0600
and remain under the private data root. Their hashes are `c4fad81e410d46617681de4daaba17638e5da2531f762332c9839cb64f9c383c`,
`2093e91c91d88e4a633e41d039fc5a40378ae6daa88f686b61148015079e27c7`, and
`e8ed06de3b416abe306e8f3cf33fb732f2641af6be96fdc98d3f9c21e0593ace` for
rows 1–3, respectively. Independent post-run checks confirmed frozen-row
hashes, official video MD5 bindings, the current runner hash, identical
support/sham visual-key dose, and one verified hook call per layer (5–16) for
each intervention condition. Model revision, precision, frame budget, and
attention setup match the preceding Minerva pilots: Qwen2.5-VL-7B-Instruct
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
64 frames, eager attention, and +ln(2) visual-key bias. Scoring took 30.5,
52.5, and 22.6 s per video; peak allocated VRAM was approximately 10.4 GiB.

| Frozen row | Support − baseline (nats/token) | Support − sham (nats/token) |
|---|---:|---:|
| 1 | +0.076259 | −0.038490 |
| 2 | +0.050283 | −0.004096 |
| 3 | +0.008642 | −0.020956 |
| Mean (n=3 videos) | +0.045062 | −0.021180 |

The reference line's mean token log probability rose under support bias versus
baseline in all three videos, but was lower under support bias than under the
matched temporal sham in all three. This indicates sensitivity to an attention
bias, but no evidence that target-specific evidence bias improves likelihood
more than a dose-matched irrelevant temporal bias. With three geometry-selected
videos already exposed to prior exploratory outputs, report only paired
descriptive differences: no inferential test, confirmatory interval, or
population claim is justified. This score measures model likelihood of one
dataset-authored rationale sentence conditioned on the video and internal
visual-key bias; it does not establish that a generated explanation is more
faithful or semantically correct. Direct faithfulness counts remain unchanged.

### Independent temporal-sampling audit and gate repair

An independent audit then compared rationale timestamp mask frames against the
actual 64 uniformly sampled decoded PTS values and checked whether each selected
evidence temporal group contained a sampled frame inside the annotated object
track. It found (same three previously exposed rows):

| Row | Median / maximum mask-to-sample error (s) | Evidence groups overlapping sampled track groups |
|---|---:|---:|
| 1 | 0.117 / 0.167 | 2/2 |
| 2 | 6.483 / 9.500 | 1/7 |
| 3 | 8.033 / 11.200 | 0/2 |

This materially weakens the earlier interpretation: for rows 2 and 3, most or
all attention-biased groups do not contain any actually sampled frame inside
the annotated object track. Even row 1 only localizes time; the intervention
covers all visual patches in selected temporal groups, not object-localized
image patches. Therefore the likelihood differences above are retained as an
attention-bias plumbing/sensitivity exploration and cannot be interpreted as a
reliable target-object evidence intervention. They provide no evidence of
faithfulness improvement.

The runner now fails before model loading unless every mask frame maps within
1.0 second of the uniform sample and every evidence group intersects a sampled
track-overlap group. Added tests cover accepted geometry and both rejection
conditions. This is a code/protocol repair only; the old output hashes remain
bound to the runner version that produced those outputs, and no GPU rerun has
been performed under the new gate. Next, screen candidate rows on CPU using
this exact alignment rule, then freeze a fresh cohort before considering
another GPU measurement. A passing temporal gate would still establish
frame-level, not object-patch-level, alignment.

### Frozen-cohort CPU rerun under the repaired gate

The repaired preflight was then run against all three rows in the frozen
exploratory cohort; it decoded the original local videos and performed no model
loading or GPU inference. Result: 1/3 passed and 2/3 failed before inference
with `mask_to_sample_time_error_above_threshold`. The passing row had a maximum
mask-to-sample error of 0.1667 s, two evidence groups, and both groups intersected
sampled track-overlap groups. The private receipt is
`data/private_minerva/minerva_ego_generated_grounding_cohort_v1_alignment_gate_20261022.json`,
SHA-256 `eaffa56fed0d4075d777c367ae47be39fb1e13a2789ee682ed7cdf9feca20916`,
mode 0600. It records the frozen cohort hash, manifest hash, source hashes,
requested/effective counts (3/1), 94.33 s elapsed, no human review, and
unverified rights status. The gate rejects the two misaligned rows before
inference as intended.

This does not justify running a GPU experiment on the single passing row: it is
part of a previously exposed, geometry-selected exploratory cohort, and the
intervention remains frame-wide rather than object-patch-local. The next
scientifically useful step is a fresh-cohort CPU eligibility and exact-sampling
screen across available candidate media, followed by freezing before any new
outcome. Direct claim-specific faithfulness remains unmeasured.

## Fresh focal-clip exploratory GPU cohort (2026-10-22)

### Cohort and protocol changes

To avoid reusing the three previously exposed videos, the deterministic
metadata selector added explicit video-ID exclusions and froze three new rows
from the remaining 35 eligible videos (46 eligible rows overall). Cohort
SHA-256: `d0f1a8d2d75a68eb0b169468027578f10c1bf7ad61c09f3e84b6d104f3c4c563`.
Three new MP4s and official time maps were downloaded locally (1,874,234,406
bytes) and all six official MD5 checks passed. Rights status remains unverified;
all media and outputs remain under the mode-0700 private directory.

The original global-uniform 64-frame sampler failed the stricter geometry gate
on all three new candidates: maximum mask-to-sampled-frame errors were 8.1,
9.4, and 1.1 seconds. One candidate had only 2/4 evidence groups overlapping
sampled tracks; another had 4/7. A first attempt to inject one exact mask frame
into the global sample still did not provide honest Qwen timestamps: for this
list-of-images API, `sample_fps` creates uniform synthetic video metadata, so
one replaced frame could be assigned a source time up to 9.3 seconds away.
That version was not used for inference.

The frozen cohort then used a 31.5-second local clip centered on a mask frame
from the first object-linked rationale line whose displayed one-second bin
overlaps its annotated track. Sampling is 64 frames at 2 fps, so the input
frame schedule and Qwen's relative temporal metadata agree. The prompt states
the clip's source-time offset. A 50 ms track-boundary tolerance accounts for
MP4 frame quantization when a rounded track boundary falls between frames.
Preflight decoded all three videos, verified exact source-frame PTS, mapped the
selected annotation frame to one support temporal group, confirmed that group
lies in a sampled track interval, and found an equal-dose track-disjoint sham.
The final CPU receipt is
`data/private_minerva/minerva_ego_generated_grounding_cohort_fresh3_focal32s_cpu_preflight_20261022.json`
(SHA-256 `e1464455040bcbdf2d39bc3185522aeaee52aeca33b1a7e1d31500c44ae5eed0`,
mode 0600): 3/3 passed in 102.95 s. No model was loaded during preflight.

### GPU execution and descriptive results

One initial GPU smoke completed three conditions on row 1. Its exact scores
were repeated after adding cohort-bound and source-hash provenance; the values
matched exactly. The three primary outputs then used the same runner and frozen
cohort. Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
64 frames, 2 fps, eager attention, and +ln(2) visual-key bias on layers 5–16.
Each video had baseline, annotation-time support bias, and equal-dose temporal
sham bias: 9 primary forwards total. Every branch used the same target-token
IDs; support and sham each biased 121 visual keys; deletion count was zero;
each of layers 5–16 had one verified hook call per condition. Peak allocated
VRAM was 10.45–10.61 GiB. The receipts independently bind video SHA-256/MD5,
time-map SHA-256, frozen cohort SHA-256, runner/helper source hashes, runtime
versions, and private output mode 0600.

| Frozen row | Support − baseline (nats/token) | Support − sham (nats/token) |
|---|---:|---:|
| 1 | +0.011003 | +0.010321 |
| 2 | −0.057439 | −0.043263 |
| 3 | +0.005227 | −0.015127 |
| Mean, n=3 | **−0.013736** | **−0.016023** |

These are descriptive paired contrasts only; no test or confidence interval is
reported at n=3. The support-minus-sham direction is negative in two of three
rows, and the support-minus-baseline direction is negative in one of three.
This exploratory likelihood proxy does not show a consistent gain from
annotation-time support bias. It is not a generated-explanation score: targets
are dataset-authored rationale lines, not independently reviewed claims. The
intervention covers all visual patches in one temporal group, not only object
pixels, and the clip omits the rest of the source video. Human review remains
0/3 for the current claim handoff (formal review 0/25); direct claim-specific
faithfulness scores remain 0/3.

Reproducible aggregation is in
[`analyze_minerva_ego_focal32s_pilot.py`](../scripts/analyze_minerva_ego_focal32s_pilot.py).
It validated all row/cohort/source bindings, target-token identity, intervention
budget, layer-hook counts and stored contrast arithmetic. The private summary
is `data/private_minerva/minerva_ego_rationale_focal32s_analysis_20261022.json`
(SHA-256 `eac31a6e6ac97c960f20eb6ef33294f01e1ac03cbf0d707cb01d2a188ba0a974`,
mode 0600). Forty-two focused tests, byte-compilation, and scoped whitespace
checks pass. No files were committed or published; the broader worktree has
pre-existing unrelated modifications and private results are local-only.

### Next action

Do not expand this likelihood pilot or tune its clip, layer, or bias parameters
against these three outcomes. Its result is mixed and slightly negative. Next,
inspect local availability and representation of the dataset's spatial object
masks, then implement a predeclared object-pixel intervention on a fresh cohort
if those masks can be bound to the exact model-input frames. Otherwise return to
the current fixed-claim NExT-GQA scorer and preserve the human-review artifact
as pending; no automated label should be treated as a human judgment.

### Provenance clarification and spatial-mask inventory

The CPU preflight's `decoder_source_sha256` is
`dbaa39391edf43c976d894e85095167c305108d85aa05abddd893e22515b8615`, which
does not equal the final primary runner hash recorded in row receipts. Direct
hash inspection resolves the discrepancy: it equals the runner hash in the
initial row-1 smoke receipt, produced after the CPU preflight and before the
final cohort-binding/provenance update. Row 1 was then rerun under the final
runner (`08ea70aca76b7d4cb0eb3d9345dbfd6583cd2308bc215bf8474f22619432462b`)
and the two contrast values were exactly repeated. Thus the preflight and
initial smoke used the same runner source; the final three primary receipts
bind to the later runner. This supports the recorded sampling-method lineage,
while the final receipt remains the authoritative primary-output provenance.

An outcome-blind inspection of `/tmp/hd_epic_mask_info.json` found per-object
records with `frame_number`, `bbox`, `3d_location`, and `fixture` fields (156
video IDs in this local metadata file). This is bounding-box metadata, not a
pixel segmentation mask. The fresh cohort preflight found 12–14 object-mask
frame records per video; only one annotation frame per row was selected for
the likelihood pilot. A spatial follow-up can therefore test a bounding-box
pixel intervention after mapping each selected box onto the exact decoded and
resized model-input frames; it must not be described as segmentation-masked
object pixels. The current three-row pilot still offers no consistent
support-specific likelihood gain and does not measure direct faithfulness.
