# Quick-review response mapping audit — 2026-10-19

## Purpose

Check whether locally imported responses to the nickname-first exploratory
three-video check can be tied to the currently distributed private artifact.
The review is intentionally a coarse impression check; it is not a complete
hallucination annotation or a formal faithfulness measure.

## Method and observed result

- Counted regular JSON response files in the owner-only local response
  directory and compared content hashes without printing response contents.
- Revalidated the unique response bytes against every local quick-review HTML
  artifact with the current item-ID parser. Also searched private text files for
  review tokens whose derived opaque IDs matched the response item IDs; only
  file counts and candidate paths were emitted.
- Observed **8 files, 1 unique byte content**. The unique response does not
  match the item IDs in the current v4 artifact or the locally present v2/v3
  item-ID artifacts. No matching source token was found among 433 scanned
  private text files. The inspector rejects the response with
  `response_item_mapping_or_judgment_invalid`.
- Nickname, notes, raw item IDs, question/claim text, and video bytes were not
  included in this log or terminal output. No response was deleted or changed.
- Follow-up filename-only search covered `/home/ato121` and the Windows user
  tree under `/mnt/c/Users/taket` for quick-review/NextGQA HTML, JSON, and
  exploratory response exports. It found no additional matching artifact or
  export. No file contents were opened by that search.

## Interpretation and limits

These files cannot be counted as eight reviewers: they are byte-identical
copies. Since the exact artifact/item mapping is unavailable, even the single
unique set of coarse choices cannot be safely assigned to the current three
questions. The files therefore provide no usable human judgment for the
current candidate set and do not change formal review counts. They must not be
used as claim truth labels, hallucination labels, or faithfulness outcomes.

The current lightweight artifact still offers the intended flow: enter a
nickname, then answer three embedded-video questions without choosing files or
timestamps. Its questions ask for a broad impression only. A fresh response
must be saved from the current v4 artifact and imported against that exact
artifact before even exploratory counts can be reported.

## Faithfulness status and next action

Direct faithfulness remains **0/3 formally reviewed claims**, **0/15**
verified player-clock/decoder-frame correspondences, **0/3** prediction-link
judgments, and **0/3** direct scores. There is no direct `ΔF`, interval, or
test. The informal responses add no valid evidence. The head-intervention
confirmation remains closed at 79/79, −0.0756 nat, 95% t CI
`[−0.1578, +0.0065]`, below the preregistered +0.25 nat practical threshold.

The next useful handoff is to reopen the current v4 artifact, enter a nickname,
answer the three coarse questions, save its JSON locally, and import that file
with the v4 artifact path. For formal direct-faithfulness scoring, separately
obtain the current cohort's externally trusted reviewer/root/release materials,
claim review, 15 clock/frame correspondences, frozen baseline `p/q`, and
prediction-link review. GPU availability does not replace these inputs; no new
direct-score GPU run is justified before they exist. Rights status remains
unverified and protected files remain local.

## Follow-up recount (2026-10-19)

A primary-side recount of regular JSON files in the private response directory
observed **9 files and 1 unique SHA-256 content**. This corrects the earlier
8-file count; it does not change the mapping result or the number of usable
responses. Filenames and response bytes were not emitted. All nine files still
fail to provide usable judgments for the current artifact item IDs, so formal
claim review remains **0/3** and direct scoring remains **0/3**.
