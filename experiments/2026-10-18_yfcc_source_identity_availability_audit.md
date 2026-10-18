# YFCC source-identity metadata availability audit

## Purpose

Check whether the local 490-video candidate pool can be joined to authoritative
source identity or event-lineage metadata after the fixed pHash screen found no
candidate pairs. This is an outcome-blind metadata-availability audit. It does
not inspect videos or measure explanation faithfulness.

## Method and local results

Read the local NExT-GQA dataset documentation and `map_vid_vidorID.json`, then
searched repository filenames (maximum depth 8) for VidOR/YFCC/source-video
metadata artifacts. A bounded Python check read only the 10,000-entry ID-to-path
mapping and printed aggregate properties; it did not print identifiers or
paths.

Reproduction of the map aggregate:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
items = json.loads(Path("NExT-GQA/datasets/nextgqa/map_vid_vidorID.json").read_text()).items()
rows = list(items)
paths = [Path(path) for _, path in rows if isinstance(path, str)]
print({
    "entries": len(rows),
    "string_paths": len(paths),
    "unique_paths": len({str(path) for path in paths}),
    "video_id_equals_path_stem": sum(
        isinstance(path, str) and Path(path).stem == str(video_id)
        for video_id, path in rows),
    "unique_path_stems": len({path.stem for path in paths}),
    "all_paths_have_two_components": all(len(path.parts) == 2 for path in paths),
})
PY
```

Observed for the local map: **10,000/10,000** entries contain a string path;
all **10,000** paths and path stems are unique; every path has two components;
and the basename stem equals the NExT-GQA video ID for **10,000/10,000** rows.
The local map README defines this file as mapping each QA video ID to a video
path. It does not include an original-media hash, creator/source metadata, or
an event/recording relation.

Reproduction provenance: Git HEAD was
`a9c08f4cccc51e0b15a5cc180895bf165c3d1e14`; interpreter was Python
`3.12.14`. SHA-256 of `map_vid_vidorID.json` is
`1955d89e7a175346faf8d4a582297ff349619231394f59fd7872f459ac8a8685`; SHA-256
of the local README is
`2a1bfef65a7f569002d4258d4943ac052ff6eb3b3f18276ae4531efcb6d6fbe8`.
The exact bounded filename inventory command was:

```bash
find . -maxdepth 8 -type f \( -ipath '*vidor*' -o -ipath '*yfcc*' -o -iname '*video*hash*' \) -not -path './.git/*' -print
```

The bounded local inventory of `NExT-GQA/datasets/nextgqa` contains the path
map, benchmark QA/grounding/time files, but no VidOR annotation JSON or YFCC
ID-to-hash index. A repository path/name search for `vidor`, `yfcc`, and
`video*hash` found no other source-hash artifact; it also found two unrelated
legacy video-time metadata result files. This is a local inventory result, not
a proof that no copy exists outside the searched workspace. No protected
review package, review export, or video was opened for this audit.

## Source interpretation

The [official NExT-GQA dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)
defines `map_vid_vidorID` only as video-ID-to-path mapping. The [VidOR dataset
documentation](https://xdshang.github.io/docs/vidor.html) identifies its
`video_id` as a YFCC100M ID and separately provides a `video_hash` field for
indexing the source media. It lists train/validation videos and annotations,
while stating that test videos and annotations are available after
participating in the grand challenges. The 490-item pool documented in
`2026-10-18_qta_candidate_pool_visual_overlap.md` is drawn from the local
NExT-GQA test-ID mapping, so the public train/validation annotation files do
not supply a crosswalk for that pool. Thus the unique local
path stems verify distinct benchmark item IDs, but do not verify distinct
source media or independent recording/event provenance. A YFCC file hash
could expose repeated identical source media under different IDs if available,
but different hashes would still not prove independent events. The
[Multimedia Commons index guidance](https://multimediacommons.wordpress.com/preparing-a-multimedia-commons-expansion-pack/)
describes an ID-to-MD5 index; the [YFCC core dataset page](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)
states that access is handled through Yahoo Webscope and, when retrieved on
2026-10-18, said Webscope was not currently offering access. That page-level
availability statement must be rechecked before any acquisition; it is not a
claim that access can never return. Neither index is present locally in this
workspace, and no official VidOR/YFCC crosswalk was joined to the 490 candidates.

## Decision and limits

Do not run another GPU similarity screen, infer independence from unique IDs or
paths, or build a speculative lineage parser without the missing source
metadata. The existing 490-video pHash result remains a coarse visual-overlap
screen only. The next defensible lineage check would require an independently
obtained, provenance-pinned mapping from the exact candidate IDs to official
VidOR/YFCC source hashes or stronger recording/event identifiers. Even exact
YFCC hash uniqueness would address identical media identity only, not shared
events, re-uploads, or causal independence.

Direct generated-claim faithfulness remains **0/3** human-reviewed claims,
**0/15** verified player-clock/decoder-PTS frame pairs, **0/3** prediction-link
judgments, and **0/3** claim-linked scores. This audit changes no direct
faithfulness denominator and has no GPU/model result. Human review still needs
the independently pinned trust root, assigned role/key roster, signed
review-only receipt, and witnessed unopened handoff.
