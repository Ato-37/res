# V6 reviewer export disclosure repair (2026-10-18)

## Purpose

An independent audit found that the local reviewer page described the v6
review JSON's top-level fields but omitted `export_disclosure_acknowledged`
from that explicit list. The checkbox and exported value already existed; the
pre-review inventory of recorded fields was incomplete. Repair the disclosure
and ensure future page changes remain synchronized with the frozen v6 export
contract.

## Implementation and checks

- Added the omitted field to the reviewer-facing top-level export list.
- Added a regression assertion that compares the policy's complete top-level
  export field set against that specific disclosure paragraph, rather than
  checking only that field names occur somewhere on the page.
- Focused renderer suite: **9/9 passed**.
- Full repository suite: **1,408/1,408 passed** in 33.6 seconds.
- `py_compile` and scoped `git diff --check`: passed.
- Re-rendered the existing private, blinded three-video review page under the
  repaired disclosure and rebuilt its isolated local package. The package
  contains exactly one HTML page and three opaque media files; directory mode
  is `0700`, all file modes are `0600`, packaged page bytes equal the rendered
  page, and packaged media bytes match the custody-verified source copies.
  The renderer and packager both retain `execution_authorization=not_granted`.

## Interpretation and limits

This repairs a reviewer-consent disclosure defect and provides a fresh
candidate handoff. It does not establish that a person opened or reviewed the
materials, authenticate reviewer identity, validate the claims, or provide
browser-to-decoder frame correspondence. No GPU inference or faithfulness
score was run. Rights status remains unverified; the packet and all
video-derived review data remain local.

Direct claim-linked video faithfulness remains **0/3 independently reviewed,
0/3 scored, with no `ΔF` estimate, interval, or test**. Verified
browser-to-decoder correspondences remain **0/15**. The package-level
trust-root/witness gate and the separate human review and frame-capture steps
remain unresolved. The primary internal-intervention treatment and production
six-condition scorer are also not frozen/implemented, so a GPU faithfulness
run is not yet justified. Repeating the existing generic real-video GPU smoke
would add no evidence.

## Next action

Have an independent human reviewer use the repaired offline page when available,
then audit the local export without exposing review content. In parallel,
freeze one outcome-independent internal treatment and implement the production
six-condition path with immutable input, label, processor, and layer bindings;
only then run the bounded real-video GPU pilot after the review and timing gates
pass.
