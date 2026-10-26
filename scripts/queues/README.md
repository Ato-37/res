# scripts/queues/

Sanitized copies of the shell queues that drove the 2026-10-23..26 GPU runs (they were written to the session scratchpad and are kept here for provenance).
Each stage runs smoke (n=3) -> full -> analysis and never stops the queue on a failure; cohorts and manifests are named in the stage lines.
Not meant to be re-run blindly: tags and receipt names are fixed per run, and runners refuse to overwrite an existing output.
