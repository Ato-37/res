#!/usr/bin/env python3
"""Apply the prespecified Stage-2 rule of experiments/2026-10-23_evidence_focusing_steering_plan.md to a dose-response summary JSON.

Confirmed iff the primary contrast (keepAnn - keepRand) has t, sign-flip and Wilcoxon p < 0.05 AND mean >= delta_min (0.3 nat).
Otherwise reports TOST equivalence at +/-delta_min (needs p < 0.05) or 'inconclusive'. Prints a short summary; writes nothing.
"""
import json
import sys

DELTA_MIN = 0.3


def verdict(row: dict, delta_min: float = DELTA_MIN) -> str:
    ps = (row["p_two_sided_t"], row["p_sign_flip"], row["p_wilcoxon"])
    if all(p < 0.05 for p in ps) and row["mean_nats"] >= delta_min:
        return "confirmed"
    if row["tost_p"] < 0.05:
        return "equivalent_within_delta_min"
    return "not_confirmed_inconclusive"


def main() -> int:
    summary = json.load(open(sys.argv[1]))
    row = summary["doses"][0]
    print(json.dumps({"n": row["n"], "mean": row["mean_nats"], "ci95": row["ci95_t"], "p_t": row["p_two_sided_t"],
                      "p_sign_flip": row["p_sign_flip"], "p_wilcoxon": row.get("p_wilcoxon"), "tost_p_0.025": row["tost_p"],
                      "verdict": verdict(row)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
