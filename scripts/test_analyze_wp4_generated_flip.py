"""CPU checks for the WP4 analysis helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_wp4_generated_flip as a


def rec(base, **arms):
    return {"baseline_letter": base, "baseline_correct": True,
            "generations": [{"arm": k, "letter": v} for k, v in arms.items()]}


def test_flip_counts_unparseable_as_flip():
    r = rec("A", cited="A", sham=None)
    assert a.flip(r, "cited") is False
    assert a.flip(r, "sham") is True


def test_mcnemar_exact_known_values():
    assert a.mcnemar_exact(0, 0) == 1.0
    assert abs(a.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert a.mcnemar_exact(5, 5) == 1.0


def test_contrast_and_decision_gate(tmp_path):
    recs = []
    for i in range(40):
        recs.append(rec("A", cited="B" if i < 20 else "A", sham="A", annotation="A", allvideo="B"))
    c = a.contrast(recs, "cited", "sham")
    assert c["only_a"] == 20 and c["only_b"] == 0 and c["diff"] == 0.5
    assert c["mcnemar_exact_p"] < 1e-4


if __name__ == "__main__":
    import tempfile
    test_flip_counts_unparseable_as_flip(); test_mcnemar_exact_known_values(); test_contrast_and_decision_gate(Path(tempfile.mkdtemp()))
    print("tests ok")
