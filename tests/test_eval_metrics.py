"""No-LLM tests for the eval harness's statistics and reference retrievers.

These guard the numbers the report is built from. A silent break here would not
fail any pipeline test — it would just publish wrong metrics, which is the worst
failure mode an eval harness has.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from retrievers import BM25Retriever  # noqa: E402
from scoring import (  # noqa: E402
    citation_hit,
    cluster_bootstrap_ci,
    confusion_matrix,
    wilson_interval,
)


# ---------------- confusion matrix ----------------


def test_confusion_matrix_counts_and_scores():
    pairs = [("A", "A"), ("A", "A"), ("A", "B"), ("B", "B"), ("C", "B")]
    cm = confusion_matrix(pairs)

    assert cm["accuracy"] == 0.6
    assert cm["n"] == 5
    assert cm["matrix"]["A"]["B"] == 1
    assert cm["per_class"]["A"] == {
        "support": 3, "predicted": 2, "tp": 2, "fp": 0, "fn": 1,
        "precision": 1.0, "recall": 0.6667, "f1": 0.8,
    }
    assert cm["per_class"]["B"]["fp"] == 2


def test_confusion_matrix_flags_a_never_correct_class():
    """The keyword-sink signature: predicted often, never right.

    Accuracy cannot express this; precision 0 with support 0 can.
    """
    pairs = [("A", "Sink"), ("B", "Sink"), ("A", "A")]
    cm = confusion_matrix(pairs)

    sink = cm["per_class"]["Sink"]
    assert sink["predicted"] == 2 and sink["tp"] == 0
    assert sink["precision"] == 0.0
    assert sink["support"] == 0
    assert sink["recall"] is None  # undefined: nothing truly belongs to it
    # F1 must still be 0, not None, or a never-correct class vanishes from macro F1.
    assert sink["f1"] == 0.0


def test_confusion_matrix_empty():
    cm = confusion_matrix([])
    assert cm["n"] == 0 and cm["accuracy"] is None


# ---------------- intervals ----------------


def test_wilson_does_not_collapse_at_the_boundary():
    """The reason Wilson is used instead of Wald: 29/29 is not certainty."""
    low, high = wilson_interval(29, 29)
    assert high == 1.0
    assert low < 0.9  # Wald would report exactly [1.0, 1.0]

    assert wilson_interval(18, 26) == (0.5001, 0.835)
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(0, 0) is None


def test_cluster_bootstrap_is_seeded_and_brackets_the_mean():
    items = [{"v": 0.0, "doc": "shared"} for _ in range(3)]
    items += [{"v": 1.0, "doc": f"d{i}"} for i in range(3)]

    first = cluster_bootstrap_ci(items, "v", "doc")
    assert first == cluster_bootstrap_ci(items, "v", "doc")  # reproducible
    assert first["n_items"] == 6
    assert first["n_clusters"] == 4  # 3 correlated cases collapse into one cluster
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_cluster_bootstrap_widens_when_cases_are_correlated():
    """Correlated cases carry less information than independent ones.

    Treating them as independent would understate the interval, which is exactly
    the mistake this function exists to avoid.
    """
    # Same 12 observations, same overall mean of 0.5. In the correlated set the
    # outcome is a property of the document — one fails wholesale, one passes —
    # which is how the refi cluster actually behaves.
    correlated = [{"v": float(i // 6), "doc": f"doc{i // 6}"} for i in range(12)]
    independent = [{"v": float(i % 2), "doc": f"d{i}"} for i in range(12)]

    c = cluster_bootstrap_ci(correlated, "v", "doc")
    i = cluster_bootstrap_ci(independent, "v", "doc")
    assert c["mean"] == i["mean"] == 0.5
    assert c["n_clusters"] == 2 and i["n_clusters"] == 12
    assert (c["ci_high"] - c["ci_low"]) > (i["ci_high"] - i["ci_low"])


def test_cluster_bootstrap_refuses_a_single_cluster():
    """One cluster means no between-cluster variance to resample.

    The naive result is a zero-width interval — narrow for the worst possible
    reason. It must be reported as unavailable, not as certainty.
    """
    one = [{"v": float(i % 2), "doc": "same"} for i in range(12)]
    out = cluster_bootstrap_ci(one, "v", "doc")

    assert out["n_clusters"] == 1
    assert out["ci_low"] is None and out["ci_high"] is None
    assert "not estimable" in out["note"]


def test_cluster_bootstrap_ignores_missing_values():
    assert cluster_bootstrap_ci([{"v": None, "doc": "a"}], "v", "doc") is None


# ---------------- citation faithfulness ----------------


def test_citation_hit_matches_document_and_page():
    cits = [{"filename": "cd_sample.pdf", "page_start": 2, "page_end": 3}]

    assert citation_hit(cits, "cd_sample.pdf", 2) is True
    assert citation_hit(cits, "closing_disclosure/cd_sample.pdf", 3) is True  # path-qualified
    assert citation_hit(cits, "cd_sample.pdf", 9) is False
    assert citation_hit(cits, "other.pdf", 2) is False


def test_citation_hit_is_unscoreable_without_a_page_label():
    """Bundle cases carry gold_page 0; they must be excluded, not counted wrong."""
    cits = [{"filename": "cd.pdf", "page_start": 1, "page_end": 1}]
    assert citation_hit(cits, "cd.pdf", 0) is None


# ---------------- BM25 baseline ----------------


@dataclass
class _Chunk:
    text: str
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int


def _chunks(*pairs):
    return [
        (name, _Chunk(text, f"c{i}", "d0", "Closing Disclosure", i, 0, 1))
        for i, (text, name) in enumerate(pairs)
    ]


def test_bm25_discounts_a_term_present_in_every_document():
    """The idf property that also explains the classifier's keyword sink."""
    corpus = _chunks(
        ("mortgage the loan amount is $162,000", "cd.pdf"),
        ("mortgage employee gross pay 7500", "paystub.pdf"),
        ("mortgage appraised value subject property", "appraisal.pdf"),
        ("mortgage mortgage mortgage mortgage", "spam.pdf"),
    )
    r = BM25Retriever(corpus)

    assert r.idf["loan"] / r.idf["mortgage"] > 10
    top = r.retrieve("mortgage loan amount", top_k=4)
    assert top[0].metadata["filename"] == "cd.pdf"


def test_bm25_term_frequency_saturates():
    r = BM25Retriever(_chunks(("loan", "one.pdf"), (" ".join(["loan"] * 50), "fifty.pdf")))
    scores = {n.metadata["filename"]: n.score for n in r.retrieve("loan", top_k=2)}
    assert scores["fifty.pdf"] / scores["one.pdf"] < 3  # not 50x


def test_bm25_metadata_matches_the_dense_path():
    """Pages are 1-indexed downstream; a mismatch would silently break citations."""
    r = BM25Retriever(_chunks(("loan amount", "cd.pdf")))
    node = r.retrieve("loan amount")[0]
    assert node.metadata["page_start"] == 1 and node.metadata["page_end"] == 2
    assert node.metadata["filename"] == "cd.pdf"


def test_bm25_handles_no_match_and_empty_corpus():
    r = BM25Retriever(_chunks(("loan amount", "cd.pdf")))
    assert r.retrieve("zzzznonexistent") == []
    assert BM25Retriever([]).retrieve("anything") == []


# ---------------- CLI override parsing ----------------


def test_parse_overrides_types_values_from_the_dataclass():
    """A bool passed as the string "false" would evaluate truthy and silently
    invalidate an ablation arm, so types come from the config field."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "evals" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)

    parsed = run_eval._parse_overrides(["use_reranker=false", "chunk_size=250", "temperature=0.7"])
    assert parsed == {"use_reranker": False, "chunk_size": 250, "temperature": 0.7}
    assert run_eval._parse_overrides([]) == {}

    for bad in (["nosuchfield=1"], ["noequals"]):
        try:
            run_eval._parse_overrides(bad)
        except SystemExit:
            continue
        raise AssertionError(f"expected rejection of {bad}")
