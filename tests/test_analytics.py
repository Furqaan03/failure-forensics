from pathlib import Path

import src.analyzer.eval_dataset as ed


def test_analytics_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "EVAL_PATH", tmp_path / "eval.jsonl")
    assert ed.failure_analytics()["total_cases"] == 0


def test_add_and_aggregate(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "EVAL_PATH", tmp_path / "eval.jsonl")
    ed.add_eval_case("t1", "extraction", "bad", "good", "extraction_hallucination")
    ed.add_eval_case("t2", "extraction", "bad2", "good2", "extraction_hallucination")
    ed.add_eval_case("t3", "classification", "bad3", "good3", "misclassification")

    stats = ed.failure_analytics()
    assert stats["total_cases"] == 3
    assert stats["by_category"]["extraction_hallucination"] == 2
    assert stats["by_step"]["extraction"] == 2
    assert stats["by_step"]["classification"] == 1
