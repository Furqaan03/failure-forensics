"""Feedback-to-eval loop: human-flagged failures become growing eval cases."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

EVAL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_dataset.jsonl"


class EvalCase(BaseModel):
    id: str
    trace_id: str
    failing_step: str
    bad_output: str
    corrected_output: str
    failure_category: str
    created_at: str


def add_eval_case(trace_id: str, failing_step: str, bad_output: str, corrected_output: str, failure_category: str) -> EvalCase:
    case = EvalCase(
        id=str(uuid.uuid4()),
        trace_id=trace_id,
        failing_step=failing_step,
        bad_output=bad_output,
        corrected_output=corrected_output,
        failure_category=failure_category,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_PATH.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json() + "\n")
    return case


def load_eval_cases() -> list[EvalCase]:
    if not EVAL_PATH.exists():
        return []
    return [EvalCase(**json.loads(line)) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure_analytics() -> dict:
    cases = load_eval_cases()
    if not cases:
        return {"total_cases": 0, "by_category": {}, "by_step": {}}
    by_category: dict[str, int] = {}
    by_step: dict[str, int] = {}
    for c in cases:
        by_category[c.failure_category] = by_category.get(c.failure_category, 0) + 1
        by_step[c.failing_step] = by_step.get(c.failing_step, 0) + 1
    return {"total_cases": len(cases), "by_category": by_category, "by_step": by_step}
