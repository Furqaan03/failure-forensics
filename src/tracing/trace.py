"""Trace and Span models + a context manager that instruments each pipeline step."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

TRACES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "traces"
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "traces.db"


class Span(BaseModel):
    step_name: str
    input_serialized: str
    output_serialized: str = ""
    llm_prompt: str = ""
    llm_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    confidence: int | None = None  # model's self-reported confidence 1-5
    error: str | None = None


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spans: list[Span] = Field(default_factory=list)
    final_output: str = ""
    status: str = "success"  # success | failure | degraded

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    def get_span(self, step_name: str) -> Span | None:
        return next((s for s in self.spans if s.step_name == step_name), None)


@contextmanager
def span(trace: Trace, step_name: str, step_input: object) -> Iterator[Span]:
    """Wraps a pipeline step: auto-captures input, output, latency, and errors.
    Instrumenting a new step is one `with span(...)` line."""
    s = Span(step_name=step_name, input_serialized=_serialize(step_input))
    start = time.perf_counter()
    try:
        yield s
    except Exception as exc:  # noqa: BLE001 — recorded on the span, re-raised after
        s.error = str(exc)
        trace.status = "failure"
        s.latency_ms = (time.perf_counter() - start) * 1000
        trace.add_span(s)
        raise
    s.latency_ms = (time.perf_counter() - start) * 1000
    trace.add_span(s)


def _serialize(obj: object) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, default=str)
    return str(obj)


def persist_trace(trace: Trace) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    (TRACES_DIR / f"{trace.trace_id}.json").write_text(trace.model_dump_json(indent=2), encoding="utf-8")

    conn = _conn()
    final_score = _mean_confidence(trace)
    conn.execute(
        "INSERT OR REPLACE INTO traces (trace_id, timestamp, status, final_score) VALUES (?, ?, ?, ?)",
        (trace.trace_id, trace.timestamp, trace.status, final_score),
    )
    conn.commit()
    conn.close()


def load_trace(trace_id: str) -> Trace:
    return Trace(**json.loads((TRACES_DIR / f"{trace_id}.json").read_text(encoding="utf-8")))


def _mean_confidence(trace: Trace) -> float | None:
    scores = [s.confidence for s in trace.spans if s.confidence is not None]
    return sum(scores) / len(scores) if scores else None


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY, timestamp TEXT, status TEXT, final_score REAL
        )"""
    )
    return conn


def list_traces(limit: int = 100) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT trace_id, timestamp, status, final_score FROM traces ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"trace_id": r[0], "timestamp": r[1], "status": r[2], "final_score": r[3]} for r in rows]
