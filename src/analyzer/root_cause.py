"""Backward trace analyzer: walks spans in reverse, scores each step's output
quality given its input, and identifies the first step where quality drops."""
from __future__ import annotations

import json

from pydantic import BaseModel

from src.llm import call_llm
from src.tracing.trace import Trace

# Failure taxonomy per the guide.
FAILURE_TYPES = [
    "extraction_hallucination",
    "misclassification",
    "propagation_error",
    "prompt_failure",
    "context_loss",
]


class StepQuality(BaseModel):
    step_name: str
    quality_score: int  # 1-5, is the output a reasonable transformation of the input?
    reasoning: str


class Diagnosis(BaseModel):
    trace_id: str
    root_cause_step: str | None
    failure_type: str | None
    evidence: str
    step_qualities: list[StepQuality]


_QUALITY_DROP_THRESHOLD = 3  # scores below this mark a suspect step


def _score_step(step_name: str, input_serialized: str, output_serialized: str) -> StepQuality:
    prompt = (
        f"Rate 1-5 whether the OUTPUT is a reasonable, faithful transformation of the INPUT "
        f"for a pipeline step called '{step_name}'. 5 = fully correct, 1 = clearly wrong/hallucinated. "
        'Respond as JSON: {"quality_score": 1-5, "reasoning": "one sentence"}.\n\n'
        f"INPUT:\n{input_serialized}\n\nOUTPUT:\n{output_serialized}"
    )
    call = call_llm(prompt)
    text = call.response.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    parsed = json.loads(text)
    return StepQuality(step_name=step_name, quality_score=int(parsed["quality_score"]), reasoning=parsed["reasoning"])


def _classify_failure(root_step: str) -> str:
    mapping = {
        "extraction": "extraction_hallucination",
        "classification": "misclassification",
        "summarization": "propagation_error",
        "intake": "prompt_failure",
    }
    return mapping.get(root_step, "prompt_failure")


def diagnose(trace: Trace) -> Diagnosis:
    """LLM-as-judge scores each step; the first (earliest) step below threshold is
    the root cause, since downstream steps inherit upstream errors."""
    qualities: list[StepQuality] = []
    for span in trace.spans:
        if not span.llm_response and span.step_name != "intake":
            continue
        qualities.append(_score_step(span.step_name, span.input_serialized, span.output_serialized))

    root_cause_step = None
    for q in qualities:  # spans are in execution order; first low score = origin
        if q.quality_score < _QUALITY_DROP_THRESHOLD:
            root_cause_step = q.step_name
            break

    failure_type = _classify_failure(root_cause_step) if root_cause_step else None
    evidence = _build_evidence(trace, root_cause_step) if root_cause_step else "No step scored below the quality threshold."

    return Diagnosis(
        trace_id=trace.trace_id,
        root_cause_step=root_cause_step,
        failure_type=failure_type,
        evidence=evidence,
        step_qualities=qualities,
    )


def _build_evidence(trace: Trace, root_step: str) -> str:
    span = trace.get_span(root_step)
    if span is None:
        return ""
    return (
        f"Step '{root_step}' received input: {span.input_serialized[:300]} "
        f"and produced output: {span.output_serialized[:300]}. "
        "This is the earliest step whose output quality dropped, so downstream "
        "steps inherited the error."
    )
