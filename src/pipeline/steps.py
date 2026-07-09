"""4-step document pipeline: intake -> extraction -> classification -> summarization.
Each step is a clean, isolated, typed function so tracing stays meaningful."""
from __future__ import annotations

import json

from pydantic import BaseModel

from src.llm import call_llm
from src.tracing.trace import Span, Trace, span

# ---- Typed intermediate data structures ----


class IntakeResult(BaseModel):
    raw_text: str


class Entity(BaseModel):
    kind: str  # name | date | amount | term
    value: str


class ExtractionResult(BaseModel):
    entities: list[Entity]
    confidence: int


class ClassificationResult(BaseModel):
    doc_type: str  # contract | invoice | report | correspondence
    confidence: int


class SummaryResult(BaseModel):
    summary: str
    confidence: int


def _parse_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    return json.loads(text)


# ---- Steps ----


def step_intake(trace: Trace, document: str) -> IntakeResult:
    with span(trace, "intake", document) as s:
        result = IntakeResult(raw_text=document.strip())
        s.output_serialized = result.model_dump_json()
    return result


def step_extraction(trace: Trace, intake: IntakeResult) -> ExtractionResult:
    with span(trace, "extraction", intake) as s:
        prompt = (
            "Extract entities from the document as JSON: "
            '{"entities": [{"kind": "name|date|amount|term", "value": "..."}], "confidence": 1-5}. '
            "Only extract entities that literally appear in the text. Do not infer.\n\n"
            f"Document:\n{intake.raw_text}"
        )
        call = call_llm(prompt)
        s.llm_prompt = call.prompt
        s.llm_response = call.response
        s.input_tokens, s.output_tokens = call.input_tokens, call.output_tokens
        parsed = _parse_json_block(call.response)
        result = ExtractionResult(
            entities=[Entity(**e) for e in parsed.get("entities", [])],
            confidence=int(parsed.get("confidence", 3)),
        )
        s.confidence = result.confidence
        s.output_serialized = result.model_dump_json()
    return result


def step_classification(trace: Trace, intake: IntakeResult, extraction: ExtractionResult) -> ClassificationResult:
    with span(trace, "classification", extraction) as s:
        prompt = (
            "Classify this document as exactly one of: contract, invoice, report, correspondence. "
            'Respond as JSON: {"doc_type": "...", "confidence": 1-5}.\n\n'
            f"Document:\n{intake.raw_text}\n\nExtracted entities: {extraction.model_dump_json()}"
        )
        call = call_llm(prompt)
        s.llm_prompt = call.prompt
        s.llm_response = call.response
        s.input_tokens, s.output_tokens = call.input_tokens, call.output_tokens
        parsed = _parse_json_block(call.response)
        result = ClassificationResult(doc_type=parsed["doc_type"], confidence=int(parsed.get("confidence", 3)))
        s.confidence = result.confidence
        s.output_serialized = result.model_dump_json()
    return result


def step_summarization(trace: Trace, intake: IntakeResult, classification: ClassificationResult) -> SummaryResult:
    with span(trace, "summarization", classification) as s:
        prompt = (
            f"Write a structured summary of this {classification.doc_type}. "
            'Respond as JSON: {"summary": "...", "confidence": 1-5}. '
            "Only use information present in the document.\n\n"
            f"Document:\n{intake.raw_text}"
        )
        call = call_llm(prompt)
        s.llm_prompt = call.prompt
        s.llm_response = call.response
        s.input_tokens, s.output_tokens = call.input_tokens, call.output_tokens
        parsed = _parse_json_block(call.response)
        result = SummaryResult(summary=parsed["summary"], confidence=int(parsed.get("confidence", 3)))
        s.confidence = result.confidence
        s.output_serialized = result.model_dump_json()
    return result


def run_pipeline(document: str) -> Trace:
    """Runs the full 4-step pipeline, producing a fully-instrumented Trace."""
    from src.tracing.trace import persist_trace

    trace = Trace()
    try:
        intake = step_intake(trace, document)
        extraction = step_extraction(trace, intake)
        classification = step_classification(trace, intake, extraction)
        summary = step_summarization(trace, intake, classification)
        trace.final_output = summary.summary
    except Exception:  # noqa: BLE001 — span() already recorded it; persist partial trace
        pass
    persist_trace(trace)
    return trace
