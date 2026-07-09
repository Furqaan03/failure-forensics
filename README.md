# Failure Forensics Tool for AI Pipelines

An observability layer for multi-step AI pipelines. Every pipeline run is fully
traced, span by span. When the final output is bad, a backward analyzer walks
the trace in reverse, uses an LLM-as-judge to score each step's output quality
against its input, and pinpoints the exact step where things first went wrong —
turning hours of manual "which step broke?" debugging into a one-click
diagnosis. Flagged failures feed a growing eval dataset. A mini
LangSmith/Braintrust.

## Why this exists

When a 4-step AI pipeline produces garbage, most teams have no idea which step
broke — the error could have originated in extraction and silently propagated
through classification into the summary. This answers "where did this go
wrong?" automatically.

## Architecture

```
src/llm.py                    thin OpenAI wrapper returning text + raw prompt/response
src/tracing/trace.py          Trace + Span models; span() context manager auto-captures
                               input/output/latency/errors — instrumenting a step is one line
src/pipeline/steps.py         4-step pipeline (intake -> extraction -> classification ->
                               summarization), each a typed, isolated function with
                               self-reported confidence scoring
src/analyzer/root_cause.py    backward analyzer: LLM-judge scores each step, earliest
                               below-threshold step = root cause; 5-type failure taxonomy
src/analyzer/eval_dataset.py  feedback-to-eval loop: flagged failures become eval cases
src/api/main.py               FastAPI: process, list/inspect traces, diagnose, flag, analytics
src/dashboard.py              Streamlit trace explorer with color-coded spans
```

## Design decisions

- **Every step is a typed, isolated function with Pydantic I/O.** The guide is
  explicit that spaghetti steps make tracing meaningless — clean typed
  boundaries are what make the span input/output diffs interpretable.
- **The `span()` context manager auto-instruments.** Wrapping a step in
  `with span(trace, name, input) as s:` captures latency, serialized I/O, the
  LLM prompt/response, token counts, and any exception — so adding a new step
  costs one line, and errors are recorded on the span *and* re-raised rather
  than swallowed.
- **Root cause is the *earliest* low-quality step, not the last.** Because
  errors propagate forward (a hallucinated entity in extraction poisons the
  summary), walking in execution order and stopping at the first step below the
  quality threshold correctly attributes blame to the origin, not the symptom.
- **Confidence is self-reported per step and stored on the span.** Low-confidence
  spans are the primary suspects when tracing backward, and they surface as
  yellow in the explorer before you even run a full diagnosis.
- **Sample docs carry deliberate failure modes** (contract with no dates,
  mixed-currency invoice, report-vs-correspondence ambiguity) so there are real
  failures to trace, not just happy paths.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env      # fill in OPENAI_API_KEY
uvicorn src.api.main:app --reload      # API on :8000
streamlit run src/dashboard.py         # explorer UI
```

## Example flow

```bash
curl -X POST localhost:8000/v1/process -H "Content-Type: application/json" \
  -d '{"document": "SERVICE AGREEMENT between Party A and Party B. Either party may terminate with notice."}'
# -> {"trace_id": "...", "status": "success", "final_output": "..."}

curl -X POST localhost:8000/v1/traces/<trace_id>/diagnose
# -> {"root_cause_step": "extraction", "failure_type": "extraction_hallucination", "evidence": "..."}

curl -X POST localhost:8000/v1/traces/<trace_id>/flag -H "Content-Type: application/json" \
  -d '{"failing_step": "extraction", "bad_output": "...", "corrected_output": "...", "failure_category": "extraction_hallucination"}'

curl localhost:8000/v1/analytics
```

## Tests

```bash
pytest tests/ -v
```

Tracing (span capture, error recording, trace status) and the feedback-to-eval
aggregation are covered offline — no API key required.

## Docker

```bash
docker build -t failure-forensics .
docker run -p 8000:8000 --env-file .env failure-forensics
```

## Status

Phases 1-5 complete (pipeline, tracing, backward analyzer, explorer, feedback
loop). The Streamlit explorer stands in for the Phase 4 React trace view; the
Phase 5 regression re-run of accumulated eval cases is scaffolded via the eval
dataset but not yet wired to an automated scheduler.
