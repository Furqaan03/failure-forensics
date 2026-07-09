"""FastAPI: run the pipeline, list/inspect traces, flag failures, get analytics."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.analyzer.eval_dataset import add_eval_case, failure_analytics
from src.analyzer.root_cause import diagnose
from src.pipeline.steps import run_pipeline
from src.tracing.trace import list_traces, load_trace

load_dotenv()

app = FastAPI(title="Failure Forensics Tool")


class ProcessRequest(BaseModel):
    document: str


@app.post("/v1/process")
def process(req: ProcessRequest) -> dict:
    trace = run_pipeline(req.document)
    return {"trace_id": trace.trace_id, "status": trace.status, "final_output": trace.final_output}


@app.get("/v1/traces")
def get_traces() -> list[dict]:
    return list_traces()


@app.get("/v1/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    try:
        return load_trace(trace_id).model_dump()
    except FileNotFoundError:
        raise HTTPException(404, "Trace not found")


@app.post("/v1/traces/{trace_id}/diagnose")
def diagnose_trace(trace_id: str) -> dict:
    trace = load_trace(trace_id)
    return diagnose(trace).model_dump()


class FlagRequest(BaseModel):
    failing_step: str
    bad_output: str
    corrected_output: str
    failure_category: str


@app.post("/v1/traces/{trace_id}/flag")
def flag_trace(trace_id: str, req: FlagRequest) -> dict:
    case = add_eval_case(trace_id, req.failing_step, req.bad_output, req.corrected_output, req.failure_category)
    return {"eval_case_id": case.id}


@app.get("/v1/analytics")
def analytics() -> dict:
    return failure_analytics()
