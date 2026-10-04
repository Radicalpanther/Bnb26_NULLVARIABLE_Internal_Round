# app.py - FastAPI /diagnose and /health service
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Body, HTTPException
from pydantic import BaseModel, Field

from features import build_features
from explain import explain_rows, write_explanation_with_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Black Box Agent Debugger API", version="1.0.0")

# Load saved model bundle on startup
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
bundle = None

def load_bundle():
    global bundle
    if bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")
        bundle = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model bundle: {bundle.get('name')} with {len(bundle.get('features', []))} features")
    return bundle

try:
    load_bundle()
except Exception as e:
    logger.warning(f"Initial model bundle loading deferred: {e}")


def score_steps(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Compute root-cause probability score for each step using the classifier."""
    features = bundle["features"]
    model = bundle["model"]
    X = df[features]
    
    # RandomForestClassifier or LGBMClassifier binary classification probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return probs[:, 0]
        return probs.ravel()
    elif hasattr(model, "predict"):
        return model.predict(X)
    else:
        raise ValueError("Model does not support predict or predict_proba")


class StepPayload(BaseModel):
    step_index: int
    step_type: str
    tool_name: Optional[str] = "none"
    input_text: Optional[str] = ""
    output_text: Optional[str] = ""
    latency_ms: Optional[float] = 0.0
    error_flag: Optional[int] = 0
    retry_count: Optional[int] = 0
    output_length: Optional[int] = None
    state_snapshot: Optional[Union[str, Dict[str, Any]]] = "{}"
    depends_on: Optional[Union[str, int, float, List[int]]] = ""
    run_id: Optional[str] = None


class DiagnoseRequest(BaseModel):
    run_id: Optional[str] = None
    steps: List[Dict[str, Any]]


class RankingItem(BaseModel):
    step_index: int
    step_type: str
    score: float
    evidence: List[str]


class DiagnoseResponse(BaseModel):
    run_id: str
    ranking: List[RankingItem]
    explanation: str
    explanation_source: str


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_run(
    payload: Union[DiagnoseRequest, List[Dict[str, Any]]] = Body(...),
    llm: bool = Query(True, description="Whether to query Gemma 2B for 2-sentence explanation")
):
    """
    Diagnose a failed agent run by ranking steps by root-cause likelihood and providing SHAP evidence.
    Query param ?llm=false skips calling Gemma and returns the fallback explanation template.
    """
    b = load_bundle()

    # Normalize request structure
    if isinstance(payload, list):
        steps_raw = payload
        run_id = steps_raw[0].get("run_id", "unknown_run") if steps_raw else "unknown_run"
    else:
        steps_raw = payload.steps
        run_id = payload.run_id or (steps_raw[0].get("run_id", "unknown_run") if steps_raw else "unknown_run")

    if not steps_raw:
        raise HTTPException(status_code=400, detail="Empty steps list provided")

    # Format steps as a DataFrame suitable for build_features
    records = []
    for idx, s in enumerate(steps_raw):
        s_dict = dict(s)
        step_idx = s_dict.get("step_index", idx)
        out_txt = str(s_dict.get("output_text", "") or "")
        out_len = s_dict.get("output_length")
        if out_len is None or pd.isna(out_len):
            out_len = len(out_txt)
            
        snap = s_dict.get("state_snapshot", "{}")
        if isinstance(snap, dict):
            snap_str = json.dumps(snap)
        else:
            snap_str = str(snap or "{}")

        deps = s_dict.get("depends_on", "")
        if isinstance(deps, list):
            deps_str = ",".join(str(d) for d in deps)
        elif pd.isna(deps) or deps is None:
            deps_str = ""
        else:
            deps_str = str(deps)

        records.append({
            "run_id": s_dict.get("run_id", run_id),
            "step_index": int(step_idx),
            "step_type": str(s_dict.get("step_type", "plan")),
            "tool_name": str(s_dict.get("tool_name", "none") or "none"),
            "input_text": str(s_dict.get("input_text", "") or ""),
            "output_text": out_txt,
            "latency_ms": float(s_dict.get("latency_ms", 0.0) or 0.0),
            "error_flag": int(s_dict.get("error_flag", 0) or 0),
            "retry_count": int(s_dict.get("retry_count", 0) or 0),
            "output_length": int(out_len),
            "state_snapshot": snap_str,
            "depends_on": deps_str,
        })

    steps_df = pd.DataFrame(records)
    
    # Feature extraction
    df_feat = build_features(steps_df, b["stats"])
    
    # Predict root-cause scores
    scores = score_steps(b, df_feat)
    
    # Compute SHAP evidence for all steps
    all_evidence = explain_rows(b, df_feat)
    
    # Build list of ranked steps
    ranked_steps = []
    for i in range(len(df_feat)):
        ranked_steps.append({
            "step_index": int(df_feat["step_index"].iloc[i]),
            "step_type": str(df_feat["step_type"].iloc[i]),
            "score": round(float(scores[i]), 4),
            "evidence": all_evidence[i]
        })

    # Sort descending by score
    ranked_steps.sort(key=lambda x: x["score"], reverse=True)

    # Use SHAP evidence for top 5 steps only; clear evidence for the rest
    for i, item in enumerate(ranked_steps):
        if i >= 5:
            item["evidence"] = []

    # Generate explanation for top diagnosed root cause step
    top_step = ranked_steps[0]
    explanation, explanation_source = write_explanation_with_source(
        step_type=top_step["step_type"],
        step_index=top_step["step_index"],
        evidence=top_step["evidence"],
        use_llm=llm,
        output_snippet=str(df_feat.loc[df_feat["step_index"] == top_step["step_index"], "output_text"].iloc[0]),
        rank_one=True,
    )

    return {
        "run_id": run_id,
        "ranking": ranked_steps,
        "explanation": explanation,
        "explanation_source": explanation_source,
    }
