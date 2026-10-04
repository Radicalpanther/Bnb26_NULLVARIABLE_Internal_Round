# explain.py - SHAP explanation extraction and LLM/fallback sentence generation
import json
import logging
import re
from decimal import Decimal, InvalidOperation
import numpy as np
import requests
import shap

logger = logging.getLogger("explain")

# Mapping of feature names to human-readable explanation templates
FEATURE_PHRASES = {
    "lat_z": "{v:.1f} standard deviations above normal",
    "err_words": "the output contained error wording",
    "repeat_count": "the same call was made earlier",
    "dep_flags": "later steps that used its output look suspicious",
    "anomaly_score": "the step looks unusual compared with successful runs",
    "len_z": "{v:.1f} standard deviations above normal",
    "error_flag": "an error flag was raised",
    "flag_any": "an error or warning condition was flagged",
    "is_first_flag": "first step in the run to show an error or warning",
    "prior_flags": "preceded by {v:.0f} flagged steps",
    "is_empty": "the output was empty",
    "overlap_next_in": "the next step did not reuse its output",
    "answer_matches_state": "the answer matched values from the state",
    "prev_err_words": "preceding step contained error wording",
    "next_err_words": "subsequent step contained error wording",
    "prev_lat_z": "preceding step had abnormal latency ({v:.1f} std)",
    "next_lat_z": "subsequent step had abnormal latency ({v:.1f} std)",
    "prev_len_z": "preceding step had abnormal output length ({v:.1f} std)",
    "next_len_z": "subsequent step had abnormal output length ({v:.1f} std)",
    "prev_error_flag": "preceding step raised an error flag",
    "next_error_flag": "subsequent step raised an error flag",
    "n_deps": "step depends on {v:.0f} prior steps",
    "n_dependents": "output is used by {v:.0f} downstream steps",
    "n_extracted": "{v:.0f} state variables were extracted",
    "rel_pos": "step occurred at relative position {v:.2f}",
    "steps_left": "{v:.0f} steps remained in execution",
    "retry_count": "step was retried {v:.0f} times",
    "output_length": "output length was {v:.0f} characters",
    "latency_ms": "latency was {v:.1f} ms",
}

EVIDENCE_WHITELIST = {
    "lat_z",
    "len_z",
    "err_words",
    "error_flag",
    "repeat_count",
    "retry_count",
    "dep_flags",
    "is_first_flag",
    "anomaly_score",
    "overlap_next_in",
    "answer_matches_state",
    "input_covers_goal",
}

def format_evidence(feature_name: str, value: float) -> str:
    """Format a single feature name and value into a human-readable evidence phrase."""
    if feature_name in FEATURE_PHRASES:
        template = FEATURE_PHRASES[feature_name]
        try:
            return template.format(v=value)
        except Exception:
            return f"{feature_name} had value {value}"
    
    # Generic phrase for unknown / one-hot features
    clean_name = feature_name.replace("st_", "step type ").replace("tool_", "tool ").replace("task_", "task ").replace("_", " ")
    if isinstance(value, (int, float, np.number)):
        if abs(value - round(value)) < 1e-5:
            return f"{clean_name} was {int(round(value))}"
        return f"{clean_name} was {value:.2f}"
    return f"{clean_name} was {value}"


def _extract_positive_shap_values(sv) -> np.ndarray:
    """Handle different SHAP values output formats for tree models and extract positive class contributions."""
    if hasattr(sv, "values"):  # shap.Explanation object
        sv = sv.values

    if isinstance(sv, list):
        if len(sv) == 2:
            return np.array(sv[1])
        return np.array(sv[-1])

    arr = np.array(sv)
    if arr.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        if arr.shape[2] == 2:
            return arr[:, :, 1]
        return arr[:, :, -1]
    elif arr.ndim == 2:
        return arr
    elif arr.ndim == 1:
        return arr.reshape(1, -1)
    
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


def explain_rows(bundle: dict, df) -> list[list[str]]:
    """
    Use shap.TreeExplainer on the saved model to find top 3 positive contributing features per row.
    Returns a list of evidence phrase lists, one list per row in df.
    """
    features = bundle["features"]
    model = bundle["model"]
    X = df[features]
    
    if len(X) == 0:
        return []

    explainer = shap.TreeExplainer(model)
    raw_sv = explainer.shap_values(X)
    pos_sv = _extract_positive_shap_values(raw_sv)
    last_in_run = df.groupby("run_id", sort=False).cumcount(ascending=False).eq(0)

    all_evidence = []
    for i in range(len(X)):
        row_sv = pos_sv[i]
        row_x = X.iloc[i]
        step_type = df["step_type"].iloc[i]
        
        # Filter features before ranking so disallowed features cannot displace evidence.
        candidate_indices = []
        for idx, feature_name in enumerate(features):
            if feature_name not in EVIDENCE_WHITELIST:
                continue
            value = float(row_x.iloc[idx])
            if feature_name in {"lat_z", "len_z"} and value < 1:
                continue
            if feature_name == "overlap_next_in" and (
                value >= 0.02 or last_in_run.iloc[i]
            ):
                continue
            if feature_name == "answer_matches_state" and step_type != "write_answer":
                continue
            if feature_name == "input_covers_goal" and (
                step_type != "call_tool" or value >= 0.5
            ):
                continue
            candidate_indices.append(idx)
        sorted_indices = sorted(candidate_indices, key=lambda idx: row_sv[idx], reverse=True)
        
        # Select top 3 features that pushed the score UP (shap value > 0)
        pos_indices = [idx for idx in sorted_indices if row_sv[idx] > 0][:3]
        if not pos_indices:
            # Fallback to top feature(s) if none strictly positive
            pos_indices = sorted_indices[:3].tolist()
            
        row_evidence = []
        for idx in pos_indices:
            feat_name = features[idx]
            val = row_x.iloc[idx]
            row_evidence.append(format_evidence(feat_name, val))
            
        all_evidence.append(row_evidence)
        
    return all_evidence


def fallback_explanation(
    step_type: str, step_index: int, evidence: list[str], rank_one: bool = False
) -> str:
    """Generate a reliable fallback explanation template."""
    if rank_one and len(evidence) >= 2:
        facts = "; ".join(evidence)
        return (
            f"Step {step_index} ({step_type}) was identified as the root cause because "
            f"it ranked highest because of the combination of signals: {facts}. "
            f"Replaying execution from step {step_index} with corrected parameters or tool choice is recommended."
        )
    if len(evidence) < 2:
        reason = "no single strong signal; its score comes from a combination of weaker signals"
    else:
        reason = "; ".join(evidence)
    return (f"Step {step_index} ({step_type}) was identified as the root cause because {reason}. "
            f"Replaying execution from step {step_index} is recommended.")


def _valid_llm_reply(reply: str, step_type: str, step_index: int, evidence: list[str], output_snippet: str) -> bool:
    if not reply.startswith(f"Step {step_index} ({step_type}"):
        return False

    sentences = re.split(r"(?<=[.!?])\s+", reply)
    if len(sentences) != 2 or not re.search(r"[.!?]$", reply):
        return False

    allowed_text = " ".join([output_snippet, *evidence, str(step_index)])
    allowed_numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?", allowed_text))
    for number in re.findall(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?", reply):
        try:
            normalized = Decimal(number.replace(",", "")).normalize()
            if not any(Decimal(item.replace(",", "")).normalize() == normalized for item in allowed_numbers):
                return False
        except InvalidOperation:
            return False

    return True


def write_explanation_with_source(
    step_type: str,
    step_index: int,
    evidence: list[str],
    use_llm: bool = True,
    output_snippet: str = "",
    rank_one: bool = False,
) -> tuple[str, str]:
    """
    Call Ollama Gemma 2B to synthesize a 2-sentence explanation from facts.
    Falls back to a fixed template on error, timeout, or if use_llm is False.
    """
    if not use_llm or not evidence:
        return fallback_explanation(step_type, step_index, evidence, rank_one), "template"

    facts_text = "\n".join(f"- {fact}" for fact in evidence)
    prompt = (
        f"Step {step_index} ({step_type}) of an AI agent execution failed. "
        f"Here is the diagnosed evidence for why this step caused the failure:\n"
        f"{facts_text}\n\n"
        f"Task: Write an explanation in exactly two plain sentences explaining why this step caused the failure using only the facts above."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:2b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            reply = data.get("response", "").strip()
            if _valid_llm_reply(reply, step_type, step_index, evidence, output_snippet):
                return reply, "llm"
    except Exception as e:
        logger.warning(f"Ollama generation failed or timed out: {e}")

    return fallback_explanation(step_type, step_index, evidence, rank_one), "template"


def write_explanation(
    step_type: str,
    step_index: int,
    evidence: list[str],
    use_llm: bool = True,
) -> str:
    """Return an explanation while retaining the original string API."""
    explanation, _ = write_explanation_with_source(
        step_type, step_index, evidence, use_llm
    )
    return explanation
