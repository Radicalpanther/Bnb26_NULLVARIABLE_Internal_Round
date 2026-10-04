# Black Box: A Flight Recorder for AI Agents (ML Diagnosis Service)

Black Box watches an AI agent's execution trace and, when a run fails, ranks the steps by how likely each one is to be the root cause. It gives evidence for the top suspects and supports replaying the run from a suspect step with a fix, to confirm the cause.

This repository is the **ML service**: data generation, feature engineering, models, evaluation, and the FastAPI `/diagnose` endpoint. The platform (run/event storage, replay, comparison, interface) is built by teammates in a separate Node.js service that calls this one.

## How it works

```
Agent runs -> writes log (JSONL)
Watcher reads the log; the run ends as failed
   -> POST /diagnose (ML service)
   -> features per step -> model -> ranked suspects + evidence
   -> ALERT to the user
   -> user opens the interface: ranked steps and explanation
   -> replay from a checkpoint with a fix -> trace diff -> SUCCESS
```

Diagnosis runs on a **completed failed run**, because the features use the steps after the suspect (for example, whether the next step reused its output).

## What we built

1. **Synthetic trace generator** (`generate_data.py`): a toy agent (plan, select_tool, call_tool, read_result, write_answer; tools: search, calculator, database_lookup; tasks: math, fact lookup, multi-hop). Each failing run has exactly one injected root-cause fault from six types: `wrong_tool_selected`, `bad_tool_arguments`, `tool_error_ignored`, `context_ignored`, `corrupted_state`, `retry_loop`. The guilty step is known, which gives ground-truth labels. Fixed seed, 5,000 runs, 31,489 steps, 34.8% failing.
2. **Counterfactual replay check** (in the generator): replaying 300 failing runs with the root-cause step fixed flipped all 300 to success, with step caching (98 cache hits). This verifies that the labels are consistent. It does not measure the model's diagnostic accuracy.
3. **Feature engineering** (`features.py`, about 45 features), in four groups:
   - *Errors:* error flag, error words, retries, whether this is the first flagged step, flags in neighbouring steps.
   - *Timing:* latency and output length, and how far they are from normal for that step type and tool (z-scores from successful runs only), plus an IsolationForest anomaly score fitted on successful steps.
   - *Text:* word overlap between a step's output and the next step's input, whether a call input covers the goal, whether the final answer matches extracted values.
   - *Structure:* position in the run, step type, tool, task type, and how many later steps depend on this step.
4. **Models** (`train.py`, `train_robust.py`): Random Forest and LightGBM, trained on failed runs, split by run (never by row). Two feature sets are compared: *all features* and *robust* (errors and text only).
5. **Evaluation** (`eda.py`, `train.py`, `train_robust.py`, `eval_unseen.py`, `leak_audit.py`): baselines, held-out runs, a leave-one-fault-type-out test (train without one fault, test on it), and leak audits.
6. **Explanations** (`explain.py`): SHAP finds the top features pushing a step's score up, which are turned into plain sentences. An optional local LLM (Gemma 2B via Ollama) writes two sentences. Its reply is validated and falls back to a template if it fails.
7. **API** (`app.py`): FastAPI `/diagnose` and `/health`; tests in `test_api.py`; a held-out check in `check_api.py`.
8. **Log watcher** (`watcher.py`): tails an agent's JSONL log, waits for a failed run, calls `/diagnose`, and prints or posts an alert.

## Results

Metrics: **Top-1** (the true culprit is ranked first), **Top-3** (in the top three), **MRR** (average of 1/rank). All results are from one train/test split by run; treat them as approximate.

**Baselines (failed runs):** blame the last step about 20%; blame the first step with an error flag about 33%; blame the first step with an error flag or error words about 48%; random forest knowing only position and step type about 36%.

**Held-out runs (in-distribution):**

| Model | Top-1 | Top-3 | MRR |
|---|---|---|---|
| All features | 98.0% | 100% | 0.989 |
| Robust (errors + text only) | 80.2% | 96.8% | 0.888 |

**Unseen fault type (trained without that fault):**

| Held-out fault | All: Top-1 | All: Top-3 | Robust: Top-1 | Robust: Top-3 |
|---|---|---|---|---|
| bad_tool_arguments | 0.0% | 52.2% | 34.1% | 60.9% |
| context_ignored | 0.0% | 75.6% | 91.7% | 98.7% |
| corrupted_state | 59.5% | 93.6% | 62.5% | 84.8% |
| retry_loop | 5.9% | 32.2% | 46.0% | 85.1% |
| tool_error_ignored | 96.8% | 100% | 71.0% | 93.2% |
| wrong_tool_selected | 1.0% | 92.3% | 39.7% | 83.2% |
| **Average** | **27%** | **74%** | **58%** | **84%** |

**What we learned:** feature-group tests showed that timing features alone reach about 88% and structure features about 87% in-distribution, while error and text features alone reach 65% and 49%. The generator makes fault types distinguishable through several independent routes, so the all-features model memorizes the generator's quirks and collapses on unseen faults (average Top-1 27%). The robust model scores lower in-distribution but is about twice as good on unseen faults, so it is the model we ship.

## Strengths

- Learns from execution traces and ranks steps, which is what the problem asks for.
- Clearly beats simple baselines, with run-level splits so no steps from one run appear in both train and test.
- Evaluation covers held-out runs and unseen fault types, and we report the weak results as well as the strong ones.
- Ground-truth labels come from fault injection and are checked by counterfactual replay (300 of 300 flipped).
- Top-3 is high (97% in-distribution, about 84% on unseen faults), which pairs well with replay: the model proposes suspects and replay confirms one.
- Small classical models: train in seconds on a CPU, no API cost, work offline.
- Explainable: every flagged step comes with evidence from SHAP, and explanations fall back to a template if the optional LLM is unavailable or gives an invalid answer.
- Decoupled design: the ML service has a simple JSON contract, so any platform can call it.
- Reproducible: fixed seed, scripts for every step, tests for the API.

## Limitations (please read)

1. **Synthetic data only.** The models are trained on one toy agent. We have **not** verified them on real agent traces, and accuracy there will probably be lower. The numbers above describe our generator, not real agents.
2. **The generator leaks.** Fault types leave recognisable timing, length and structural patterns. The all-features model exploits them, which is why its in-distribution score (98%) is not believable as a real-world number and why it fails on unseen faults. The robust model reduces this but does not remove it.
3. **Partial generalization.** Even the robust model reaches about 58% Top-1 on unseen fault types, and results vary by fault. For example, `context_ignored` is partly spotted through wording that a real agent would not produce. Results come from a single split with no confidence intervals, and unseen-fault numbers swung a lot between runs.
4. **Scores are not calibrated probabilities.** The model is often extremely confident (1.0 for the top step). Read the score as a suspicion rank.
5. **One root cause per run.** Real failures can have several causes or cascades.
6. **Diagnosis only after the run ends.** Features use later steps, so we cannot reliably alert in the middle of a run. Mid-run detection would need a model trained on partial traces.
7. **The replay "100% fix rate" is true by construction.** The generator knows the correct step. For a real agent, the fix must be chosen by the user, and replay shows whether it works.
8. **Explanations describe the model's reasoning, not proven causes.** SHAP says why the model scored a step highly. Replay is what tests causality. Evidence wording for low-ranked steps is basic, and the optional Gemma 2B explanation can be slow or wrong, which is why it is validated and not the default.
9. **Agent-specific assumptions.** "Normal" latency and output length are learned from the toy agent's successful runs and must be refit for another agent. The step types must be mapped to our five types. If `depends_on` is missing, a step is assumed to depend on the previous step.
10. **The watcher needs our log format.** One JSON object per line in the step schema, plus a final `run_end` event. A run with no new lines for 30 seconds is treated as failed.

## Status

- [x] Synthetic data generator with fault injection and replay check
- [x] Features, models, evaluation, leak audits
- [x] FastAPI `/diagnose`, tests, held-out API check
- [ ] Log watcher tested end to end
- [ ] Platform backend and interface calling `/diagnose`
- [ ] Replay from a chosen step demonstrated in the demo
- [ ] Test on real agent traces (future work)

## Run it

```
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm shap fastapi uvicorn requests joblib httpx pytest

python generate_data.py          # writes data/steps.csv and data/runs.csv
python eda.py                    # baselines
python train.py                  # all-features models, writes model.joblib
python train_robust.py           # robust model, writes model_robust.joblib
cp model_robust.joblib model.joblib   # ship the robust model

uvicorn app:app --port 8000      # then open http://localhost:8000/docs
python check_api.py              # checks the live API on 100 held-out runs
python watcher.py agent.log.jsonl --from-start
```

Optional: for LLM explanations, run Ollama with the `gemma2:2b` model. Use `?llm=false` for template explanations.

## API

`POST /diagnose?llm=false` with

```json
{"run_id": "r1", "steps": [{"step_index": 0, "step_type": "plan", "tool_name": null,
  "input_text": "...", "output_text": "...", "latency_ms": 230, "error_flag": 0,
  "retry_count": 0, "state_snapshot": {"task_type": "math_lookup", "extracted_values": {}},
  "depends_on": []}]}
```

returns `{"run_id", "ranking": [{"step_index", "step_type", "score", "evidence": [...]}], "explanation", "explanation_source"}`.

## Future work

Test on real agent traces (for example Who&When); retrain on traces from the platform's own agent; calibrate scores; multi-cause failures; mid-run detection.

## Related work

Who&When (failure attribution benchmark), Who&When Pro and Aegis (fault injection to create labeled failures), AgenTracer (counterfactual replay), TRAIL (trace annotation).