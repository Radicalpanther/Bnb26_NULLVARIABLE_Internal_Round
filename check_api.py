# check_api.py - sends held-out failed runs through the LIVE API and checks the answers
import json
import pandas as pd, requests
from sklearn.model_selection import train_test_split

API = "http://localhost:8000/diagnose?llm=false"
steps = pd.read_csv("data/steps.csv"); runs = pd.read_csv("data/runs.csv")
fail = runs.loc[runs.final_outcome == "fail", "run_id"]
_, test_ids = train_test_split(fail, test_size=0.2, random_state=0)   # same split as train.py
BANNED = ["task ", "n steps", "relative position", "remained", "preceding", "subsequent", "step type"]

def to_payload(rid):
    g = steps[steps.run_id == rid].sort_values("step_index")
    out = []
    for r in g.to_dict("records"):
        dep = str(r["depends_on"])
        out.append({
            "step_index": int(r["step_index"]), "step_type": r["step_type"],
            "tool_name": None if pd.isna(r["tool_name"]) else r["tool_name"],
            "input_text": "" if pd.isna(r["input_text"]) else str(r["input_text"]),
            "output_text": "" if pd.isna(r["output_text"]) else str(r["output_text"]),
            "latency_ms": float(r["latency_ms"]), "error_flag": int(r["error_flag"]),
            "retry_count": int(r["retry_count"]),
            "state_snapshot": json.loads(r["state_snapshot"]),
            "depends_on": [] if dep == "nan" else [int(x) for x in dep.replace(".0", "").split(",") if x.strip().isdigit()],
        })
    return out

hit1 = hit3 = bad = n = 0
for i, rid in enumerate(test_ids.head(100)):
    truth = int(steps[(steps.run_id == rid) & (steps.is_root_cause == 1)].step_index.iloc[0])
    resp = requests.post(API, json={"run_id": rid, "steps": to_payload(rid)}, timeout=60).json()
    order = [s["step_index"] for s in resp["ranking"]]
    hit1 += order[0] == truth
    hit3 += truth in order[:3]
    n += 1
    for s in resp["ranking"]:
        bad += sum(any(b in e.lower() for b in BANNED) for e in s["evidence"])
    if i < 2:                                    # print two full examples to read
        print(f"\n--- {rid}: true culprit = step {truth}")
        for s in resp["ranking"][:3]:
            print(f"  step {s['step_index']} ({s['step_type']}) score {s['score']}: {s['evidence']}")
print(f"\nAPI top-1 {hit1/n:.3f}  top-3 {hit3/n:.3f}  over {n} held-out runs; banned-phrase hits: {bad}")