# make_sample_log.py
import json, pandas as pd
s = pd.read_csv("data/steps.csv"); r = pd.read_csv("data/runs.csv")
rid = r[r.final_outcome == "fail"].run_id.iloc[0]
enc = lambda o: o.item() if hasattr(o, "item") else str(o)
with open("agent.log.jsonl", "w") as f:
    for row in s[s.run_id == rid].sort_values("step_index").to_dict("records"):
        row["state_snapshot"] = json.loads(row["state_snapshot"])
        dep = str(row["depends_on"])
        row["depends_on"] = [] if dep == "nan" else [int(x) for x in dep.replace(".0", "").split(",") if x.strip().isdigit()]
        row["tool_name"] = None if str(row["tool_name"]) == "nan" else row["tool_name"]
        row.pop("is_root_cause")
        f.write(json.dumps(row, default=enc) + "\n")
    f.write(json.dumps({"type": "run_end", "run_id": rid, "status": "failed"}) + "\n")

    