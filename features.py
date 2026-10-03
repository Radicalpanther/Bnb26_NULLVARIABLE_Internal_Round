# features.py - one place for all feature engineering (training AND API use this)
import json, re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

STEP_TYPES = ["plan", "select_tool", "call_tool", "read_result", "write_answer"]
TOOLS = ["search", "calculator", "database_lookup", "none"]
TASKS = ["math_lookup", "fact_lookup", "multi_hop"]
ERR_PATTERN = r"error|timeout|503|500|not found|none|could not|failed|0 relevant|invalid"
LAG_COLS = ["lat_z", "len_z", "error_flag", "err_words", "retry_count"]

def _tokens(t):
    return set(re.findall(r"[a-z0-9\.\-]+", str(t).lower()))

def _jaccard(a, b):            # word overlap between two texts: 0 = nothing shared, 1 = identical
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0

def _norm_input(t):            # drop "[Attempt 2/Retry]" tags so repeated calls look identical
    return re.sub(r"\[attempt[^\]]*\]", "", str(t).lower()).strip()

def fit_normal_stats(ok_steps):
    """Mean/std of latency and output length per (step_type, tool) from SUCCESSFUL runs only + IsolationForest."""
    ok = ok_steps.copy()
    ok["tool_name"] = ok["tool_name"].fillna("none")
    stats = {}
    for col in ["latency_ms", "output_length"]:
        d = {"_all": [float(ok[col].mean()), float(ok[col].std())]}
        for (st, tn), g in ok.groupby(["step_type", "tool_name"])[col]:
            d[f"{st}|{tn}"] = [float(g.mean()), float(g.std(ddof=0))]
        stats[col] = d

    # IsolationForest for anomaly detection
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=0)
    iso.fit(ok[["latency_ms", "output_length"]])
    stats["iso"] = iso
    return stats

def _z(stats, col, st, tn, val):   # z-score = how many standard deviations from normal
    m, s = stats[col].get(f"{st}|{tn}", stats[col]["_all"])
    return (val - m) / (s if s and s > 1e-6 else 1.0)

def _parse_deps(x):
    return [int(t) for t in str(x).replace(".0", "").split(",") if t.strip().isdigit()]

def feature_names():
    base = ["step_index", "rel_pos", "steps_left", "n_steps", "latency_ms", "output_length",
            "error_flag", "retry_count", "lat_z", "len_z", "err_words",
            "flag_any", "prior_flags", "is_first_flag", "repeat_count",
            "overlap_next_in", "overlap_final", "n_deps", "n_dependents", "dep_flags", "n_extracted",
            "anomaly_score", "input_covers_goal", "answer_matches_state"]
    lags = [f"{p}_{c}" for c in LAG_COLS for p in ("prev", "next")]
    onehot = [f"st_{s}" for s in STEP_TYPES] + [f"tool_{t}" for t in TOOLS] + [f"task_{t}" for t in TASKS]
    return base + lags + onehot

def build_features(steps, stats):
    df = steps.sort_values(["run_id", "step_index"]).reset_index(drop=True).copy()
    df["tool_name"] = df["tool_name"].fillna("none")
    df["input_text"] = df["input_text"].fillna("").astype(str)
    df["output_text"] = df["output_text"].fillna("").astype(str)

    snap = df["state_snapshot"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    df["task_type"] = snap.apply(lambda d: d.get("task_type", "unknown"))
    df["n_extracted"] = snap.apply(lambda d: len(d.get("extracted_values", {})))
    df["extracted_values_dict"] = snap.apply(lambda d: d.get("extracted_values", {}))

    df["lat_z"] = [_z(stats, "latency_ms", a, b, c) for a, b, c in zip(df.step_type, df.tool_name, df.latency_ms)]
    df["len_z"] = [_z(stats, "output_length", a, b, c) for a, b, c in zip(df.step_type, df.tool_name, df.output_length)]
    df["err_words"] = df.output_text.str.contains(ERR_PATTERN, case=False, regex=True).astype(int)
    df["flag_any"] = ((df.error_flag == 1) | (df.err_words == 1)).astype(int)

    df["anomaly_score"] = stats["iso"].decision_function(df[["latency_ms", "output_length"]])

    df["input_covers_goal"] = [
        (len(set(re.findall(r"\d+|[A-Z][a-z]+", inp)) & set(re.findall(r"\d+|[A-Z][a-z]+", goal))) / max(len(set(re.findall(r"\d+|[A-Z][a-z]+", goal))), 1))
        if step_type == "call_tool" else 0
        for inp, goal, step_type in zip(df.input_text, snap.apply(lambda d: d.get("goal", "")), df.step_type)
    ]

    def _parse_floats(text):
        # Find all numbers including those with commas and decimals
        nums = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
        return [float(n.replace(",", "")) for n in nums]

    def _match_state(row):
        if row["step_type"] != "write_answer": return 0
        vals = row["extracted_values_dict"]
        ans = row["output_text"]
        ans_nums = _parse_floats(ans)
        state_vals = [float(v) for v in vals.values() if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() or (isinstance(v, str) and v.replace(',', '').replace('.', '', 1).isdigit())]

        matches = 0
        for an in ans_nums:
            if any(abs(an - sv) < 1e-3 for sv in state_vals):
                matches += 1
        return matches / len(ans_nums) if ans_nums else 0

    df["answer_matches_state"] = df.apply(_match_state, axis=1)

    g = df.groupby("run_id")
    df["n_steps"] = g.step_index.transform("count")
    df["rel_pos"] = df.step_index / (df.n_steps - 1).clip(lower=1)
    df["steps_left"] = df.n_steps - 1 - df.step_index
    df["prior_flags"] = g.flag_any.cumsum() - df.flag_any            # flagged steps before this one
    df["is_first_flag"] = ((df.flag_any == 1) & (df.prior_flags == 0)).astype(int)

    for c in LAG_COLS:                                               # neighbours' values
        df[f"prev_{c}"] = g[c].shift(1).fillna(0)
        df[f"next_{c}"] = g[c].shift(-1).fillna(0)

    df["norm_in"] = df.step_type + "|" + df.tool_name + "|" + df.input_text.map(_norm_input)
    df["repeat_count"] = df.groupby(["run_id", "norm_in"]).cumcount()   # same call seen earlier

    nxt_in = g.input_text.shift(-1).fillna("")
    df["overlap_next_in"] = [_jaccard(a, b) for a, b in zip(df.output_text, nxt_in)]
    final = g.output_text.transform("last")
    df["overlap_final"] = [_jaccard(a, b) for a, b in zip(df.output_text, final)]

    deps = df["depends_on"].map(_parse_deps)
    df["n_deps"] = deps.map(len)
    n_dep, dep_flags = np.zeros(len(df)), np.zeros(len(df))
    for _, idx in df.groupby("run_id").indices.items():
        pos = {df.step_index.iat[i]: i for i in idx}
        for i in idx:
            for d in deps.iat[i]:
                if d in pos:
                    n_dep[pos[d]] += 1
                    dep_flags[pos[d]] += df.flag_any.iat[i]
    df["n_dependents"], df["dep_flags"] = n_dep, dep_flags

    for s in STEP_TYPES: df[f"st_{s}"] = (df.step_type == s).astype(int)
    for t in TOOLS:      df[f"tool_{t}"] = (df.tool_name == t).astype(int)
    for t in TASKS:      df[f"task_{t}"] = (df.task_type == t).astype(int)
    return df
