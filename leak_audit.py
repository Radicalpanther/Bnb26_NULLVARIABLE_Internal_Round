# leak_groups.py - which GROUP of features produces the ~98%?
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import fit_normal_stats, build_features, feature_names

steps = pd.read_csv("data/steps.csv"); runs = pd.read_csv("data/runs.csv")
fail = runs.loc[runs.final_outcome == "fail", "run_id"]
ok = runs.loc[runs.final_outcome == "success", "run_id"]
tr_f, te_f = train_test_split(fail, test_size=0.2, random_state=0)   # split by run, as in train.py
tr_ok, _ = train_test_split(ok, test_size=0.2, random_state=0)
stats = fit_normal_stats(steps[steps.run_id.isin(tr_ok)])
tr = build_features(steps[steps.run_id.isin(tr_f)], stats)
te = build_features(steps[steps.run_id.isin(te_f)], stats)
ALL = feature_names()

errors = {"error_flag", "err_words", "flag_any", "prior_flags", "is_first_flag", "retry_count", "repeat_count"}
errors |= {f"{p}_{c}" for p in ("prev", "next") for c in ("error_flag", "err_words", "retry_count")}
timing = {"latency_ms", "output_length", "lat_z", "len_z", "anomaly_score"}
timing |= {f"{p}_{c}" for p in ("prev", "next") for c in ("lat_z", "len_z")}
text = {"overlap_next_in", "overlap_final", "input_covers_goal", "answer_matches_state"}
structure = set(ALL) - errors - timing - text     # position, step type, tool, task, dependency counts

def top1(F):
    rf = RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0)
    rf.fit(tr[F], tr.is_root_cause)
    d = te.assign(s=rf.predict_proba(te[F])[:, 1])
    return d.loc[d.groupby("run_id").s.idxmax()].is_root_cause.mean()

rows = []
for name, g in {"errors": errors, "timing": timing, "text": text, "structure": structure}.items():
    only = [c for c in ALL if c in g]
    rest = [c for c in ALL if c not in g]
    if only and rest:
        rows.append((name, len(only), round(top1(only), 3), round(top1(rest), 3)))
print(pd.DataFrame(rows, columns=["group", "n_features", "only_this_group", "all_except_group"]).to_string(index=False))