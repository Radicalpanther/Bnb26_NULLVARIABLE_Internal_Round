import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import fit_normal_stats, build_features, feature_names
from sklearn.model_selection import train_test_split

# Load data
steps = pd.read_csv("data/steps.csv")
runs = pd.read_csv("data/runs.csv")

# Ensure all faults
fault_types = runs[runs.final_outcome == "fail"].fault_type.unique()

def metrics(d, score):
    d = d.assign(score=score)
    d["rank"] = d.groupby("run_id").score.rank(ascending=False, method="first")
    r = d.loc[d.is_root_cause == 1, "rank"]
    return {"top1": round((r == 1).mean(), 3), "top3": round((r <= 3).mean(), 3), "mrr": round((1 / r).mean(), 3)}

# LOFTO Eval
results = {}
F = feature_names()

# 1. Normal In-Distribution
ok_ids = runs[runs.final_outcome == "success"].run_id
tr_ok_ids, te_ok_ids = train_test_split(ok_ids, test_size=0.2, random_state=42)
stats = fit_normal_stats(steps[steps.run_id.isin(tr_ok_ids)])

for hold_out in list(fault_types) + ["None (Normal)"]:
    if hold_out == "None (Normal)":
        tr_fail_ids = runs[(runs.final_outcome == "fail")].run_id # Use all just for baseline
        te_fail_ids = runs[(runs.final_outcome == "fail")].run_id
    else:
        tr_fail_ids = runs[(runs.final_outcome == "fail") & (runs.fault_type != hold_out)].run_id
        te_fail_ids = runs[(runs.final_outcome == "fail") & (runs.fault_type == hold_out)].run_id

    tr_steps = steps[steps.run_id.isin(tr_ok_ids) | steps.run_id.isin(tr_fail_ids)]
    te_steps = steps[steps.run_id.isin(te_fail_ids)]

    tr = build_features(tr_steps, stats)
    te = build_features(te_steps, stats)

    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(tr[F], tr.is_root_cause)
    results[hold_out] = metrics(te, clf.predict_proba(te[F])[:, 1])

print(pd.DataFrame(results).T)
