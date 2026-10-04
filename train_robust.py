# train_robust.py - compare "all features" with "robust features" (errors + text only)
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import fit_normal_stats, build_features, feature_names

steps = pd.read_csv("data/steps.csv"); runs = pd.read_csv("data/runs.csv")
fail = runs.loc[runs.final_outcome == "fail", "run_id"]
ok = runs.loc[runs.final_outcome == "success", "run_id"]
ALL = feature_names()

errors = {"error_flag", "err_words", "flag_any", "prior_flags", "is_first_flag", "retry_count", "repeat_count"}
errors |= {f"{p}_{c}" for p in ("prev", "next") for c in ("error_flag", "err_words", "retry_count")}
text = {"overlap_next_in", "overlap_final", "input_covers_goal", "answer_matches_state"}
ROBUST = [c for c in ALL if c in errors | text]
SETS = {"all_features": ALL, "robust": ROBUST}

def metrics(d, score):
    d = d.assign(score=score)
    d["rank"] = d.groupby("run_id").score.rank(ascending=False, method="first")
    r = d.loc[d.is_root_cause == 1, "rank"]
    return {"top1": round((r == 1).mean(), 3), "top3": round((r <= 3).mean(), 3), "mrr": round((1 / r).mean(), 3)}

def fit(F, d):
    rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0)
    return rf.fit(d[F], d.is_root_cause)

# 1) normal split by run, same as train.py
tr_f, te_f = train_test_split(fail, test_size=0.2, random_state=0)
tr_ok, _ = train_test_split(ok, test_size=0.2, random_state=0)
stats = fit_normal_stats(steps[steps.run_id.isin(tr_ok)])
tr = build_features(steps[steps.run_id.isin(tr_f)], stats)
te = build_features(steps[steps.run_id.isin(te_f)], stats)
res, models = {}, {}
for name, F in SETS.items():
    models[name] = fit(F, tr)
    res[name] = metrics(te, models[name].predict_proba(te[F])[:, 1])
print("\nIn-distribution (held-out runs):\n", pd.DataFrame(res).T)

# 2) leave-one-fault-type-out, for both feature sets
fault_of = runs.set_index("run_id").fault_type
allf = build_features(steps[steps.run_id.isin(fail)], stats)
allf["fault"] = allf.run_id.map(fault_of)
rows = {}
for ft in sorted(allf.fault.dropna().unique()):
    trn, tst = allf[allf.fault != ft], allf[allf.fault == ft]
    row = {}
    for name, F in SETS.items():
        m = metrics(tst, fit(F, trn).predict_proba(tst[F])[:, 1])
        row[f"{name}_top1"], row[f"{name}_top3"] = m["top1"], m["top3"]
    rows[ft] = row
print("\nUnseen fault type (trained without it):\n", pd.DataFrame(rows).T)

# 3) save the robust model next to the existing one (does not overwrite model.joblib)
joblib.dump({"model": models["robust"], "name": "random_forest_robust",
             "features": ROBUST, "stats": stats}, "model_robust.joblib")
print("\nsaved model_robust.joblib")