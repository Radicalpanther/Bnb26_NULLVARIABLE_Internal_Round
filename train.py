# train.py
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import fit_normal_stats, build_features, feature_names

steps = pd.read_csv("data/steps.csv"); runs = pd.read_csv("data/runs.csv")
fail = runs.loc[runs.final_outcome == "fail", "run_id"]
ok = runs.loc[runs.final_outcome == "success", "run_id"]
tr_f, te_f = train_test_split(fail, test_size=0.2, random_state=0)      # split by RUN
tr_ok, _ = train_test_split(ok, test_size=0.2, random_state=0)

stats = fit_normal_stats(steps[steps.run_id.isin(tr_ok)])               # "normal" from training runs only
tr = build_features(steps[steps.run_id.isin(tr_f)], stats)
te = build_features(steps[steps.run_id.isin(te_f)], stats)
F = feature_names()

def metrics(d, score):
    d = d.assign(score=score)
    d["rank"] = d.groupby("run_id").score.rank(ascending=False, method="first")
    r = d.loc[d.is_root_cause == 1, "rank"]
    return {"top1": round((r == 1).mean(), 3), "top3": round((r <= 3).mean(), 3), "mrr": round((1 / r).mean(), 3)}

results, models = {}, {}
results["baseline_last_step"] = metrics(te, te.step_index)
results["baseline_first_flag"] = metrics(te, te.flag_any * 1000 - te.step_index)

rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=0)
rf.fit(tr[F], tr.is_root_cause); models["random_forest"] = rf
results["random_forest"] = metrics(te, rf.predict_proba(te[F])[:, 1])


lg = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", verbose=-1, random_state=0)
lg.fit(tr[F], tr.is_root_cause); models["lightgbm"] = lg
results["lightgbm"] = metrics(te, lg.predict_proba(te[F])[:, 1])

print(pd.DataFrame(results).T)
best = max(models, key=lambda k: results[k]["top1"])
joblib.dump({"model": models[best], "name": best, "features": F, "stats": stats}, "model.joblib")
print("saved:", best)