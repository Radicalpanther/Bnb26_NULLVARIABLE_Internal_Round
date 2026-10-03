import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

steps = pd.read_csv("data/steps.csv"); runs = pd.read_csv("data/runs.csv")
df = steps.merge(runs[["run_id", "task_type", "final_outcome"]], on="run_id")
f = df[df.final_outcome == "fail"].reset_index(drop=True)

def top1(d, col):                      # share of runs where the top-scored step is the guilty one
    best = d.loc[d.groupby("run_id")[col].idxmax()]
    return best.is_root_cause.mean()

X = pd.get_dummies(f[["step_type", "task_type"]]).join(f[["step_index"]])
tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(f, groups=f.run_id))
rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                            n_jobs=-1, random_state=0)
rf.fit(X.iloc[tr], f.is_root_cause.iloc[tr])

t = f.iloc[te].copy()
t["s_last"] = t.step_index                       # blame the last step
t["s_first_err"] = t.error_flag * 1000 - t.step_index   # blame the first step with an error flag
t["s_rf"] = rf.predict_proba(X.iloc[te])[:, 1]   # forest that only knows position and step type
print("last step:", top1(t, "s_last"))
print("first error:", top1(t, "s_first_err"))
print("position-only RF:", top1(t, "s_rf"))