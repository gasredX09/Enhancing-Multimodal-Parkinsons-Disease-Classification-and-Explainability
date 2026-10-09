import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


def read_and_label(control_tsv, patient_tsv, task=None, id_col="participant_id", task_col="task_name"):
    df_c = pd.read_csv(control_tsv, sep="\t", dtype=str)
    df_p = pd.read_csv(patient_tsv, sep="\t", dtype=str)

    df_c["label"] = 0
    df_p["label"] = 1

    # Align different columns
    common_cols = sorted(set(df_c.columns).intersection(set(df_p.columns)))
    df_c = df_c[common_cols]
    df_p = df_p[common_cols]

    df = pd.concat([df_c, df_p], ignore_index=True)

    # Filter by acoustic task (optional)
    if task is not None:
        if task_col not in df.columns:
            raise ValueError(f"cannot find task column {task_col}，actual column：{list(df.columns)[:30]}")
        df[task_col] = df[task_col].astype(str).str.strip()
        df = df[df[task_col] == task].copy()

    # QC: ensure no duplicate participant in patient and health
    if id_col in df.columns:
        n_dup = df.duplicated(subset=[id_col]).sum()
        if n_dup > 0:
            print(f"[WARN] {id_col} duplicate：{n_dup} row（check your dataset）")

    return df


def prepare_xy(df, label_col="label"):
    y = df[label_col].astype(int).values

    drop_cols = set([label_col, "participant_id", "session_id", "task_name"])
    feat_cols = [c for c in df.columns if c not in drop_cols]

    # 全部转成数值，不可转的变 NaN
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    print(f"[INFO] samples: {len(df)}  pos={y.sum()}  neg={len(df)-y.sum()}  features={X.shape[1]}")
    return X, y, feat_cols


def build_models():
    scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    noscale = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    models = {}

    models["LogReg"] = Pipeline([
        ("prep", scaled),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=800,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    base_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    models["LinearSVM(calibrated)"] = Pipeline([
        ("prep", scaled),
        ("clf", CalibratedClassifierCV(base_svm, method="sigmoid", cv=3))
    ])

    models["XGBoost"] = Pipeline([
        ("prep", noscale),
        ("clf", XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    models["LightGBM"] = Pipeline([
        ("prep", noscale),
        ("clf", LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    models["CatBoost"] = Pipeline([
        ("prep", noscale),
        ("clf", CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False
        ))
    ])

    models["MLP"] = Pipeline([
        ("prep", scaled),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=3e-4,
            max_iter=600,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ))
    ])

    return models


def proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    return model.predict(X).astype(float)


def evaluate(X, y, models, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in models.items():
        aucs, accs, baccs, f1s, pres, recs = [], [], [], [], [], []

        for tr, te in skf.split(X, y):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]

            model.fit(Xtr, ytr)
            p = proba(model, Xte)

            # threshold 0.5
            yhat = (p >= 0.5).astype(int)

            aucs.append(roc_auc_score(yte, p))
            accs.append(accuracy_score(yte, yhat))
            baccs.append(balanced_accuracy_score(yte, yhat))
            f1s.append(f1_score(yte, yhat))
            pres.append(precision_score(yte, yhat, zero_division=0))
            recs.append(recall_score(yte, yhat, zero_division=0))

        rows.append({
            "model": name,
            "AUROC(mean)": np.mean(aucs), "AUROC(std)": np.std(aucs),
            "Acc(mean)": np.mean(accs),   "Acc(std)": np.std(accs),
            "BalAcc(mean)": np.mean(baccs),"BalAcc(std)": np.std(baccs),
            "Precision(mean)": np.mean(pres), "Precision(std)": np.std(pres),
            "Recall(mean)": np.mean(recs),    "Recall(std)": np.std(recs),
            "F1(mean)": np.mean(f1s),     "F1(std)": np.std(f1s),
        })

    return pd.DataFrame(rows).sort_values("AUROC(mean)", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_tsv", required=True)
    ap.add_argument("--patient_tsv", required=True)
    ap.add_argument("--task", default="prolonged-vowel", help="不想按 task 过滤就传空：--task ''")
    ap.add_argument("--id_col", default="participant_id")
    ap.add_argument("--task_col", default="task_name")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    task = args.task if args.task != "" else None

    df = read_and_label(args.control_tsv, args.patient_tsv, task=task, id_col=args.id_col, task_col=args.task_col)
    X, y, _ = prepare_xy(df, label_col="label")

    models = build_models()
    res = evaluate(X, y, models, n_splits=args.cv)

    print("\n=== Benchmark ===")
    print(res.to_string(index=False))

    out = "static_models_benchmark.csv"
    res.to_csv(out, index=False)
    print(f"\n[DONE] saved -> {out}")


if __name__ == "__main__":
    main()
