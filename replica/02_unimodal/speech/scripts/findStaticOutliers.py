import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DROP_COLS_DEFAULT = {"participant_id", "session_id", "task_name", "label"}

def to_numeric_df(df, feature_cols):
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def iqr_bounds(s, k=1.5):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return None
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return float(lo), float(hi), float(q1), float(q3), float(iqr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_tsv", required=True)
    ap.add_argument("--patient_tsv", required=True)
    ap.add_argument("--out_dir", default="static_qc_out")
    ap.add_argument("--task", default="prolonged-vowel", help="不想按task过滤就传空：--task ''")
    ap.add_argument("--id_col", default="participant_id")
    ap.add_argument("--task_col", default="task_name")
    ap.add_argument("--iqr_k", type=float, default=1.5)
    ap.add_argument("--max_features", type=int, default=0, help=">0则只画前N个特征（按缺失率从低到高排序）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读入两组数据并打标签
    df_c = pd.read_csv(args.control_tsv, sep="\t", dtype=str)
    df_p = pd.read_csv(args.patient_tsv, sep="\t", dtype=str)
    df_c["label"] = "control"
    df_p["label"] = "patient"

    # 取列交集，避免两边列不一致
    common_cols = sorted(set(df_c.columns).intersection(set(df_p.columns)))
    df = pd.concat([df_c[common_cols], df_p[common_cols]], ignore_index=True)

    # 可选 task 过滤
    task = args.task if args.task != "" else None
    if task is not None:
        if args.task_col not in df.columns:
            raise ValueError(f"找不到 task 列 {args.task_col}")
        df[args.task_col] = df[args.task_col].astype(str).str.strip()
        df = df[df[args.task_col] == task].copy()

    # 基本检查：ID列
    if args.id_col not in df.columns:
        raise ValueError(f"找不到 participant id 列：{args.id_col}")

    # 选特征列：排除ID/任务/label等非特征列
    drop_cols = set(DROP_COLS_DEFAULT)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 转成数值矩阵（无法转的变 NaN）
    X = to_numeric_df(df, feature_cols)

    # 过滤掉“几乎全是NaN/非数值”的列（>95%缺失）
    miss_rate = X.isna().mean()
    keep = miss_rate[miss_rate < 0.95].index.tolist()
    X = X[keep]
    feature_cols = keep

    # 如果指定 max_features：按缺失率升序取前N个
    if args.max_features and args.max_features > 0:
        feature_cols = miss_rate.loc[feature_cols].sort_values().head(args.max_features).index.tolist()
        X = X[feature_cols]

    print(f"[INFO] samples={len(df)}  features={len(feature_cols)}  task={task or 'ALL'}")

    # 输出QC汇总（缺失率）
    qc = pd.DataFrame({
        "feature": feature_cols,
        "missing_rate_overall": X.isna().mean().values
    })
    qc.to_csv(out_dir / "static_qc_missing.tsv", sep="\t", index=False)

    # 逐特征做箱线图 & outlier
    outlier_rows = []
    summary_rows = []

    pdf_path = out_dir / "static_boxplots.pdf"
    with PdfPages(pdf_path) as pdf:
        for feat in feature_cols:
            vals = X[feat]
            # 分组
            mask_c = (df["label"] == "control").values
            mask_p = (df["label"] == "patient").values

            v_c = vals[mask_c].dropna()
            v_p = vals[mask_p].dropna()

            # 计算每组IQR边界并找outlier
            for grp, v, m in [("control", vals[mask_c], mask_c), ("patient", vals[mask_p], mask_p)]:
                b = iqr_bounds(v, k=args.iqr_k)
                if b is None:
                    continue
                lo, hi, q1, q3, iqr = b
                is_out = (v < lo) | (v > hi)
                is_out = is_out.fillna(False)
                out_ids = df.loc[m, args.id_col][is_out.values].astype(str).tolist()
                out_vals = v[is_out].tolist()
                for pid, vv in zip(out_ids, out_vals):
                    outlier_rows.append({
                        "feature": feat,
                        "group": grp,
                        "participant_id": pid,
                        "value": vv,
                        "lower": lo,
                        "upper": hi,
                        "iqr": iqr
                    })

            # 汇总信息
            summary_rows.append({
                "feature": feat,
                "n_control": int(v_c.shape[0]),
                "n_patient": int(v_p.shape[0]),
                "missing_rate_overall": float(vals.isna().mean()),
            })

            # 画箱线图（每个特征一页）
            fig = plt.figure(figsize=(7, 4))
            plt.boxplot([v_c.values, v_p.values], labels=["control", "patient"], showfliers=True)
            plt.title(feat)
            plt.ylabel("value")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # 保存 outliers 和 summary
    out_df = pd.DataFrame(outlier_rows)
    out_df.to_csv(out_dir / "static_outliers.tsv", sep="\t", index=False)

    sum_df = pd.DataFrame(summary_rows)
    # 加上每个特征的outlier计数
    if not out_df.empty:
        cnt = out_df.groupby(["feature", "group"]).size().unstack(fill_value=0)
        cnt.columns = [f"outliers_{c}" for c in cnt.columns]
        sum_df = sum_df.merge(cnt, left_on="feature", right_index=True, how="left").fillna(0)

    sum_df.to_csv(out_dir / "static_qc_summary.tsv", sep="\t", index=False)

    print(f"[DONE] PDF: {pdf_path}")
    print(f"[DONE] Outliers: {out_dir / 'static_outliers.tsv'}")
    print(f"[DONE] Summary: {out_dir / 'static_qc_summary.tsv'}")

if __name__ == "__main__":
    main()
