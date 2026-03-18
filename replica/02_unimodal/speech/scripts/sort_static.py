import pandas as pd

# ====== 你需要改的路径 ======
static_tsv = "features/static_features.tsv"          # 你的 static feature 表
control_id_tsv = "/home/taor/work/bridge2AI-data/phenotype/diagnosis/parkinsons_disease.tsv" # 第一列是 participant_id
out_tsv = "static_parkinson_prolonged-vowel.tsv"

TASK = "prolonged-vowel"

# ====== 读入 control participant_id（第一列）======
ctrl_ids = pd.read_csv(control_id_tsv, sep="\t", header=None, dtype=str).iloc[:, 0]
ctrl_ids = ctrl_ids.dropna().astype(str).str.strip()
ctrl_ids = set(ctrl_ids[ctrl_ids != ""].unique())

print(f"[INFO] control IDs: {len(ctrl_ids)}")

# ====== 读入 static features ======
df = pd.read_csv(static_tsv, sep="\t", dtype=str)  # 全部先按字符串读，避免ID被转成数字
df.columns = [c.strip() for c in df.columns]

# ====== 基本列检查 ======
need_cols = ["participant_id", "task_name"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"static_features.tsv 缺少必要列: {missing}，实际列名: {list(df.columns)[:30]}...")

# ====== 清洗 participant_id / task_name ======
df["participant_id"] = df["participant_id"].astype(str).str.strip()
df["task_name"] = df["task_name"].astype(str).str.strip()

# （可选）如果你的 control_id_tsv 里是 sub-xxx，而 static 里是 xxx（或相反），可以统一一下前缀：
# df["participant_id"] = df["participant_id"].str.removeprefix("sub-")
# ctrl_ids = {i.removeprefix("sub-") for i in ctrl_ids}

# ====== 过滤：control + task ======
df_ctrl = df[df["participant_id"].isin(ctrl_ids)].copy()
print(f"[INFO] rows after control filter: {len(df_ctrl)}")

df_ctrl_task = df_ctrl[df_ctrl["task_name"] == TASK].copy()
print(f"[INFO] rows after task filter ({TASK}): {len(df_ctrl_task)}")

# ====== 缺失/重复检查（最简单版）======
key_cols = [c for c in ["participant_id", "session_id", "task_name"] if c in df_ctrl_task.columns]
if key_cols:
    # 缺失
    miss = df_ctrl_task[key_cols].isna().any(axis=1).sum()
    print(f"[QC] missing in key cols {key_cols}: {miss}")
    # 重复
    dup = df_ctrl_task.duplicated(subset=key_cols).sum()
    print(f"[QC] duplicated keys {key_cols}: {dup}")

# ====== 保存 ======
df_ctrl_task.to_csv(out_tsv, sep="\t", index=False)
print(f"[DONE] saved -> {out_tsv}")
