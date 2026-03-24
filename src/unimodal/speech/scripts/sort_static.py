import pandas as pd

# ====== Set your directory ======
static_tsv = "features/static_features.tsv"          # directory of static feature 
control_id_tsv = "/home/taor/work/bridge2AI-data/phenotype/diagnosis/parkinsons_disease.tsv" # first column is participant_id
out_tsv = "static_parkinson_prolonged-vowel.tsv"

TASK = "prolonged-vowel"

# ====== read control participant_id（first column）======
ctrl_ids = pd.read_csv(control_id_tsv, sep="\t", header=None, dtype=str).iloc[:, 0]
ctrl_ids = ctrl_ids.dropna().astype(str).str.strip()
ctrl_ids = set(ctrl_ids[ctrl_ids != ""].unique())

print(f"[INFO] control IDs: {len(ctrl_ids)}")

# ======read static features ======
df = pd.read_csv(static_tsv, sep="\t", dtype=str)  
df.columns = [c.strip() for c in df.columns]

# ====== check missing ======
need_cols = ["participant_id", "task_name"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"static_features.tsv missing column: {missing}，acutal column: {list(df.columns)[:30]}...")

# ====== clean participant_id / task_name ======
df["participant_id"] = df["participant_id"].astype(str).str.strip()
df["task_name"] = df["task_name"].astype(str).str.strip()

# 
# df["participant_id"] = df["participant_id"].str.removeprefix("sub-")
# ctrl_ids = {i.removeprefix("sub-") for i in ctrl_ids}

# ====== filiter voice task：control + task ======
df_ctrl = df[df["participant_id"].isin(ctrl_ids)].copy()
print(f"[INFO] rows after control filter: {len(df_ctrl)}")

df_ctrl_task = df_ctrl[df_ctrl["task_name"] == TASK].copy()
print(f"[INFO] rows after task filter ({TASK}): {len(df_ctrl_task)}")

# ====== check missing and duplicates ======
key_cols = [c for c in ["participant_id", "session_id", "task_name"] if c in df_ctrl_task.columns]
if key_cols:
    # missing
    miss = df_ctrl_task[key_cols].isna().any(axis=1).sum()
    print(f"[QC] missing in key cols {key_cols}: {miss}")
    # duplicate
    dup = df_ctrl_task.duplicated(subset=key_cols).sum()
    print(f"[QC] duplicated keys {key_cols}: {dup}")

# ====== save results ======
df_ctrl_task.to_csv(out_tsv, sep="\t", index=False)
print(f"[DONE] saved -> {out_tsv}")
