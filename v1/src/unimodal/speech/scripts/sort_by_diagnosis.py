import argparse
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

def load_ids_from_tsv(tsv_path: Path, id_col: str = "participant_id") -> list[str]:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, comment="#")
    if id_col in df.columns:
        s = df[id_col]
    else:
        s = df.iloc[:, 0]  
    s = s.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.drop_duplicates().tolist()

def filter_parquet_arrow(in_path: Path, out_path: Path, ids: list[str], id_col: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    
    pf = pq.ParquetFile(in_path)

    ）
    ids_arr = pa.array(ids, type=pa.string())

    writer = None
    total_written = 0

    
    for batch in pf.iter_batches(batch_size=1024):
        tbl = pa.Table.from_batches([batch])

        if id_col not in tbl.column_names:
            raise ValueError(f"Column {id_col} does not exist in {in_path.name} ，actual columns are：{tbl.column_names}")

        
        id_as_str = pc.cast(tbl[id_col], pa.string())
        mask = pc.is_in(id_as_str, value_set=ids_arr)

        filtered = tbl.filter(mask)
        if filtered.num_rows == 0:
            continue

        if writer is None:
            writer = pq.ParquetWriter(out_path, filtered.schema, compression="zstd")

        writer.write_table(filtered)
        total_written += filtered.num_rows

    if writer is not None:
        writer.close()

    return total_written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col_in_tsv", default="participant_id")
    ap.add_argument("--id_col_in_parquet", default="participant_id")
    ap.add_argument("--glob", default="*.parquet")
    args = ap.parse_args()

    tsv = Path(args.tsv).resolve()
    parquet_dir = Path(args.parquet_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    ids = load_ids_from_tsv(tsv, args.id_col_in_tsv)
    print(f"[INFO] IDs: {len(ids)}")

    files = sorted(parquet_dir.rglob(args.glob))
    print(f"[INFO] Parquet files: {len(files)}")

    for i, fp in enumerate(files, 1):
        rel = fp.relative_to(parquet_dir)
        out_path = out_dir / rel
        try:
            n = filter_parquet_arrow(fp, out_path, ids, args.id_col_in_parquet)
            print(f"[{i}/{len(files)}] OK  {rel}  kept_rows={n}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL {rel} -> {e}")

if __name__ == "__main__":
    main()