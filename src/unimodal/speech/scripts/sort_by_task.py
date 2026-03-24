import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

def filter_one(in_path: Path, out_path: Path, task: str, batch_size: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(in_path)

    writer = None
    kept = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        tbl = pa.Table.from_batches([batch])
        if "task_name" not in tbl.column_names:
            raise ValueError(f"{in_path.name} no task_name column（Actual column：{tbl.column_names}）")

        mask = pc.equal(pc.cast(tbl["task_name"], pa.string()), pa.scalar(task))
        ft = tbl.filter(mask)
        if ft.num_rows == 0:
            continue

        if writer is None:
            writer = pq.ParquetWriter(out_path, ft.schema, compression="zstd")
        writer.write_table(ft)
        kept += ft.num_rows

    if writer:
        writer.close()
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--task", default="prolonged-vowel")
    ap.add_argument("--glob", default="*.parquet")
    args = ap.parse_args()

    in_dir = Path(args.parquet_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    files = sorted(in_dir.rglob(args.glob))
    print(f"[INFO] files: {len(files)}  task={args.task}")

    for i, fp in enumerate(files, 1):
        rel = fp.relative_to(in_dir)
        out_path = out_dir / rel

        
        bs = 256 if "spectrogram" in fp.name else 8192

        try:
            kept = filter_one(fp, out_path, args.task, bs)
            print(f"[{i}/{len(files)}] OK  {rel}  kept_rows={kept}")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL {rel} -> {e}")

if __name__ == "__main__":
    main()