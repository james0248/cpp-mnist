import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Label (1 byte) + Image (28 * 28 bytes)
REC_LEN = 1 + 28 * 28


def write_rec(ds_split, out_path: Path):
    print(f"[prep] Creating directory: {out_path.parent}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ds_split)
    print(f"[write] Starting to write {n} records to {out_path.name}")

    with open(out_path, "wb") as w:
        for data in tqdm(ds_split, desc=f"Writing {out_path.name}", unit="samples"):
            label = int(data["label"])
            if not (0 <= label <= 9):
                raise ValueError(f"label out of range: {label}")
            w.write(bytes([label]))

            arr = np.asarray(data["image"], dtype=np.uint8)
            if arr.shape != (28, 28):
                raise ValueError(f"unexpected image shape: {arr.shape}")
            w.write(arr.reshape(-1).tobytes())

    # Check if the file size is correct
    size = out_path.stat().st_size
    expect = n * REC_LEN
    if size != expect:
        raise IOError(f"size mismatch: {size} vs {expect}")

    print(
        f"[write] {out_path.name}: {n} recs, {REC_LEN} B/rec, total {size / 1024 / 1024:.2f} MB"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output directory")
    args = ap.parse_args()
    out_dir = Path(args.out)

    print(f"[init] Output directory: {out_dir.resolve()}")
    print("[load] Loading MNIST dataset from HuggingFace...")
    ds = load_dataset("mnist")
    tr, te = ds["train"], ds["test"]
    print(f"[load] Train samples: {len(tr)}, Test samples: {len(te)}")
    assert len(tr) == 60000 and len(te) == 10000, "split sizes unexpected"

    write_rec(tr, out_dir / "train.rec")
    write_rec(te, out_dir / "test.rec")

    print(f"[done] Successfully saved binary files to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
