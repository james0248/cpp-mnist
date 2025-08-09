import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

REC_LEN = 1 + 28 * 28  # 785 bytes per record


def write_rec(ds_split, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(ds_split)
    with open(out_path, "wb") as w:
        for i, ex in enumerate(ds_split):
            label = int(ex["label"])
            if not (0 <= label <= 255):
                raise ValueError(f"label out of range: {label}")
            w.write(bytes([label]))
            arr = np.asarray(ex["image"], dtype=np.uint8)
            if arr.shape != (28, 28):
                raise ValueError(f"unexpected image shape: {arr.shape}")
            w.write(arr.reshape(-1).tobytes())
            if (i + 1) % 5000 == 0 or (i + 1) == n:
                print(f"\r  wrote {i + 1}/{n}", end="", flush=True)
    print()
    size = out_path.stat().st_size
    expect = n * REC_LEN
    if size != expect:
        raise IOError(f"size mismatch: {size} vs {expect}")
    print(
        f"[write] {out_path.name}: {n} recs, {REC_LEN} B/rec, total {size / 1024 / 1024:.2f} MB"
    )


def quick_check(path: Path):
    with open(path, "rb") as f:
        b = f.read(REC_LEN)
    if len(b) != REC_LEN:
        raise IOError("quick_check: file too small")
    lbl = b[0]
    img = b[1:]
    print(f"[check] first label={lbl}, first image bytes={len(img)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, help="output directory")
    args = ap.parse_args()
    out_dir = Path(args.out)

    print("[load] datasets: mnist (train/test)")
    ds = load_dataset("mnist")  # 자동 다운로드 & 캐시
    tr, te = ds["train"], ds["test"]
    assert len(tr) == 60000 and len(te) == 10000, "split sizes unexpected"

    write_rec(tr, out_dir / "train.rec")
    write_rec(te, out_dir / "test.rec")

    quick_check(out_dir / "train.rec")
    quick_check(out_dir / "test.rec")
    print("[done] saved to", out_dir.resolve())


if __name__ == "__main__":
    main()
