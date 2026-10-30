"""Per-image comparison of two runs tested with the same protocol (e.g. baseline vs DWT).

For every test image both runs translated: PSNR, SSIM (same calls as evaluate.py), DAB (brown) area error and the
Inception feature distance between fake_B and its paired real_B. Prints win rates, Wilcoxon signed-rank tests,
a per-HER2-score breakdown and a paired bootstrap of the FID/KID difference, and writes one CSV row per image.

    uv run python compare_per_sample.py --run_a <baseline run> --run_b <dwt run> --images_subdir test_latest_full1024 \
        --out /mnt/d/thesis/runs/_analysis/bci_pyramid_dwt
"""

import argparse
import csv
import os
import re

import cv2 as cv
import numpy as np
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from evaluate import InceptionFeatureExtractor, calculate_fid, polynomial_kernel

# metric -> True if higher is better
METRICS = {"psnr": True, "ssim": True, "dab_err": False, "feat_dist": False}


def brown_frac(rgb):
    """Fraction of DAB-brown pixels (same rule as the SIU figure scripts)."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return float(((r > g) & (g > b) & ((r - b) > 25) & (r < 235)).mean())


def pairs(run, subdir):
    d = os.path.join(run, subdir, "images")
    return {f[: -len("_fake_B.png")]: (os.path.join(d, f), os.path.join(d, f.replace("fake_B", "real_B")))
            for f in os.listdir(d) if f.endswith("_fake_B.png")}


def pixel_metrics(fake_path, real_path):
    fake, real = cv.imread(fake_path), cv.imread(real_path)
    return {
        "psnr": peak_signal_noise_ratio(fake, real),
        "ssim": structural_similarity(fake, real, channel_axis=2),
        "dab_err": abs(brown_frac(fake[..., ::-1]) - brown_frac(real[..., ::-1])),
    }


def kid(fr, ff, rr, idx):
    """Unbiased KID from precomputed kernel matrices, restricted to the (resampled) image indices idx."""
    n = len(idx)
    sub = lambda K: K[np.ix_(idx, idx)]
    off = lambda K: (K.sum() - np.trace(K)) / (n * (n - 1))
    return off(sub(rr)) + off(sub(ff)) - 2 * sub(fr).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_a", required=True, help="reference run (e.g. baseline)")
    p.add_argument("--run_b", required=True, help="compared run (e.g. dwt)")
    p.add_argument("--images_subdir", default="test_latest_full1024")
    p.add_argument("--out", required=True, help="directory for the per-image CSV")
    p.add_argument("--n_boot", type=int, default=200)
    opt = p.parse_args()

    pa, pb = pairs(opt.run_a, opt.images_subdir), pairs(opt.run_b, opt.images_subdir)
    names = sorted(pa.keys() & pb.keys())
    # a paired comparison only makes sense if both runs were scored against the same real image
    same_real = [cv.imread(pa[n][1]).tobytes() == cv.imread(pb[n][1]).tobytes() for n in tqdm(names, desc="check real_B")]
    print(f"{len(names)} common images, real_B identical in {sum(same_real)}")
    assert all(same_real), "runs were tested on different crops; per-image comparison is meaningless"

    rows = []
    for n in tqdm(names, desc="pixel metrics"):
        a, b = pixel_metrics(*pa[n]), pixel_metrics(*pb[n])
        score = re.search(r"_(\d\+?)$", n)
        rows.append({"name": n, "her2": score.group(1) if score else "",
                     **{f"{k}_a": v for k, v in a.items()}, **{f"{k}_b": v for k, v in b.items()}})

    ext = InceptionFeatureExtractor()
    real = ext.extract_features([pa[n][1] for n in names])
    fa = ext.extract_features([pa[n][0] for n in names])
    fb = ext.extract_features([pb[n][0] for n in names])
    for r, da, db in zip(rows, np.linalg.norm(fa - real, axis=1), np.linalg.norm(fb - real, axis=1)):
        r["feat_dist_a"], r["feat_dist_b"] = da, db

    os.makedirs(opt.out, exist_ok=True)
    csv_path = os.path.join(opt.out, f"per_sample_{opt.images_subdir}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== per image: B vs A ({len(rows)} images) ===")
    print(f"{'metric':<10}{'mean A':>10}{'mean B':>10}{'B better':>10}{'median d':>11}{'p (Wilcoxon)':>14}")
    for m, higher in METRICS.items():
        a, b = np.array([r[f"{m}_a"] for r in rows]), np.array([r[f"{m}_b"] for r in rows])
        d = (b - a) if higher else (a - b)  # > 0: B better
        print(f"{m:<10}{a.mean():>10.4f}{b.mean():>10.4f}{(d > 0).mean():>9.1%}{np.median(d):>11.4f}{wilcoxon(d).pvalue:>14.2e}")

    print("\n=== share of images where B is better, by HER2 score ===")
    groups = sorted({r["her2"] for r in rows})
    print(f"{'metric':<10}" + "".join(f"{g + ' (' + str(sum(r['her2'] == g for r in rows)) + ')':>12}" for g in groups))
    for m, higher in METRICS.items():
        line = f"{m:<10}"
        for g in groups:
            d = np.array([(r[f"{m}_b"] - r[f"{m}_a"]) * (1 if higher else -1) for r in rows if r["her2"] == g])
            line += f"{(d > 0).mean():>12.1%}"
        print(line)

    # set-level metrics: per HER2 score (KID is unbiased for small sets, FID is not) and a paired bootstrap over images
    K = lambda X, Y: polynomial_kernel(X, Y)
    rr, ffa, ffb, fra, frb = K(real, real), K(fa, fa), K(fb, fb), K(fa, real), K(fb, real)
    print("\n=== KID x1000 by HER2 score (A -> B) ===")
    for g in groups:
        idx = np.array([i for i, r in enumerate(rows) if r["her2"] == g])
        print(f"{g:<4} n={len(idx):<4} {1000 * kid(fra, ffa, rr, idx):8.2f} -> {1000 * kid(frb, ffb, rr, idx):8.2f}")

    rng = np.random.default_rng(0)
    d_fid, d_kid = [], []
    for _ in tqdm(range(opt.n_boot), desc="bootstrap"):
        idx = rng.integers(0, len(rows), len(rows))
        d_fid.append(calculate_fid(real[idx], fa[idx]) - calculate_fid(real[idx], fb[idx]))
        d_kid.append(kid(fra, ffa, rr, idx) - kid(frb, ffb, rr, idx))
    d_fid, d_kid = np.array(d_fid), 1000 * np.array(d_kid)
    print(f"\n=== paired bootstrap over images ({opt.n_boot}x), A - B (> 0: B better) ===")
    print(f"FID full set {calculate_fid(real, fa):.2f} -> {calculate_fid(real, fb):.2f}; "
          f"diff 95% CI [{np.percentile(d_fid, 2.5):.2f}, {np.percentile(d_fid, 97.5):.2f}], B better in {(d_fid > 0).mean():.1%}")
    print(f"KID x1000 diff 95% CI [{np.percentile(d_kid, 2.5):.2f}, {np.percentile(d_kid, 97.5):.2f}], "
          f"B better in {(d_kid > 0).mean():.1%}")
    print(f"\nper-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
