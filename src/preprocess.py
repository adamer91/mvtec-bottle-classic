import argparse, os, glob, csv, yaml
import numpy as np
import cv2
from tqdm import tqdm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_clahe(gray, cfg):
    clahe_cfg = cfg.get("clahe", {})
    clip = clahe_cfg.get("clip_limit", 2.0)
    grid = tuple(clahe_cfg.get("tile_grid_size", [8,8]))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)

def denoise(img, cfg):
    d = cfg.get("denoise", {})
    method = d.get("method", "median")
    k = int(d.get("ksize", 3))
    if method == "median":
        return cv2.medianBlur(img, k)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k|1, k|1), 0)
    return img

def segment_roi(gray, cfg):
    thr_cfg = cfg.get("threshold", {})
    method = thr_cfg.get("method", "otsu")
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == "adaptive":
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, 5)
    else:
        th = (gray > 0).astype(np.uint8) * 255
    # Assume bright background → invert if needed to get bottle as foreground
    if np.mean(th) > 127:
        th = 255 - th
    # Morphology
    m = cfg.get("morphology", {})
    open_k = int(m.get("open_kernel", 3))
    close_k = int(m.get("close_kernel", 5))
    it = int(m.get("iterations", 1))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((open_k,open_k), np.uint8), iterations=it)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((close_k,close_k), np.uint8), iterations=it)
    return th

def largest_contour_mask(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bin_img)
    if not cnts:
        return mask
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask

def process_image(path, cfg):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if cfg.get("grayscale", True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg.get("resize"):
        s = int(cfg["resize"])
        gray = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA)
    gray = apply_clahe(gray, cfg)
    gray = denoise(gray, cfg)
    roi_mask = None
    if cfg.get("roi", {}).get("use_roi", True):
        bin_img = segment_roi(gray, cfg)
        roi_mask = largest_contour_mask(bin_img)
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    return gray, roi_mask

def build_manifest(mvtec_root, category, out_manifest):
    rows = []
    train_dir = os.path.join(mvtec_root, category, "train", "good")
    for p in glob.glob(os.path.join(train_dir, "**", "*.png"), recursive=True):
        id_ = os.path.relpath(p, mvtec_root).replace(os.sep, "/")
        rows.append([id_, "train", "OK", p, ""])
    test_dir = os.path.join(mvtec_root, category, "test")
    for cls in os.listdir(test_dir):
        for p in glob.glob(os.path.join(test_dir, cls, "**", "*.png"), recursive=True):
            id_ = os.path.relpath(p, mvtec_root).replace(os.sep, "/")
            label = "OK" if cls == "good" else "DEFECT"
            # ground truth masks exist for defective images
            mask = ""
            if label == "DEFECT":
                gt = os.path.join(mvtec_root, category, "ground_truth", cls,
                                  os.path.basename(p).replace(".png", "_mask.png"))
                if os.path.exists(gt):
                    mask = gt
            rows.append([id_, "test", label, p, mask])
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","split","label","img_path","mask_path"])
        w.writerows(rows)
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec_root", required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--out_preproc", default="preproc")
    ap.add_argument("--out_manifest", default="data/manifest.csv")
    ap.add_argument("--config", default="configs/preprocessing_config.yaml")
    ap.add_argument("--out_roi_dir", default="roi_masks")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(args.out_preproc)
    ensure_dir(os.path.dirname(args.out_manifest))
    ensure_dir(args.out_roi_dir)

    print("[1/2] Building manifest...")
    n = build_manifest(args.mvtec_root, args.category, args.out_manifest)
    print(f"Manifest rows: {n}")

    print("[2/2] Preprocessing images...")
    with open(args.out_manifest, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in tqdm(list(rdr)):
            in_path = row["img_path"]
            img, roi = process_image(in_path, cfg)
            out_name = row["id"].replace("/", "_")
            out_path = os.path.join(args.out_preproc, out_name)
            cv2.imwrite(out_path, img)
            if roi is not None and cfg.get("save_masks", True):
                roi_path = os.path.join(args.out_roi_dir, out_name)
                cv2.imwrite(roi_path, roi)

if __name__ == "__main__":
    main()
