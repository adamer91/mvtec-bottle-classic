import argparse, os, csv, json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import gabor

def glcm_feats(img, distances=(1,2,3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=256, symmetric=True, normed=True):
    img_q = (img / (256/levels)).astype(np.uint8)
    g = graycomatrix(img_q, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    feats = {}
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        vals = graycoprops(g, prop).ravel()
        feats.update({f"glcm_{prop}_{i}": float(v) for i, v in enumerate(vals)})
    feats["glcm_entropy"] = float(-np.sum(g * np.log(g + 1e-12)))
    return feats

def lbp_feats(img, radius=1, n_points=8):
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points+2), density=True)
    return {f"lbp_u{i}": float(v) for i, v in enumerate(hist)}

def hog_feats(img, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
    h = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, visualize=False, feature_vector=True)
    # reduce dimensionality via simple stats over blocks
    return {
        "hog_mean": float(np.mean(h)),
        "hog_std": float(np.std(h)),
        "hog_energy": float(np.sum(h**2))
    }

def gabor_feats(img, freqs=(0.1, 0.2, 0.3), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    feats = {}
    k = 0
    for f in freqs:
        for t in thetas:
            filt_real, filt_imag = gabor(img, frequency=f, theta=t)
            feats[f"gabor_mean_{k}"] = float(np.mean(filt_real))
            feats[f"gabor_var_{k}"] = float(np.var(filt_real))
            k += 1
    return feats

def shape_moments(img):
    # assume foreground is non-zero (ROI)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"area":0,"perimeter":0,"circularity":0,"eccentricity":0, **{f"hu{i+1}":0 for i in range(7)}}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    circularity = (4*np.pi*area)/(perim**2 + 1e-12)
    M = cv2.moments(c)
    hu = cv2.HuMoments(M).flatten()
    # Eccentricity from moments (approx)
    a = (M["mu20"] + M["mu02"])/2
    b = np.sqrt(4*M["mu11"]**2 + (M["mu20"]-M["mu02"])**2)/2
    lam1, lam2 = a + b, a - b
    ecc = np.sqrt(1 - lam2/(lam1 + 1e-12)) if lam1>0 else 0
    feats = {"area":float(area),"perimeter":float(perim),"circularity":float(circularity),"eccentricity":float(ecc)}
    feats.update({f"hu{i+1}": float(v) for i, v in enumerate(hu)})
    return feats

def intensity_stats(img):
    vals = img[img>0] if np.any(img>0) else img.ravel()
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    sk = float(pd.Series(vals).skew()) if len(vals)>2 else 0.0
    kurt = float(pd.Series(vals).kurt()) if len(vals)>3 else 0.0
    p10, p50, p90 = [float(np.percentile(vals, q)) for q in (10,50,90)]
    return {"mean":mean,"std":std,"skew":sk,"kurtosis":kurt,"p10":p10,"p50":p50,"p90":p90}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--roi_dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        id_ = r["id"].replace("/", "_")
        p = os.path.join(args.preproc_dir, id_)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats = {}
        feats["id"] = r["id"]
        feats["label"] = 1 if r["label"]=="DEFECT" else 0
        feats.update(intensity_stats(img))
        feats.update(shape_moments(img))
        feats.update(glcm_feats(img))
        feats.update(lbp_feats(img))
        feats.update(hog_feats(img))
        feats.update(gabor_feats(img))
        rows.append(feats)

    out_df = pd.DataFrame(rows).fillna(0.0)
    out_df.to_csv(args.out_csv, index=False)
    # feature defs (brief)
    defs = {
        "intensity": ["mean","std","skew","kurtosis","p10","p50","p90"],
        "shape": ["area","perimeter","circularity","eccentricity","hu1..hu7"],
        "glcm": "contrast/dissimilarity/homogeneity/energy/correlation/ASM + entropy across distances/angles",
        "lbp": "uniform LBP histogram",
        "hog": "mean/std/energy of HOG vector",
        "gabor": "mean/var responses for multiple frequencies/orientations"
    }
    with open(os.path.join(os.path.dirname(args.out_csv), "feature_defs.json"), "w", encoding="utf-8") as f:
        json.dump(defs, f, indent=2)

if __name__ == "__main__":
    main()
