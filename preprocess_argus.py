#!/usr/bin/env python3
"""
preprocess_argus.py
-------------------
Converts a raw Argus task export JSON into an enriched JSON
ready to load into the Argus Cluster Annotator web tool.

What it does:
  1. Downloads each frame image from Argus CDN
  2. Crops every bounding box (all types: LOGO, TEXT, LOCATION)
  3. Extracts visual features (HOG + color histogram)
  4. Runs UMAP dimensionality reduction
  5. Clusters with HDBSCAN
  6. Outputs enriched JSON with base64 crops + cluster assignments

Usage:
  python preprocess_argus.py --input task_export.json --output enriched.json

With API key (if CDN requires auth):
  python preprocess_argus.py --input task_export.json --output enriched.json --api_key YOUR_KEY

With local frame image (if CDN is not accessible):
  python preprocess_argus.py --input task_export.json --output enriched.json --frame_image /path/to/frame.jpg

Upload output to S3:
  python preprocess_argus.py --input task_export.json --output enriched.json --s3_bucket my-bucket --s3_key path/enriched.json

Install dependencies:
  pip install umap-learn hdbscan pillow numpy requests scikit-image boto3
"""

import argparse
import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
from PIL import Image

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────

CROP_SIZE = (64, 64)   # resize all crops before feature extraction


# ── image helpers ─────────────────────────────────────────────────────────────

def download_image(url: str, headers: dict = None) -> Image.Image:
    resp = requests.get(url, timeout=30, headers=headers or {})
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def crop_bbox(img: Image.Image, bbox: dict) -> Image.Image:
    left   = max(0, int(bbox.get("left",   0)))
    top    = max(0, int(bbox.get("top",    0)))
    right  = min(img.width,  int(bbox.get("right",  img.width)))
    bottom = min(img.height, int(bbox.get("bottom", img.height)))
    right  = max(right,  left + 1)
    bottom = max(bottom, top  + 1)
    return img.crop((left, top, right, bottom))


def image_to_b64(img: Image.Image, max_w: int = 128) -> str:
    """Resize to thumbnail and return base64 JPEG string."""
    ratio = max_w / max(img.width, 1)
    new_h = max(1, int(img.height * ratio))
    thumb = img.resize((max_w, new_h), Image.LANCZOS)
    buf   = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(img: Image.Image) -> np.ndarray:
    """
    HOG + HSV color histogram feature vector (560 dims), L2-normalised.
    Works without GPU, no PyTorch required.
    """
    small = img.resize(CROP_SIZE, Image.LANCZOS)
    arr   = np.array(small, dtype=np.uint8)

    # HSV color histogram (48 dims)
    try:
        import cv2
        hsv  = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hist = np.concatenate([
            np.histogram(hsv[:,:,c], bins=16, range=(0,256))[0].astype(np.float32)
            for c in range(3)
        ])
    except ImportError:
        # fallback: RGB histogram
        hist = np.concatenate([
            np.histogram(arr[:,:,c], bins=16, range=(0,256))[0].astype(np.float32)
            for c in range(3)
        ])

    # HOG (512 dims)
    try:
        from skimage.feature import hog
        gray = np.array(small.convert("L"))
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(8,8),
                       cells_per_block=(1,1), feature_vector=True).astype(np.float32)
    except Exception:
        gray = np.array(small.convert("L")).flatten().astype(np.float32) / 255.0
        hog_feat = gray

    feat = np.concatenate([hist, hog_feat])
    norm = np.linalg.norm(feat)
    return (feat / norm) if norm > 0 else feat


# ── clustering ────────────────────────────────────────────────────────────────

def run_umap_hdbscan(features: np.ndarray, n_neighbors: int, min_cluster_size: int):
    import umap
    import hdbscan as hdb

    n = len(features)
    n_neighbors = min(n_neighbors, max(2, n - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(features)

    clusterer = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_2d)

    return coords_2d, labels


# ── main pipeline ─────────────────────────────────────────────────────────────

def process(argus_json: dict, min_cluster_size: int, umap_neighbors: int,
            frame_image_path: str = None, api_key: str = None) -> dict:

    task_id      = argus_json.get("taskId", "unknown")
    media_info   = argus_json.get("result", {}).get("mediaInfo", {})
    frames       = argus_json.get("result", {}).get("frames", [])
    sample_rate  = argus_json.get("taskParameters", {}).get("sampleRate", 1000) or 1000

    # frame index = frame.time / sampleRate  (matches Argus portal frame numbers)
    frame_index_map = {
        fr.get("id"): int(round(fr.get("time", 0) / sample_rate))
        for fr in frames
    }

    records = []
    total_frames = len(frames)

    for fi, frame in enumerate(frames):
        frame_id   = frame.get("id")
        frame_url  = frame.get("frameUrl")
        frame_time = frame.get("time", 0)
        frame_idx  = frame_index_map.get(frame_id, fi)

        print(f"  Frame {fi+1}/{total_frames}  f{frame_idx}  (id={frame_id})", flush=True)

        # load frame image
        try:
            if frame_image_path:
                p = Path(frame_image_path)
                if p.is_file():
                    frame_img = Image.open(p).convert("RGB")
                else:
                    candidates = list(p.glob(f"*{frame_id}*")) or list(p.glob("*.jpg")) or list(p.glob("*.png"))
                    if not candidates:
                        raise FileNotFoundError(f"No image for frame {frame_id} in {frame_image_path}")
                    frame_img = Image.open(candidates[0]).convert("RGB")
            else:
                headers = {"X-API-Key": api_key} if api_key else {}
                try:
                    frame_img = download_image(frame_url, headers)
                except Exception:
                    # try mediaUrl as fallback
                    media_url = argus_json.get("taskParameters", {}).get("mediaUrl")
                    if media_url and media_url != frame_url:
                        print(f"    ↳ frameUrl failed, trying mediaUrl…", flush=True)
                        frame_img = download_image(media_url, headers)
                    else:
                        raise
        except Exception as e:
            print(f"    ⚠ Could not load frame: {e}", flush=True)
            continue

        detections = frame.get("detections", [])
        included = [d for d in detections if d.get("type", "").upper() in ("LOGO", "TEXT")]
        skipped  = len(detections) - len(included)
        print(f"    {len(detections)} detections → {len(included)} LOGO/TEXT kept, {skipped} LOCATION skipped", flush=True)

        for det in included:
            bbox = det.get("boundingBox", {})
            try:
                crop  = crop_bbox(frame_img, bbox)
                feat  = extract_features(crop)
                b64   = image_to_b64(crop)
            except Exception as e:
                print(f"    ⚠ Detection {det.get('id','?')[:8]} skipped: {e}", flush=True)
                continue

            records.append({
                "detection_id": det.get("id"),
                "frame_id":     frame_id,
                "frame_index":  frame_idx,
                "frame_time":   frame_time,
                "frame_url":    frame_url,
                "type":         det.get("type", ""),
                "argus_class":  det.get("class", ""),
                "confidence":   det.get("confidence"),
                "overlay":      det.get("overlay", False),
                "logo_quality": det.get("logo_quality"),
                "bbox":         bbox,
                "crop_b64":     b64,
                "_feat":        feat,           # temporary — removed after clustering
                "label":        None,
                "label_status": "unlabeled",
                "cluster_id":   -1,
                "umap_x":       0.0,
                "umap_y":       0.0,
            })

    if len(records) < 2:
        print(f"ERROR: Only {len(records)} detections extracted — need at least 2.", flush=True)
        sys.exit(1)

    print(f"\n  Clustering {len(records)} detections (UMAP → HDBSCAN)…", flush=True)

    features_matrix = np.array([r["_feat"] for r in records])
    coords_2d, labels = run_umap_hdbscan(features_matrix, umap_neighbors, min_cluster_size)

    for i, rec in enumerate(records):
        rec["umap_x"]    = float(coords_2d[i, 0])
        rec["umap_y"]    = float(coords_2d[i, 1])
        rec["cluster_id"] = int(labels[i])
        del rec["_feat"]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  → {n_clusters} clusters, {n_noise} noise points\n", flush=True)

    return {
        "task_id":          task_id,
        "media_info":       media_info,
        "sample_rate_ms":   sample_rate,
        "n_detections":     len(records),
        "n_clusters":       n_clusters,
        "n_noise":          n_noise,
        "min_cluster_size": min_cluster_size,
        "umap_neighbors":   umap_neighbors,
        "detections":       records,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Argus task JSON → enriched cluster JSON for the Argus Cluster Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_argus.py --input task.json --output enriched.json
  python preprocess_argus.py --input task.json --output enriched.json --api_key YOUR_KEY
  python preprocess_argus.py --input task.json --output enriched.json --frame_image frame.jpg
  python preprocess_argus.py --input task.json --output enriched.json --s3_bucket my-bucket --s3_key data/enriched.json
        """
    )
    parser.add_argument("--input",            required=True,  help="Argus task export JSON path")
    parser.add_argument("--output",           required=True,  help="Output enriched JSON path")
    parser.add_argument("--min_cluster_size", type=int, default=2,  help="Min detections per cluster (default: 2)")
    parser.add_argument("--umap_neighbors",   type=int, default=5,  help="UMAP n_neighbors (default: 5)")
    parser.add_argument("--api_key",          default=None,   help="X-API-Key for Argus CDN auth")
    parser.add_argument("--frame_image",      default=None,   help="Local frame image or directory (bypasses CDN)")
    parser.add_argument("--s3_bucket",        default=None,   help="S3 bucket to upload output to")
    parser.add_argument("--s3_key",           default=None,   help="S3 key for upload")
    args = parser.parse_args()

    print(f"\nArgus Preprocessor")
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    print(f"  min_cluster_size={args.min_cluster_size}  umap_neighbors={args.umap_neighbors}\n")

    with open(args.input) as f:
        argus_json = json.load(f)

    result = process(
        argus_json,
        min_cluster_size=args.min_cluster_size,
        umap_neighbors=args.umap_neighbors,
        frame_image_path=args.frame_image,
        api_key=args.api_key,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved → {args.output}  ({Path(args.output).stat().st_size // 1024} KB)")

    if args.s3_bucket:
        import boto3
        s3_key = args.s3_key or Path(args.output).name
        print(f"Uploading to s3://{args.s3_bucket}/{s3_key}…")
        boto3.client("s3").upload_file(args.output, args.s3_bucket, s3_key)
        print(f"✓ Uploaded → s3://{args.s3_bucket}/{s3_key}")


if __name__ == "__main__":
    main()
