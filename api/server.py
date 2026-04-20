"""
Argus Cluster Moderation — API Server
======================================
Accepts a raw Argus task JSON, runs the preprocessing pipeline
(crop → features → UMAP → HDBSCAN), returns enriched JSON.

Run locally:
    pip install fastapi uvicorn python-multipart umap-learn hdbscan pillow numpy requests scikit-image
    uvicorn api.server:app --reload --port 8000

Environment variables:
    ARGUS_API_KEY   — X-API-Key header for cdn.stage.highfive-api.com (optional)
    ALLOWED_ORIGINS — comma-separated CORS origins (default: *)
"""

import io
import json
import logging
import os
import warnings
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("argus")

app = FastAPI(title="Argus Cluster Moderation API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── serve frontend ─────────────────────────────────────────────────────────────
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.exists(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

@app.get("/")
def root():
    index = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "message": "Argus Cluster Moderation API"}

# ── image helpers ──────────────────────────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    api_key = os.getenv("ARGUS_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.get(url, headers=headers, timeout=30)
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
    import base64
    ratio = max_w / max(img.width, 1)
    new_h = max(1, int(img.height * ratio))
    thumb = img.resize((max_w, new_h), Image.LANCZOS)
    buf   = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


# ── feature extraction ─────────────────────────────────────────────────────────

CROP_SIZE = (64, 64)

def extract_features(img: Image.Image) -> np.ndarray:
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
        hist = np.concatenate([
            np.histogram(arr[:,:,c], bins=16, range=(0,256))[0].astype(np.float32)
            for c in range(3)
        ])

    # HOG (512 dims)
    try:
        from skimage.feature import hog
        gray     = np.array(small.convert("L"))
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(8,8),
                       cells_per_block=(1,1), feature_vector=True).astype(np.float32)
    except Exception:
        gray     = np.array(small.convert("L")).flatten().astype(np.float32) / 255.0
        hog_feat = gray

    feat = np.concatenate([hist, hog_feat])
    norm = np.linalg.norm(feat)
    return (feat / norm) if norm > 0 else feat


# ── clustering ─────────────────────────────────────────────────────────────────

def run_umap_hdbscan(features: np.ndarray, n_neighbors: int, min_cluster_size: int):
    import umap
    import hdbscan as hdb

    n = len(features)
    n_neighbors = min(n_neighbors, max(2, n - 1))

    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=0.1, metric="cosine", random_state=42,
    )
    coords_2d = reducer.fit_transform(features)

    clusterer = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_2d)
    return coords_2d, labels


# ── main preprocessing endpoint ────────────────────────────────────────────────

@app.post("/api/preprocess")
async def preprocess(request: Request):
    """
    Accepts raw Argus task JSON in request body.
    Returns enriched JSON with crops, UMAP coords, and cluster assignments.
    """
    try:
        body = await request.body()
        argus_json = json.loads(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # validate it's an Argus task JSON
    if "result" not in argus_json or "frames" not in argus_json.get("result", {}):
        raise HTTPException(status_code=400, detail="Not a valid Argus task JSON (missing result.frames)")

    # params from query string with defaults
    params      = request.query_params
    min_cluster = int(params.get("min_cluster_size", 2))
    n_neighbors = int(params.get("umap_neighbors",   5))
    types_param = params.get("types", "LOGO,TEXT").upper().split(",")

    task_id     = argus_json.get("taskId", "unknown")
    media_info  = argus_json.get("result", {}).get("mediaInfo", {})
    frames      = argus_json.get("result", {}).get("frames", [])
    sample_rate = argus_json.get("taskParameters", {}).get("sampleRate", 1000) or 1000

    frame_index_map = {
        fr.get("id"): int(round(fr.get("time", 0) / sample_rate))
        for fr in frames
    }

    records = []
    frame_cache = {}  # url → Image

    for frame in frames:
        frame_id   = frame.get("id")
        frame_url  = frame.get("frameUrl")
        frame_time = frame.get("time", 0)
        frame_idx  = frame_index_map.get(frame_id, 0)

        log.info(f"Frame f{frame_idx} (id={frame_id})")

        # download frame image (cached per URL)
        if frame_url not in frame_cache:
            try:
                frame_cache[frame_url] = download_image(frame_url)
            except Exception as e:
                log.warning(f"  Could not download {frame_url}: {e}")
                frame_cache[frame_url] = None

        frame_img = frame_cache[frame_url]
        if frame_img is None:
            continue

        detections = frame.get("detections", [])
        included   = [d for d in detections if d.get("type", "").upper() in types_param]
        log.info(f"  {len(detections)} detections → {len(included)} kept ({'/'.join(types_param)})")

        for det in included:
            bbox = det.get("boundingBox", {})
            try:
                crop = crop_bbox(frame_img, bbox)
                feat = extract_features(crop)
                b64  = image_to_b64(crop)
            except Exception as e:
                log.warning(f"  Detection {det.get('id','?')[:8]} skipped: {e}")
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
                "_feat":        feat.tolist(),
                "label":        None,
                "label_status": "unlabeled",
                "cluster_id":   -1,
                "umap_x":       0.0,
                "umap_y":       0.0,
            })

    if len(records) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(records)} detections extracted — need at least 2. Check that frameUrl images are accessible from the server."
        )

    log.info(f"Clustering {len(records)} detections…")
    features_matrix = np.array([r["_feat"] for r in records])
    coords_2d, labels = run_umap_hdbscan(features_matrix, n_neighbors, min_cluster)

    for i, rec in enumerate(records):
        rec["umap_x"]    = float(coords_2d[i, 0])
        rec["umap_y"]    = float(coords_2d[i, 1])
        rec["cluster_id"] = int(labels[i])
        del rec["_feat"]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info(f"→ {n_clusters} clusters, {n_noise} noise")

    return JSONResponse({
        "task_id":          task_id,
        "media_info":       media_info,
        "sample_rate_ms":   sample_rate,
        "n_detections":     len(records),
        "n_clusters":       n_clusters,
        "n_noise":          n_noise,
        "min_cluster_size": min_cluster,
        "umap_neighbors":   n_neighbors,
        "detections":       records,
    })


@app.get("/api/health")
def health():
    return {"status": "ok"}
