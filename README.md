# Argus Cluster Annotator

A browser-based annotation tool for [Argus](https://stage.highfive-api.com) logo/text detection results.  
Upload a raw Argus task JSON → the backend clusters every detection by visual similarity → you review, label, and merge clusters → export an updated Argus JSON with labels applied.

---

## How It Works

```
Argus Task JSON
      │
      ▼  (Python FastAPI backend)
Download frame images → crop each bounding box → resize to 64×64
      │
      ├─ HSV colour histogram  (48 values)  — colour/lighting signature
      └─ HOG shape features    (512 values) — edge/texture structure
           └─ concat + L2 normalise → 560D feature vector per detection
      │
      ▼
UMAP  560D → 2D  (cosine metric, fully deterministic)
      │
      ▼
HDBSCAN on 2D coords → cluster labels  (noise = −1)
      │
      ▼
Average-linkage on UMAP centroids → ranked merge suggestions
      │
      ▼
Enriched JSON returned to browser SPA
```

See [Backend System Overview](docs/Argus_Backend_Overview.docx) for a detailed write-up.

---

## Features

- **Drop a raw Argus task JSON** → server downloads frames, extracts features, clusters, and returns enriched JSON
- **Visual clustering pipeline** — HOG + HSV histogram features, UMAP dimensionality reduction, HDBSCAN density-based clustering — all running server-side in Python
- **Cluster browser** — filterable by All / Unlabeled / Labeled / Noise with per-category counts
- **Suggested merges** — automatically surfaces cluster groups that are visually similar, ranked by similarity score
- **Full keyboard workflow** for suggestion review:
  - `↑` / `↓` navigate suggestion rows
  - `Enter` on a suggestion → opens merge confirmation
  - `Enter` in the confirmation dialog → confirms merge
  - After merge, label input is auto-focused; `Enter` accepts and saves
  - Auto-advances to the next suggestion after saving or rejecting
- **Drag & drop** crops between clusters
- **Right-click** any crop to move it or remove it to noise
- **Click** a crop to focus it; use **arrow keys** to navigate crops; **Delete** moves focused crop to noise
- **Merge clusters** via drag, dropdown, or suggested merge panel
- **Label clusters** → accept / reject / defer with keyboard shortcuts (`A` / `R` / `D`)
- **Correct frame numbers** matching the Argus portal (based on `sampleRate`)
- **Two exports:**
  - `↓ enriched JSON` — saves your session; reload it later to continue
  - `↓ labeled Argus JSON` — original Argus structure with labels applied

---

## Deployment (Railway)

The app is deployed on [Railway](https://railway.app) using `nixpacks.toml`.

**Environment variables required on Railway:**

| Variable | Description |
|---|---|
| `PORT` | Set automatically by Railway |
| `ARGUS_API_KEY` | API key forwarded as `X-API-Key` when downloading frame images (optional) |

Railway builds via `nixpacks.toml` — no manual setup needed. Push to `master` and Railway redeploys automatically.

---

## Running Locally

**Requirements:** Python 3.11+

```bash
git clone https://github.com/srivaishnavg/argus-moderation.git
cd argus-moderation

pip install -r requirements.txt

python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000**

The `--reload` flag hot-reloads the backend when you save `api/server.py`. The frontend (`public/index.html`) is served as a static file — just refresh the browser after edits.

> **First request is slow** (~30–90s depending on detection count) because numba JIT-compiles the UMAP/pynndescent code. Subsequent requests on the same process are faster.

---

## Usage Workflow

```
1.  Open the tool
2.  Drop your Argus task export JSON
3.  Wait for preprocessing (downloads frames, extracts features, clusters)
4.  Browse clusters in the left panel — filter by All / Unlabeled / Labeled / Noise
5.  Click a cluster → review crops on the right panel
6.  Type a label → press Enter (or click Accept / press A)
7.  Use Suggested Merges panel to review visually similar cluster groups:
      • Click a suggestion row to preview clusters side-by-side
      • ↑/↓ to move between rows, Enter to merge, Esc to dismiss
8.  Drag crops or right-click to move individual detections between clusters
9.  Export:
      • "↓ enriched JSON"      → save progress, reload later
      • "↓ labeled Argus JSON" → upload back to the Argus pipeline
```

---

## Keyboard Shortcuts

### Cluster navigation
| Key | Action |
|---|---|
| `→` / `Tab` | Next cluster |
| `←` / `Shift+Tab` | Previous cluster |
| `A` | Accept cluster with current label |
| `R` | Reject cluster |
| `D` | Defer cluster |

### Detection (crop) navigation
| Key | Action |
|---|---|
| Click crop | Focus it |
| `← → ↑ ↓` | Navigate crops while one is focused |
| `Delete` | Move focused crop to noise |

### Suggested merge review
| Key | Action |
|---|---|
| `↑` / `↓` | Move between suggestion rows |
| `Enter` | Merge the currently previewed suggestion |
| `Enter` (in confirm dialog) | Confirm merge |
| `Esc` | Dismiss / cancel |

### After a suggestion merge
| Key | Action |
|---|---|
| Type label | Label input is auto-focused |
| `Enter` | Accept + save; auto-advances to next suggestion |
| `R` | Reject cluster; auto-advances to next suggestion |

---

## How Labels Are Applied on Export

When you accept a cluster with a label (e.g. `nike_swoosh`), the **labeled Argus JSON** export applies these mutations per detection:

| Original type | `class` | `type` | `source.type` | `manualSource` |
|---|---|---|---|---|
| `LOGO` | → your label | unchanged | unchanged | `ClusterModeration` |
| `TEXT`, `LOCATION`, etc. | → your label | → `LOGO` | → `LOGO` | `ClusterModeration` |

All accepted detections also get `isManuallyEdited: true` and `edit_timestamp` set.

Detections in rejected / deferred / unlabeled clusters are **left completely untouched**.

---

## Clustering Parameters

Adjustable via query params on the `/api/preprocess` endpoint or via the UI before loading:

| Parameter | Default | Description |
|---|---|---|
| `min_cluster_size` | `2` | Minimum detections for a valid cluster. Raise for fewer, larger clusters |
| `umap_neighbors` | `5` | UMAP neighbourhood size. Raise for more global structure |
| `types` | `LOGO,TEXT` | Comma-separated detection types to include |

---

## API

### `POST /api/preprocess`

Accepts a raw Argus task JSON body. Returns an enriched JSON with UMAP coordinates, cluster assignments, crop thumbnails (base64 JPEG), and merge suggestions.

**Query params:** `min_cluster_size`, `umap_neighbors`, `types`

**Response shape:**
```json
{
  "task_id": "...",
  "n_detections": 3323,
  "n_clusters": 948,
  "n_noise": 267,
  "merge_suggestions": [
    { "group": [3, 7, 12], "similarity": 0.91 }
  ],
  "detections": [
    {
      "detection_id": "...",
      "frame_index": 42,
      "cluster_id": 7,
      "umap_x": 3.14,
      "umap_y": -1.27,
      "crop_b64": "...",
      "label": null,
      "label_status": "unlabeled"
    }
  ]
}
```

### `GET /api/health`

Returns `{"status": "ok"}`. Responds immediately before any heavy ML imports — used by Railway for health checks.

---

## Project Structure

```
argus-annotator/
├── api/
│   └── server.py          # FastAPI backend — feature extraction, UMAP, HDBSCAN
├── public/
│   └── index.html         # Entire frontend SPA — single self-contained file
├── requirements.txt       # Python dependencies
├── nixpacks.toml          # Railway build config
└── README.md
```

---

## Tech Stack

**Backend (Python)**
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — API server
- [umap-learn](https://umap-learn.readthedocs.io/) — dimensionality reduction
- [hdbscan](https://hdbscan.readthedocs.io/) — density-based clustering
- [scikit-image](https://scikit-image.org/) — HOG feature extraction
- [opencv-python-headless](https://github.com/opencv/opencv-python) — HSV colour histogram
- [Pillow](https://pillow.readthedocs.io/) — image loading and cropping
- [numba](https://numba.pydata.org/) — JIT compilation for UMAP (pinned to 1 thread for determinism)

**Frontend**
- Vanilla HTML / CSS / JavaScript — zero build step, zero dependencies
- Served as static files from `public/` by the FastAPI backend

**Infrastructure**
- [Railway](https://railway.app) — deployment and hosting
- [nixpacks](https://nixpacks.com/) — build system (Python 3.11 + gcc)

---

## Determinism

The same input JSON always produces the same clusters. Three fixes are applied together to guarantee this:

1. `init='random'` on UMAP — removes the spectral initialiser which uses non-deterministic BLAS eigendecomposition
2. `NUMBA_NUM_THREADS=1` + `numba.set_num_threads(1)` at server startup — serialises UMAP's SGD so OS thread scheduling cannot affect results
3. `np.random.seed(42)` before each UMAP call — locks pynndescent's approximate nearest-neighbour search

---

## Contributing

Built by Zoomph for internal logo/text detection annotation workflows.
