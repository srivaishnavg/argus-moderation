# Argus Cluster Annotator — Claude Context

## What This Project Is

A browser-based annotation tool for Argus logo/text detection results. Users upload a raw Argus task JSON, the backend runs a computer vision clustering pipeline, and the frontend lets them review, label, merge clusters, and export a labeled Argus JSON.

Deployed on Railway. Also runs locally on port 8000.

---

## Project Structure

```
argus-annotator/
├── api/server.py       # FastAPI backend — ALL pipeline logic lives here
├── public/index.html   # Entire frontend SPA — single self-contained file, zero build step
├── requirements.txt
├── nixpacks.toml       # Railway build config
└── docs/
    └── backend_overview.md   # Detailed pipeline write-up
```

**There are only two files to ever edit: `api/server.py` and `public/index.html`.**

---

## Architecture Rules — Read Before Making Any Change

### Frontend (`public/index.html`)

- **Single file, zero dependencies, zero build step.** No npm, no bundler. Edit and refresh.
- **`data.detections[]` is the ONLY source of truth.** Never store cluster membership separately. `clusters` is always derived via `buildClusters()` / `getCluster(cid)` from `data.detections`.
- **State variables** (all declared near the top of the `<script>`):
  - `data` — enriched JSON from server
  - `rawArgusJson` — original Argus task JSON (for full-fidelity export)
  - `selectedCluster` — currently focused cluster ID (string)
  - `clusterLocations` — `{ clusterId: locationClass }` persisted in enriched JSON
  - `clusterOverlays` — `Set<clusterId>` for digital overlay state, persisted as `_cluster_overlays`
  - `mergeSuggestions` — computed from UMAP coords, frontend-only
  - `frameIndexMap` — `{ frame_id: frameIndex }` derived from `sampleRate`
- **Enriched JSON** (the "save file") includes `_cluster_locations`, `_cluster_overlays`, `_raw_argus`. Load it back via the upload screen to resume without reprocessing.
- **No-cache headers** are set on ALL static file responses in the backend (`NO_CACHE_HEADERS`). If the browser is serving old JS after a code change, hard-refresh (`Ctrl+Shift+R`).

### Backend (`api/server.py`)

- **`/api/health`** must respond immediately — it is called before heavy ML imports for Railway health checks.
- **`_FRAME_CACHE`** — module-level LRU `OrderedDict` (max 120 frames). Shared between `/api/preprocess` (warms the cache) and `/api/frame-preview` (instant hover tooltip after preprocessing).
- **Determinism**: three fixes applied together — `init='random'` on UMAP, `NUMBA_NUM_THREADS=1` + `numba.set_num_threads(1)` at startup, `np.random.seed(42)` before each UMAP call. Same input always produces the same clusters.
- **Overlap suppression**: non-LOGO detections whose bbox is ≥50% covered by a LOGO bbox in the same frame are skipped before clustering. Threshold controllable via `overlap_threshold` query param.

---

## Key Data Shapes

### Argus detection (inside `frame.detections[]` in the task JSON)
```json
{
  "id": "uuid",
  "type": "LOGO",
  "class": "nike_swoosh",
  "boundingBox": { "left": 100, "top": 200, "right": 300, "bottom": 400 },
  "overlay": false,
  "locations": [{ "id": "uuid", "class": "baseline_apron", "manualSource": "..." }],
  "confidence": 0.95,
  "source": { "type": "LOGO", "modelId": "" },
  "isManuallyEdited": false,
  "isDeleted": false
}
```

### Enriched detection (in `data.detections[]` after preprocessing)
```json
{
  "detection_id": "uuid",
  "frame_id": "uuid",
  "frame_index": 42,
  "frame_url": "https://...",
  "type": "LOGO",
  "argus_class": "nike_swoosh",
  "bbox": { "left": 100, "top": 200, "right": 300, "bottom": 400 },
  "overlay": false,
  "cluster_id": 7,
  "umap_x": 3.14,
  "umap_y": -1.27,
  "crop_b64": "...(base64 JPEG)...",
  "label": null,
  "label_status": "unlabeled"
}
```

---

## Export Logic

### Labeled Argus JSON export
- **Label map**: `detection_id → label` for accepted detections only.
- **Patching**: LOGO type → update `class` + `manualSource`. Non-LOGO type → also promote `type` and `source.type` to `LOGO`.
- **Location injection**: builds a new LOCATION detection cloned from the LOGO bbox; links it via `det.locations[]`. Old location links are replaced (existing LOCATION detections are NOT deleted — only the link on `det.locations` is replaced).
- **Overlay clusters**: detections get `overlay: true` and `locations: []` (link removed, LOCATION detection itself untouched). Location injection is skipped for overlay clusters.
- **Eligibility for location**: `det.type === 'LOGO'` OR `(det.type === 'TEXT' AND det.label_status === 'accepted')` (TEXT accepted = will be promoted to LOGO on export).

### Two export paths
1. **With `rawArgusJson`** (preferred): deep-clone original, patch in-place. Full fidelity — all original fields preserved.
2. **Without `rawArgusJson`**: reconstruct Argus structure from enriched data. Functional but missing fields not captured during preprocessing.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Health check — immediate response, no ML |
| `/api/preprocess` | POST | Full pipeline: features → UMAP → HDBSCAN → suggestions |
| `/api/frame-preview` | GET | Full frame JPEG with red bbox drawn, scaled to 480px wide |
| `/` | GET | Serves `public/index.html` |

**`/api/preprocess` query params**: `min_cluster_size` (default 2), `umap_neighbors` (default 5), `types` (default `LOGO,TEXT`), `overlap_threshold` (default 0.5)

**`/api/frame-preview` query params**: `url`, `left`, `top`, `right`, `bottom` (bbox coords in original frame pixel space)

---

## Running Locally (Windows)

```powershell
# Kill any existing Python processes first
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Start server (from repo root)
cd "C:\Users\Srivaishnav Gandhe\source\repos\argus-annotator"
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

**First request is slow** (~30–90s) — numba JIT-compiles UMAP/pynndescent. Subsequent requests on the same process are fast.

### Common Windows Gotchas
- `pkill` doesn't exist on Windows — always use `Stop-Process` in PowerShell
- `--reload` hot-reload can be unreliable on Windows file watchers — kill and restart if changes aren't picked up
- Two Python processes can both listen on port 8000 simultaneously; the old one handles requests. Always kill first.
- If the browser is serving stale JS, hard-refresh (`Ctrl+Shift+R`) — no-cache headers are set but the browser may have a warm cache from before they were added

---

## Argus Portal URL Format

```
https://highfive-api.com/ai/tasks?task_id=<task_id>&shown_tab=frame&current_frame_index=<frame_index>
```

Frame index = `Math.round(frame_time_ms / sample_rate_ms)`

---

## Merge Suggestion Algorithm

Computed entirely in the frontend from UMAP 2D coordinates:

1. Centroid per cluster (average UMAP x,y)
2. Adaptive threshold = median of all nearest-neighbour distances × 2.0
3. All-pairs centroid distance comparison
4. Average-linkage agglomerative grouping (not single-linkage — avoids chain-linking)
5. Score = convert average intra-group distance to similarity [0.5, 0.99]

---

## Environment Variables

| Variable | Description |
|---|---|
| `PORT` | Set by Railway automatically |
| `ARGUS_API_KEY` | Forwarded as `X-API-Key` when downloading frame images from Argus CDN |
