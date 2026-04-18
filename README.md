# Argus Cluster Annotator

A browser-based annotation tool for [Argus](https://stage.highfive-api.com) logo/asset detection results. Upload a raw Argus task JSON, visually review auto-clustered detections, label them, and export an updated Argus JSON with your labels applied.

**No server required. No Python. Everything runs in the browser.**

---

## Live Demo

Deploy to Vercel in one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Zoomph-Dev/argus-annotator)

---

## Features

- **Drop a raw Argus task JSON** → the tool handles everything
- **In-browser clustering pipeline**
  - MobileNet (TF.js) visual embeddings on each bounding box crop
  - UMAP dimensionality reduction
  - HDBSCAN density-based clustering
- **Cluster browser** — all clusters shown as thumbnail grids
- **Drag & drop** crops between clusters
- **Right-click** any crop to move it or remove it to noise
- **Merge clusters** with one click
- **Label clusters** → accept / reject / defer with keyboard shortcuts
- **Correct frame numbers** matching the Argus portal (based on `sampleRate`)
- **Two exports:**
  - `↓ enriched JSON` — save your session mid-way and reload it later
  - `↓ labeled Argus JSON` — original Argus structure with labels applied

---

## How Labels Are Applied on Export

When you accept a cluster with a label (e.g. `pereira`), the **labeled Argus JSON** export applies these mutations per detection:

| Original type | `class` | `type` | `source.type` | `manualSource` |
|---|---|---|---|---|
| `LOGO` | → your label | unchanged | unchanged | `ClusterModeration` |
| `TEXT`, `LOCATION`, etc. | → your label | → `LOGO` | → `LOGO` | `ClusterModeration` |

All accepted detections also get `isManuallyEdited: true` and `edit_timestamp` set.

Detections in rejected / deferred / unlabeled clusters are **left completely untouched**.

---

## Usage

### Option A: Vercel (recommended)
1. Fork this repo to your GitHub account (or push directly to `Zoomph-Dev/argus-annotator`)
2. Go to [vercel.com](https://vercel.com) → New Project → Import from GitHub
3. Select the repo → Deploy (no build settings needed — it's a static site)
4. Open the deployed URL

### Option B: Run locally
```bash
# Just open the HTML file directly — no server needed
open public/index.html

# Or serve with any static server
npx serve public
python -m http.server 8080 --directory public
```

### Option C: GitHub Pages
1. Go to repo Settings → Pages
2. Set source to `main` branch, `/public` folder
3. Your tool will be live at `https://zoomph-dev.github.io/argus-annotator`

---

## Workflow

```
1. Open the tool
2. Drop your Argus task export JSON (e.g. task_export_865d32fa.json)
3. Wait for clustering (30–120s depending on detection count)
4. Browse clusters on the left panel
5. Click a cluster → review crops on the right
6. Type a label → click Accept (or press A)
7. Drag/right-click individual crops to move them between clusters
8. Merge similar clusters using the merge dropdown
9. Export:
   - "↓ enriched JSON"     → save progress, reload later
   - "↓ labeled Argus JSON" → upload back to Argus pipeline
```

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` | Accept cluster with current label |
| `R` | Reject cluster |
| `D` | Defer cluster |
| `→` / `Tab` | Next cluster |
| `←` / `Shift+Tab` | Previous cluster |
| `Esc` | Close drawer / context menu |

---

## Clustering Parameters

Adjust before loading a file:

| Parameter | Default | Description |
|---|---|---|
| `min cluster size` | `2` | Minimum detections for a valid cluster. Increase for fewer, larger clusters |
| `umap neighbors` | `5` | UMAP neighborhood size. Increase for more global structure |

---

## Important Note on Frame Access

The tool downloads `frameUrl` images from the Argus CDN (`cdn.stage.highfive-api.com`) directly in the browser to extract crops. The browser must have access to this domain. If you're accessing the hosted tool from within your Zoomph network or VPN, this works automatically.

If you see blank crops, it means the CDN URL was not accessible from your browser at the time of processing. The clustering will still work using color histogram fallback features, but crop thumbnails may appear dark.

---

## Project Structure

```
argus-annotator/
├── public/
│   └── index.html          # entire app — single self-contained file
├── vercel.json             # Vercel deployment config
├── .gitignore
└── README.md
```

---

## Tech Stack

- [TensorFlow.js](https://www.tensorflow.org/js) + MobileNet v2 — visual embeddings
- [umap-js](https://github.com/PAIR-code/umap-js) — dimensionality reduction
- Custom HDBSCAN — density-based clustering (implemented in vanilla JS)
- Zero build step — plain HTML/CSS/JS

---

## Contributing

Built by Zoomph for internal logo detection annotation workflows.
PRs welcome for:
- Better clustering algorithms
- Support for video frame scrubbing
- Multi-task batch annotation
- S3 upload integration on export
