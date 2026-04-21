# Argus Annotator — Backend System Overview

> Feature Extraction · UMAP · HDBSCAN · Merge Suggestions

---

## Overview

When you upload an Argus task JSON, the backend runs a full computer vision pipeline on every detection in the video — downloading frame images, extracting visual features, reducing dimensions, and clustering — before returning a single enriched JSON that the frontend uses to drive the annotation UI.

The pipeline has five sequential steps:

1. **Input parsing** — read frames, detections, and filter by type
2. **Feature extraction** — crop each detection, compute HSV + HOG features
3. **Dimensionality reduction** — compress 560D features to 2D with UMAP
4. **Clustering** — group detections by visual density with HDBSCAN
5. **Merge suggestions** — identify clusters that should be merged

---

## Step 1 — Input Parsing

The backend receives a raw Argus task JSON and reads:

- All frames from `result.frames`
- Each frame's detections, filtered to only the requested types (`TEXT`, `LOGO`, or both)
- The `sampleRate` to convert timestamps into frame numbers
- The `frameUrl` for each frame to download the actual image

Frames are deduplicated by URL — if multiple detections share the same frame, the image is downloaded only once and cached in memory for that request.

---

## Step 2 — Feature Extraction (per detection)

For each detection, the backend:

1. **Crops** the bounding box out of the full frame image, clamped to image boundaries with a minimum 1×1 pixel size
2. **Resizes** the crop to a fixed **64×64** thumbnail using Lanczos resampling
3. **Extracts two types of visual features** and concatenates them into a single vector

### HSV Colour Histogram — 48 values

The 64×64 crop is converted from RGB to HSV colour space using OpenCV. A 16-bin histogram is computed independently for each of the three channels (Hue, Saturation, Value), giving **3 × 16 = 48 values**.

HSV is used rather than RGB because it separates colour (hue) from brightness (value), making the features more robust to lighting changes across frames.

### HOG — Histogram of Oriented Gradients — 512 values

The crop is converted to greyscale and HOG features are extracted using scikit-image with:

- 8 orientation bins
- 8×8 pixel cells
- 1×1 block size

HOG captures the local edge and texture structure of the detection — the shape of letters, logo outlines, etc. — independent of colour.

### Combined Feature Vector — 560 dimensions

The 48 colour values and 512 HOG values are concatenated, then **L2-normalised** (divided by the vector's Euclidean length) so every detection sits on the unit hypersphere.

This normalisation is critical: it means cosine similarity between any two detections equals their dot product, and Euclidean distance in this space is a meaningful proxy for visual dissimilarity — a prerequisite for accurate clustering.

---

## Step 3 — Dimensionality Reduction with UMAP

With potentially thousands of detections each described by 560 numbers, direct clustering in that space is expensive and noisy (the "curse of dimensionality"). **UMAP** (Uniform Manifold Approximation and Projection) compresses each 560-dimensional feature vector down to **2D coordinates** while preserving the local neighbourhood structure — detections that look similar end up near each other on the 2D plane.

### Key parameters

| Parameter | Value | Purpose |
|---|---|---|
| `n_neighbors` | 5 | How many neighbours each point considers when building the manifold graph. Low values preserve tight local structure and produce finer separation between similar-but-distinct logos or text runs. |
| `min_dist` | 0.1 | Minimum distance allowed between points in the output. Small values pack similar items closely together, making clusters visually compact. |
| `metric` | `cosine` | UMAP measures similarity in 560D using cosine distance, consistent with how the features were L2-normalised. |
| `random_state` | `42` | Seeds the random initialisation and the stochastic gradient descent optimiser. |
| `init` | `random` | Uses random initialisation instead of spectral, removing a non-deterministic BLAS eigendecomposition step. |

### Why determinism required three fixes

By default, identical inputs produced different cluster counts on every request (variance of ±3%). Three sources of non-determinism were identified and fixed:

| Fix | What it addresses |
|---|---|
| `init='random'` | Removes the spectral initialiser which uses multi-threaded BLAS eigendecomposition |
| `NUMBA_NUM_THREADS=1` + `numba.set_num_threads(1)` at startup | Serialises UMAP's SGD gradient updates so OS thread scheduling order cannot affect results |
| `np.random.seed(42)` before each UMAP call | Locks pynndescent's approximate nearest-neighbour search to a deterministic sequence |

After all three fixes, the NWSL dataset (3,323 detections) produces exactly **948 clusters on every run**.

---

## Step 4 — Clustering with HDBSCAN

The 2D UMAP coordinates are fed into **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise). Rather than requiring you to specify the number of clusters upfront, HDBSCAN finds clusters automatically by identifying dense regions and treating sparse regions as noise.

### How it works

1. Builds a hierarchy of density levels across the 2D plane
2. Clusters are regions that remain stable (dense) across a range of density thresholds
3. Points in sparse regions that don't belong to any stable cluster are labelled **noise** (cluster ID = −1)

### Key parameters

| Parameter | Value | Effect |
|---|---|---|
| `min_cluster_size` | 2 | A cluster must contain at least 2 detections. Single isolated detections fall into noise. |
| `min_samples` | 1 | Allows every point to be a core point, reducing the number of detections classified as noise. Produces more clusters with fewer orphaned detections. |
| `metric` | `euclidean` | Distance measured in 2D UMAP space. The UMAP projection has already encoded visual similarity, so Euclidean distance in 2D is a good proxy. |

The number of clusters is entirely data-driven. A video with many visually distinct logos might produce 50 clusters; a broadcast with repeating on-screen text might produce 5.

---

## Step 5 — Suggested Merges

HDBSCAN sometimes splits what is visually the same entity (same logo, same brand watermark, same recurring text) into multiple clusters due to minor visual variation between frames — slight crops, lighting changes, compression artefacts. The merge suggestion system identifies these split clusters and surfaces them for the annotator to review.

### How suggestions are computed

The frontend computes suggestions from the **UMAP 2D coordinates** using average-linkage agglomerative clustering in five sub-steps:

1. **Centroid per cluster** — for each cluster, compute the average (x, y) of all its detections in UMAP space
2. **Adaptive threshold** — for every cluster, find the distance to its nearest neighbouring cluster centroid. Take the **median** of all those nearest-neighbour distances and multiply by **2.0**. This gives a threshold relative to the actual spread of clusters in this specific video.
3. **All-pairs comparison** — every cluster centroid is compared against every other (O(n²) scan)
4. **Average-linkage merge** — pairs within the threshold are grouped using agglomerative clustering. Two groups merge only if the **average** of all cross-pair distances between their members is within the threshold.
5. **Scoring** — each suggestion group is scored by converting the average intra-group distance to a 0–1 similarity score, clamped to [0.5, 0.99]. Groups are sorted highest-similarity first.

### Linkage strategy comparison

| Method | Merge condition | Risk |
|---|---|---|
| Single-linkage | Any one pair within threshold | Chain-linking — unrelated clusters can merge through intermediate bridges |
| Complete-linkage | Every pair within threshold | One distant outlier pair vetoes the entire group — too conservative |
| **Average-linkage** *(current)* | **Mean of all cross-pairs within threshold** | Balanced — tolerates one outlier pair without allowing full chain-linking |

---

## Summary — Full Pipeline Flow

```
Argus Task JSON
      │
      ▼  (Python FastAPI backend)
Download frame images (cached per unique URL)
      │
      ▼  for each detection:
Crop bounding box → resize to 64×64
      │
      ├─ HSV histogram  (48 values)   — colour / lighting signature
      └─ HOG features   (512 values)  — edge / texture / shape
           └─ concat + L2 normalise  →  560D unit vector
      │
      ▼
UMAP  560D → 2D  (cosine metric, deterministic with random init + single-threaded numba)
      │
      ▼
HDBSCAN on 2D coords  →  cluster labels  (noise = −1)
      │
      ▼
Average-linkage on UMAP centroids  →  ranked merge suggestions
      │
      ▼
Enriched JSON: detections with crop thumbnails (base64),
               UMAP coords, cluster IDs, merge suggestions
```

| Stage | What happens |
|---|---|
| 1. Input | Parse Argus task JSON — extract frames, detections, frame URLs |
| 2. Image download | Fetch each unique frame URL; cache per request to avoid duplicate downloads |
| 3. Crop + resize | Crop bounding box from frame; resize to 64×64 with Lanczos |
| 4. HSV histogram | 16-bin histogram per channel in HSV space → 48 values |
| 5. HOG | Histogram of Oriented Gradients on greyscale crop → 512 values |
| 6. L2 normalise | Concatenate + divide by Euclidean norm → 560D unit vector |
| 7. UMAP | 560D → 2D, cosine metric, deterministic with random init + single-threaded numba |
| 8. HDBSCAN | Density clustering on 2D coords → cluster labels (noise = −1) |
| 9. Merge suggestions | Average-linkage on UMAP centroids, adaptive threshold → ranked suggestion groups |
| 10. Response | Enriched JSON: detections with crop images (base64), UMAP coords, cluster IDs, merge suggestions |
