<p align="center">
  <img src="assets/assetlens_logo.png" alt="AssetLens logo" width="160">
</p>

# AssetLens

**Local AI-powered image search, similarity analysis and creative asset organization.**

AssetLens is a Python and Streamlit application that helps users explore their own photographic collections. It combines a pretrained vision-language model with vector search, perceptual hashing and unsupervised clustering, all running locally on a CPU.

> **Project scope:** AssetLens is an AI engineering application built around a pretrained OpenCLIP model. The project integrates and evaluates the model; it does **not** train a new foundation model.

## Features

| Feature | What it does |
| --- | --- |
| **Semantic Search** | Find photographs using natural-language descriptions and CLIP similarity scores. |
| **Similar Images** | Select an existing photograph and retrieve visually or semantically related images. |
| **Duplicate Detection** | Identify byte-identical files with SHA-256 and potential near-duplicates using perceptual hashing (dHash). |
| **Duplicate-Aware Search** | Optionally suppress likely duplicate photographs in search results. |
| **Smart Library** | Browse and filter by filename extension, orientation and minimum image width; sort by size or resolution. |
| **AI Collections** | Organize image embeddings into similarity-based groups with K-Means clustering. |
| **Incremental Indexing** | Cache image embeddings and reuse them when files have not changed. |

AssetLens supports `.jpg`, `.jpeg`, `.png`, `.jfif`, `.webp` and `.avif` files in the local image collection, subject to Pillow's decoding support.

## How it works

```text
Local photographs
       |
       v
Pillow image preprocessing
       |
       v
Pretrained OpenCLIP ViT-B/32 (CPU)
       |
       v
Normalized 512-dimensional image embeddings
       |
       +--> Cached in image_index.npz
       |
       +--> Text-to-image similarity search
       |
       +--> Image-to-image similarity search
       |
       +--> K-Means clustering (AI Collections)

Separate image fingerprints --> Exact / near-duplicate detection
Metadata extraction        --> Smart Library filters
```

The pretrained model is **OpenCLIP `ViT-B-32`**, using the `laion2b_s34b_b79k` weights. Image and text embeddings are compared using cosine similarity (a dot product of normalized vectors). Similarity scores are **not** calibrated confidence probabilities.

## Requirements

- Windows and Python **3.12** (the tested setup).
- A CPU; a dedicated GPU is not required.
- Internet access for the initial installation and first download of pretrained model weights. Subsequent model inference runs locally after the weights have been cached.
- Your **own photographs**. The development image collection is intentionally **not** included in this repository.

The project was tested on an **Intel Core i5-9400**, **8 GB RAM**, and Intel integrated graphics. These are test conditions, not formal minimum hardware requirements.

## Quick start (Windows PowerShell)

### 1. Clone the project

```powershell
git clone https://github.com/Mmzebedas/AssetLens.git
cd AssetLens
```

Alternatively, download the repository ZIP from GitHub, extract it and open a PowerShell terminal in that folder.

### 2. Create a Python virtual environment

```powershell
py -3.12 -m venv .venv
```

### 3. Install the dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` records the versions used for development, including CPU-only PyTorch packages. Installing dependencies may require a substantial download.

### 4. Add images to the library

Copy photographs into the `images/` folder. The folder is initially empty, apart from `.gitkeep`. **AssetLens does not automatically download a photograph collection or upload your images to a remote service.**

### 5. Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, typically `http://localhost:8501`.

### 6. Index the images

Open **Library** and select **Index / Update images**. This generates the CLIP embeddings required for semantic search and clustering. New or changed images are processed; unchanged files are reused from the local index.

## Using AssetLens

- **Search:** Enter a natural-language description. Adjust the number of results and choose whether to hide likely duplicate photographs.
- **Similar Images:** Select a photograph already in your indexed collection and run the similarity search.
- **Duplicates:** Run exact SHA-256 detection or configure the perceptual-hash threshold for potentially similar images.
- **Library:** Index or update photographs, inspect the gallery, use metadata filters and generate AI Collections. The K-Means slider sets the number of groups; these are similarity clusters, not predefined image categories.

## Performance and evaluation

The following figures are **single-run local measurements** from the hardware described above, not service-level guarantees.

| Measurement | Observed result |
| --- | ---: |
| Indexed image collection | 193 photographs |
| Mean search latency across three benchmark queries | **39.3 ms** |
| CLIP model loading | 3.816 s |
| Cached index check (193 images, search benchmark) | 0.007 s |
| Initial encoding of 25 images (separate scalability benchmark) | 2.398 s |
| Incremental encoding of 168 additional images | 14.329 s |
| Fully cached check (193 images, scalability benchmark) | 0.016 s |
| Final Python process memory in search benchmark | ~950 MB |
| Automated tests | **7 passed** |

Search latency was measured **after model initialization** and includes query text embedding and ranking within a small local collection. Process RAM figures are snapshots, not peak memory measurements. Small timing differences between runs should not be treated as evidence of an optimization.

Manual retrieval checks on curated images were encouraging, but the dataset and query set are too limited to claim a general model accuracy percentage.

To reproduce benchmarks with your own images:

```powershell
.\.venv\Scripts\python.exe benchmark.py
.\.venv\Scripts\python.exe benchmark_scalability.py
```

The scalability benchmark uses a separate temporary index, leaving the regular image cache unchanged.

## Testing

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The seven tests cover exact and perceptual duplicate detection, hash distance, cache reuse, corrupted images, deleted images and corrupted cache recovery.

For the main application's lint checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check app.py image_index.py
```

## Project structure

```text
AssetLens/
├── app.py                  # Streamlit interface
├── ai_engine.py            # OpenCLIP model and image/text embeddings
├── image_index.py          # Embedding cache and incremental indexing
├── duplicate_utils.py     # Exact and perceptual duplicate detection
├── search_dedup.py         # Search result duplicate suppression
├── clustering_engine.py   # K-Means grouping
├── cluster_browser.py      # Visual AI Collections
├── metadata_utils.py      # Image metadata extraction
├── metadata_browser.py    # Smart Library explorer
├── benchmark.py           # Search and resource benchmarks
├── benchmark_scalability.py
├── tests/                  # Automated tests
├── assets/                 # Logo and visual styles
├── .streamlit/config.toml  # Streamlit theme
├── images/.gitkeep         # User-supplied photographs go here
└── requirements.txt
```

## Design choices and limitations

- **Local-first workflow:** Photographs and generated embeddings remain on the computer running AssetLens. Model weights may be downloaded during initial setup.
- **Pretrained model:** Retrieval depends on OpenCLIP's learned representations; AssetLens does not fine-tune CLIP.
- **Finite evaluation:** Current reported performance comes from 193 images. Larger collections and more independent relevance judgments need further testing.
- **Approximate duplicates:** Perceptual hashes can miss altered copies or incorrectly flag different images.
- **K-Means:** The user chooses the number of groups; coherent clusters are not guaranteed.
- **Storage architecture:** A compressed NumPy `.npz` index is appropriate for the collections tested. It has not been validated as a large-scale image database.
- **No packaged demo photographs:** Add your own images before indexing and searching.

## Possible next steps

Broader benchmarking, more systematic relevance evaluation, additional reliability tests, and image handling for much larger collections.

---

**Purpose:** An applied AI engineering project demonstrating pretrained-model integration, vector search, image processing, caching, unsupervised learning, software testing and local CPU performance evaluation.
