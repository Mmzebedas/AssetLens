# AssetLens

**AI-powered image search and organization — running locally on your computer.**

AssetLens is a Python and Streamlit application for exploring a collection of photographs using pretrained computer-vision models. It combines natural-language image search, visual similarity retrieval, duplicate detection, clustering, and metadata browsing in one interface.

![AssetLens logo](assets/assetlens_logo.png)

> **Local application, not a hosted website.** You run AssetLens on your own computer and open its Streamlit interface in a web browser. The repository does not include a photo collection; you supply your own images.

## Features

| Section | What it does |
| --- | --- |
| **Search** | Find images with descriptions such as *"a white dog outdoors"*. Results are ranked by CLIP similarity, with optional duplicate filtering. |
| **Similar Images** | Select an indexed reference image and retrieve visually related photographs. |
| **Duplicates** | Find byte-for-byte duplicates with SHA-256 and potential near duplicates with perceptual dHash. |
| **Library** | View your photographs, inspect image metadata, filter the collection, index new images, and explore AI Collections generated using K-Means. |

Additional capabilities include incremental image indexing, cached CLIP embeddings, and support for **JPG, JPEG, PNG, JFIF, WebP, and AVIF** images (subject to your Pillow installation's format support).

## Quick start — Windows (PowerShell)

These steps are intended for a fresh Windows computer and were developed with **Python 3.12**.

### 1. Install prerequisites

Install **Python 3.12** and **Git** if you do not already have them:

- Python: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads/win

Open **PowerShell** or the terminal in **VS Code**. Check the installations:

```powershell
py -3.12 --version
git --version
```

If the Python command fails, install Python 3.12 and ensure the Windows Python launcher is available. You can use `python --version` to check an existing installation, but the commands below assume the `py` launcher.

### 2. Download AssetLens

```powershell
git clone https://github.com/Mmzebedas/AssetLens.git
cd AssetLens
```

**Don't have Git?** On the [AssetLens GitHub page](https://github.com/Mmzebedas/AssetLens), select **Code → Download ZIP**, extract it, and open PowerShell inside the extracted project folder. Continue with step 3.

### 3. Create a Python virtual environment

```powershell
py -3.12 -m venv .venv
```

You do **not** need to activate the environment. The following commands call its Python executable directly, avoiding PowerShell execution-policy problems.

### 4. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Installation requires an internet connection and may download large packages, including CPU-only PyTorch. The versions are defined in `requirements.txt`.

### 5. Add your photographs

Copy your own pictures into the project folder:

```text
AssetLens/
├── app.py
├── requirements.txt
├── images/
│   ├── holiday-photo.jpg
│   └── landscape.png
└── ...
```

The repository intentionally keeps `images/` empty apart from a placeholder. **No photographs are downloaded automatically.** Your photos are excluded from Git by `.gitignore`.

### 6. Launch the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal (typically `http://localhost:8501`). Keep the terminal running while you use AssetLens.

### 7. Index your photographs

In the application's sidebar, open **Library**, then click **Index / Update images**. The first indexing run creates image embeddings; later runs reuse cached results for unchanged images.

The first use of CLIP may need internet access to download the pretrained model weights. After they are cached, inference runs locally on the CPU.

Now explore **Search**, **Similar Images**, **Duplicates**, and **Library**.

## Troubleshooting

**`py` is not recognized:** Install Python 3.12 with the Windows Python launcher, reopen the terminal, and retry. If you have Python 3.12 available through `python`, substitute `python` for `py -3.12` in the virtual-environment command.

**`git` is not recognized:** Install Git and reopen PowerShell, or use **Code → Download ZIP** on GitHub.

**PowerShell blocks environment activation:** Activation is unnecessary for this guide; run the full `\.venv\Scripts\python.exe` commands above.

**Dependencies fail to install:** Check that you are using Python 3.12 on Windows, upgrade pip as shown above, and copy the complete error output. The CPU PyTorch package comes from the additional PyTorch package index.

**The app opens but has no search results:** Add photographs to `images/`, then go to **Library → Index / Update images**. Search needs an indexed collection; Similar Images needs at least two indexed photographs.

**The model is slow to start:** First-time model downloading and loading can take longer than later searches; CPU performance depends on your computer.

**Port 8501 is in use:** Streamlit may offer another local port; use the URL shown in the terminal. You can also specify one manually:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8507
```

## Technical architecture

- **Vision-language embeddings:** OpenCLIP `ViT-B-32` with pretrained LAION-2B weights. Image and text inputs are mapped into a shared 512-dimensional embedding space.
- **Semantic retrieval:** Normalized embeddings are compared using cosine similarity and ranked to retrieve matching images. Similarity scores are **not** calibrated confidence percentages.
- **Similarity search:** A selected image embedding is compared against the indexed image collection.
- **Duplicate detection:** SHA-256 checks identical file content; perceptual dHash identifies potentially similar images and can produce false positives.
- **AI Collections:** K-Means groups images based on their existing embeddings; the number of clusters is user-selected.
- **Incremental index:** A local NumPy cache (`image_index.npz`) avoids recomputing embeddings for unchanged files.
- **Interface:** Streamlit, with custom styling in `assets/styles.css`.

**Model disclosure:** AssetLens integrates a **pretrained** model. The CLIP foundation model was **not trained or fine-tuned** as part of this project.

## Local benchmark results

Preliminary results measured on an **Intel Core i5-9400, 8 GB RAM, integrated graphics, CPU-only inference**, with a collection of **193 photographs**:

| Measurement | Observed result |
| --- | ---: |
| Indexed images | 193 |
| Average semantic search time across 3 searches, after model load | **39.3 ms** |
| CLIP model loading | 3.816 s |
| Incremental indexing, 168 new images while reusing 25 cached | 14.329 s |
| Fully cached indexing, 193 images | 0.016 s |
| Approximate final Python process memory | ~950 MB |
| Automated tests in the development environment | 7 passed |

These are **local development measurements**, not an independent accuracy evaluation or a performance guarantee for other machines. Search timings do not include model initialization. Retrieval relevance varies with the query and image collection.

## Tests and code quality

From the project folder, after installation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Benchmark scripts are included for repeatable local experiments, but timings depend on your machine and image collection.

## Project structure

```text
AssetLens/
├── app.py                    # Streamlit interface and navigation
├── ai_engine.py              # Pretrained CLIP model and embeddings
├── image_index.py            # Image discovery, indexing, and cache
├── duplicate_utils.py        # SHA-256 and dHash checks
├── search_dedup.py           # Duplicate-aware search results
├── clustering_engine.py      # K-Means clustering
├── cluster_browser.py        # AI Collections interface
├── metadata_utils.py         # Metadata extraction
├── metadata_browser.py       # Library exploration and filters
├── benchmark.py              # Semantic search benchmarking
├── benchmark_scalability.py  # Indexing and scalability tests
├── tests/                    # Automated tests
├── assets/                   # Logo and CSS
├── images/                   # Your local pictures (not committed)
├── .streamlit/               # Streamlit theme configuration
├── requirements.txt
└── README.md
```

## Limitations and privacy

- There is **no hosted public demo** and **no in-browser photo upload** in the current version; add files directly to the local `images/` folder.
- Users must provide their own photographs. Check your rights before using or sharing images.
- The model weights are downloaded separately on first use and are subject to their own applicable licenses.
- Duplicate matches based on perceptual hashes are approximate, and clustering depends on the chosen number of groups.
- The current benchmark collection has 193 images; results do not establish large-scale performance or general search accuracy.
- AssetLens operates locally for indexing and inference once its dependencies and model weights are installed. It is not a cloud storage service.

## Project purpose

AssetLens demonstrates applied AI engineering: pretrained model integration, embedding-based retrieval, CPU inference, incremental caching, clustering, duplicate detection, application development, automated testing, and performance measurement.

**Repository:** https://github.com/Mmzebedas/AssetLens
