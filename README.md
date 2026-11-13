
# AssetLens

### AI-Powered Creative Asset Search and Organization

AssetLens is a local AI application for searching,
analyzing and organizing photographic collections.

It combines pretrained vision-language models,
vector similarity search and unsupervised machine
learning to help users explore their images.

## Features

- Natural-language image search using CLIP
- Image-to-image similarity retrieval
- Exact duplicate detection using SHA-256
- Near-duplicate detection using perceptual hashing
- Duplicate-aware semantic search
- Automatic image clustering using K-Means
- Image metadata exploration and filtering
- Incremental indexing and embedding caching
- Interactive interface built with Streamlit

## Technology Stack

- Python 3.12
- PyTorch (CPU inference)
- OpenCLIP
- NumPy
- scikit-learn
- Pillow
- Streamlit
- pytest
- Ruff

## AI Architecture

AssetLens uses the pretrained OpenCLIP ViT-B/32
model with LAION-2B weights.

The model transforms images and text into
512-dimensional embeddings.

Semantic similarity is calculated by comparing
normalized image and text vectors.

K-Means clustering uses the existing image
embeddings to organize photographs automatically.

The CLIP model was not trained by this project.

## Performance

Preliminary benchmarks on an Intel Core i5-9400
computer with 8 GB RAM and integrated graphics:

| Metric | Result |
|---|---|
| Indexed images | 193 |
| Mean search latency | 39.3 ms |
| CLIP model loading | 3.816 s |
| Cached indexing | 7 ms |
| Final Python process RAM | ~950 MB |
| Automated tests | 7 passed |

Results represent individual local benchmark runs
rather than comprehensive performance guarantees.

## Installation

Tested with Python 3.12 on Windows.

### 1. Clone the repository

After publication, clone the GitHub repository
or download the source code.

### 2. Create a virtual environment

    python -m venv .venv

### 3. Install dependencies

    .venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

### 4. Add photographs

Place your photographs inside the images/ folder.

Supported formats:
JPG, JPEG, PNG, JFIF, WebP and AVIF.

### 5. Start AssetLens

    .venv\Scripts\python.exe -m streamlit run app.py

Open the local address displayed by Streamlit.

### 6. Index the library

Navigate to Library and click:

Index / Update images

The first model initialization may require internet
access to download pretrained model weights.
Subsequent inference runs locally using cached weights.

## Testing

Run all automated tests:

    .venv\Scripts\python.exe -m pytest -q

Check code quality:

    .venv\Scripts\python.exe -m ruff check .

## Project Structure

- app.py: Main application
- ai_engine.py: CLIP model and embeddings
- image_index.py: Persistent embedding cache
- duplicate_utils.py: Duplicate detection
- search_dedup.py: Duplicate-aware ranking
- clustering_engine.py: K-Means clustering
- cluster_browser.py: Visual cluster explorer
- metadata_utils.py: Image metadata extraction
- metadata_browser.py: Smart Library interface
- benchmark.py: Search performance benchmark
- benchmark_scalability.py: Indexing benchmark
- tests/: Automated tests

## Limitations

- Retrieval quality depends on the pretrained model
  and the image collection.
- Clustering requires choosing the number of groups.
- Perceptual hashes can produce false matches.
- Current scalability testing covers 193 images.
- Search evaluation results are preliminary.
- Model weights and image assets have their own
  applicable licenses.

## Future Development

- Larger-scale performance evaluation
- Improved search result diversification
- Automated cluster-quality evaluation
- More extensive testing
- Alternative model comparisons

## Project Purpose

AssetLens demonstrates practical AI engineering,
including pretrained model integration, image
embeddings, vector search, caching, clustering,
software testing and CPU performance optimization.
