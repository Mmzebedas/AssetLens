
# ---------------------------------
# ASSETLENS - PERFORMANCE BENCHMARK
# ---------------------------------

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from ai_engine import encode_image, encode_text, load_clip_model
from image_index import IMAGE_FOLDER, index_images, load_index

# 1. Performance measurement tools
process = psutil.Process()


def memory_mb():
    return round(
        process.memory_info().rss / (1024 ** 2), 2
    )


def elapsed(start):
    return round(time.perf_counter() - start, 3)


# 2. Prepare the experiment
torch.set_num_threads(4)


files = sorted(
    p for p in IMAGE_FOLDER.iterdir()
    if p.is_file()
    and p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
        ".webp",
        ".avif"
    }
)
if not files:
    raise SystemExit("No images found.")

results = {
    "timestamp": datetime.now().isoformat(),
    "image_count": len(files)
}

print("\nASSETLENS PERFORMANCE BENCHMARK")
print("--------------------------------")
print(f"Images found: {len(files)}")

results["initial_memory_mb"] = memory_mb()


# 3. Measure model loading
print("\nLoading CLIP model...")

start = time.perf_counter()
model, preprocess, tokenizer = load_clip_model()

results["model_load_seconds"] = elapsed(start)
results["model_memory_mb"] = memory_mb()

print("Model loaded!")
print("Time:", results["model_load_seconds"], "s")
print("RAM:", results["model_memory_mb"], "MB")


# 4. Measure image processing
print("\nProcessing one image...")

start = time.perf_counter()

image_vector = encode_image(
    model, preprocess, files[0]
)

results["single_image_seconds"] = elapsed(start)

print("Time:", results["single_image_seconds"], "s")


# 5. Measure indexing
print("\nTesting image indexing...")

def model_loader():
    return model, preprocess, tokenizer


start = time.perf_counter()

processed, reused, errors = index_images(
    files, model_loader
)

results["indexing_seconds"] = elapsed(start)
results["images_processed"] = processed
results["images_reused"] = reused

print("Indexing time:", results["indexing_seconds"], "s")
print("Processed:", processed)
print("Reused:", reused)
print("Errors:", len(errors))


# 6. Measure warm-cache performance
print("\nTesting cached indexing...")

start = time.perf_counter()

processed, reused, errors = index_images(
    files, model_loader
)

results["cached_index_seconds"] = elapsed(start)

print("Cached indexing:", results["cached_index_seconds"], "s")
print("Reused:", reused)


# 7. Prepare search vectors
cache = load_index()

names = []
vectors = []

for file in files:
    entry = cache.get(file.name)

    if entry is None:
        continue

    stat = file.stat()

    if (
        entry[0] == stat.st_size
        and entry[1] == stat.st_mtime_ns
    ):
        names.append(file.name)
        vectors.append(entry[2])

if not vectors:
    raise SystemExit("No valid cached embeddings.")

image_matrix = np.stack(vectors)


# 8. Measure semantic searches
queries = [
    "A large white dog outdoors",
    "A modern tall building",
    "A blue ocean with waves"
]

results["searches"] = []

print("\nTesting semantic search...")

for query in queries:

    start = time.perf_counter()

    text_vector = encode_text(
        model, tokenizer, query
    )

    scores = image_matrix @ text_vector
    best_index = int(np.argmax(scores))

    search_time = elapsed(start)

    results["searches"].append({
        "query": query,
        "time_seconds": search_time,
        "best_match": names[best_index]
    })

    print(f"\nQuery: {query}")
    print(f"Time: {search_time} s")
    print(f"Best match: {names[best_index]}")


# 9. Final measurements
results["final_memory_mb"] = memory_mb()

output = Path(__file__).parent / "benchmark_results.json"

with open(output, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n--------------------------------")
print("BENCHMARK COMPLETED")
print("Final RAM:", results["final_memory_mb"], "MB")
print("Results saved to:", output.name)
