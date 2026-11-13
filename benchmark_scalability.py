
import tempfile
import time
from pathlib import Path

import psutil

import image_index
from ai_engine import load_clip_model

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".jfif", ".webp", ".avif"
}


def memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(label, files, model_loader):
    start = time.perf_counter()

    processed, reused, errors = image_index.index_images(
        files,
        model_loader
    )

    elapsed = time.perf_counter() - start

    print(f"\n{label}")
    print("-" * 40)
    print(f"Images requested: {len(files)}")
    print(f"Processed: {processed}")
    print(f"Reused from cache: {reused}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.3f} seconds")
    print(f"Process RAM: {memory_mb():.2f} MB")

    for error in errors[:5]:
        print(f"Error: {error}")


def main():
    print("\nASSETLENS SCALABILITY BENCHMARK")
    print("=" * 40)

    files = sorted(
        p for p in image_index.IMAGE_FOLDER.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"Total photographs: {len(files)}")

    if len(files) < 2:
        print("Not enough photographs to benchmark.")
        return

    # Load CLIP only once
    print("\nLoading CLIP model...")

    start = time.perf_counter()
    model_data = load_clip_model()

    print(
        f"Model loading: "
        f"{time.perf_counter() - start:.3f} seconds"
    )

    def model_loader():
        return model_data

    # Use a completely separate temporary cache.
    # Our original image_index.npz stays untouched.
    original_cache = image_index.CACHE_FILE

    with tempfile.TemporaryDirectory() as folder:
        image_index.CACHE_FILE = (
            Path(folder) / "benchmark_index.npz"
        )

        try:
            initial_count = min(25, len(files))

            # Test 1: Index first 25 photographs
            run_benchmark(
                "TEST 1 - COLD INDEXING",
                files[:initial_count],
                model_loader,
            )

            # Test 2: Add remaining photographs
            run_benchmark(
                "TEST 2 - INCREMENTAL INDEXING",
                files,
                model_loader,
            )

            # Test 3: Reindex unchanged collection
            run_benchmark(
                "TEST 3 - FULLY CACHED",
                files,
                model_loader,
            )

        finally:
            image_index.CACHE_FILE = original_cache

    print("\nBENCHMARK COMPLETED")
    print("Original image index was not modified.")


if __name__ == "__main__":
    main()
