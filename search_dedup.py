
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@lru_cache(maxsize=10000)
def image_fingerprint(path_string, file_size, modified_time):
    """
    Calculate a perceptual hash for an image.
    File size and modification time help invalidate the cache.
    """
    with Image.open(path_string) as image:
        image = ImageOps.exif_transpose(image)

        aspect_ratio = image.width / image.height

        image = image.convert("L").resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(image)

    differences = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0

    for bit in differences.flat:
        hash_value = (hash_value << 1) | int(bit)

    return hash_value, aspect_ratio


def filter_duplicate_results(
    ranked_results,
    max_results=5,
    threshold=3
):
    """
    ranked_results: list of (image_path, similarity_score)
    ordered from highest to lowest similarity.

    Returns the best results without obvious duplicates.
    """
    selected = []
    fingerprints = []

    for image_path, similarity_score in ranked_results:
        path = Path(image_path)

        try:
            stat = path.stat()

            image_hash, aspect_ratio = image_fingerprint(
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns
            )

        except (OSError, ValueError):
            continue

        duplicate = False

        for previous_hash, previous_ratio in fingerprints:
            distance = (
                image_hash ^ previous_hash
            ).bit_count()

            if (
                distance <= threshold
                and abs(aspect_ratio - previous_ratio) < 0.05
            ):
                duplicate = True
                break

        if duplicate:
            continue

        selected.append((path, similarity_score))
        fingerprints.append(
            (image_hash, aspect_ratio)
        )

        if len(selected) >= max_results:
            break

    return selected
