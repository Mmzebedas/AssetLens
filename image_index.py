
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ai_engine import EMBEDDING_DIM, MODEL_NAME, WEIGHTS, encode_image

# 1. File locations
BASE = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE / "images"
CACHE_FILE = BASE / "image_index.npz"


# 2. Load saved image embeddings
def load_index():

    if not CACHE_FILE.exists():
        return {}

    try:
        with np.load(CACHE_FILE, allow_pickle=False) as data:

            if data["model"].item() != MODEL_NAME:
                return {}

            if data["weights"].item() != WEIGHTS:
                return {}

            names = data["names"].tolist()
            sizes = data["sizes"].tolist()
            mtimes = data["mtimes"].tolist()
            vectors = data["vectors"]

            if (
                len(names) != len(sizes)
                or len(names) != len(mtimes)
                or vectors.shape != (len(names), EMBEDDING_DIM)
            ):
                return {}

            cache = {}

            for name, size, mtime, vector in zip(
                names, sizes, mtimes, vectors
            ):
                if np.isfinite(vector).all():
                    cache[name] = (
                        int(size),
                        int(mtime),
                        vector.astype(np.float32)
                    )

            return cache

    except (
        OSError,
        ValueError,
        EOFError,
        KeyError,
        IndexError,
        TypeError,
        BadZipFile,
    ) as error:
        print("Cache loading error:", error)
        return {}


# 3. Process new images and reuse cached ones
def index_images(files, model_loader):

    old_index = load_index()
    new_index = {}

    processed = 0
    reused = 0
    errors = []

    model = None
    preprocess = None

    for file in files:

        try:
            stat = file.stat()
            previous = old_index.get(file.name)

            # Check whether the image is unchanged
            if (
                previous is not None
                and previous[0] == stat.st_size
                and previous[1] == stat.st_mtime_ns
            ):

                vector = previous[2]
                reused += 1

            else:

                # Load CLIP only when needed
                if model is None:
                    model, preprocess, _ = model_loader()

                vector = encode_image(
                    model, preprocess, file
                )

                processed += 1

            new_index[file.name] = (
                stat.st_size,
                stat.st_mtime_ns,
                vector
            )

        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{file.name}: {error}")

    # 4. Save the updated index
    if new_index:

        if (
            processed > 0
            or set(new_index) != set(old_index)
            or not CACHE_FILE.exists()
        ):

            names = list(new_index)

            np.savez_compressed(
                CACHE_FILE,
                model=MODEL_NAME,
                weights=WEIGHTS,
                names=np.array(names),
                sizes=np.array(
                    [new_index[n][0] for n in names],
                    dtype=np.int64
                ),
                mtimes=np.array(
                    [new_index[n][1] for n in names],
                    dtype=np.int64
                ),
                vectors=np.stack(
                    [new_index[n][2] for n in names]
                ).astype(np.float32)
            )

    elif CACHE_FILE.exists():
        CACHE_FILE.unlink()

    return processed, reused, errors
