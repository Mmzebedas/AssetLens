
import numpy as np

import image_index


def test_image_cache(tmp_path, monkeypatch):

    # Create a temporary image file
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"image version one")

    # Use a temporary cache, not the real one
    cache_file = tmp_path / "test_cache.npz"

    monkeypatch.setattr(
        image_index,
        "CACHE_FILE",
        cache_file
    )

    model_loads = []

    # Simulate loading an AI model
    def fake_model_loader():
        model_loads.append(1)
        return object(), object(), None

    # Simulate generating an image embedding
    def fake_encode_image(model, preprocess, path):
        return np.ones(
            image_index.EMBEDDING_DIM,
            dtype=np.float32
        ) / np.sqrt(image_index.EMBEDDING_DIM)

    monkeypatch.setattr(
        image_index,
        "encode_image",
        fake_encode_image
    )

    # First execution: process the image
    processed, reused, errors = image_index.index_images(
        [image], fake_model_loader
    )

    assert processed == 1
    assert reused == 0
    assert errors == []
    assert cache_file.exists()

    # Second execution: reuse cached embedding
    processed, reused, errors = image_index.index_images(
        [image], fake_model_loader
    )

    assert processed == 0
    assert reused == 1
    assert errors == []

    # The AI model should only have loaded once
    assert len(model_loads) == 1

    # Modify the image
    image.write_bytes(b"image version two - modified")

    # Third execution: recalculate the embedding
    processed, reused, errors = image_index.index_images(
        [image], fake_model_loader
    )

    assert processed == 1
    assert reused == 0
    assert errors == []
