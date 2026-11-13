
# ---------------------------------
# ASSETLENS - RELIABILITY TESTS
# ---------------------------------

import numpy as np
from PIL import Image

import image_index


# Simulate the AI model to make tests fast
def fake_model_loader():
    return object(), object(), None


def fake_encode_image(model, preprocess, path):
    # Verify that the photograph is valid
    with Image.open(path) as img:
        img.verify()

    return np.ones(
        image_index.EMBEDDING_DIM,
        dtype=np.float32
    ) / np.sqrt(image_index.EMBEDDING_DIM)


# ---------------------------------
# TEST 1 - CORRUPTED IMAGE
# ---------------------------------

def test_corrupted_image(tmp_path, monkeypatch):

    cache = tmp_path / "cache.npz"

    monkeypatch.setattr(
        image_index, "CACHE_FILE", cache
    )

    monkeypatch.setattr(
        image_index, "encode_image", fake_encode_image
    )

    valid = tmp_path / "valid.jpg"
    corrupted = tmp_path / "corrupted.jpg"

    Image.new("RGB", (100, 100), "blue").save(valid)

    corrupted.write_bytes(b"invalid image data")

    processed, reused, errors = image_index.index_images(
        [valid, corrupted],
        fake_model_loader
    )

    assert processed == 1
    assert reused == 0
    assert len(errors) == 1

    saved = image_index.load_index()

    assert "valid.jpg" in saved
    assert "corrupted.jpg" not in saved


# ---------------------------------
# TEST 2 - DELETED IMAGE
# ---------------------------------

def test_deleted_image(tmp_path, monkeypatch):

    monkeypatch.setattr(
        image_index,
        "CACHE_FILE",
        tmp_path / "cache.npz"
    )

    monkeypatch.setattr(
        image_index, "encode_image", fake_encode_image
    )

    image1 = tmp_path / "image1.jpg"
    image2 = tmp_path / "image2.jpg"

    Image.new("RGB", (100, 100), "red").save(image1)
    Image.new("RGB", (100, 100), "green").save(image2)

    image_index.index_images(
        [image1, image2],
        fake_model_loader
    )

    # Simulate removing the second image
    image2.unlink()

    processed, reused, errors = image_index.index_images(
        [image1],
        fake_model_loader
    )

    saved = image_index.load_index()

    assert processed == 0
    assert reused == 1
    assert errors == []
    assert "image1.jpg" in saved
    assert "image2.jpg" not in saved


# ---------------------------------
# TEST 3 - CORRUPTED CACHE
# ---------------------------------

def test_corrupted_cache(tmp_path, monkeypatch):

    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"invalid cache content")

    monkeypatch.setattr(
        image_index, "CACHE_FILE", cache
    )

    monkeypatch.setattr(
        image_index, "encode_image", fake_encode_image
    )

    # The corrupted cache should be ignored
    assert image_index.load_index() == {}

    valid = tmp_path / "valid.jpg"
    Image.new("RGB", (100, 100), "yellow").save(valid)

    processed, reused, errors = image_index.index_images(
        [valid],
        fake_model_loader
    )

    assert processed == 1
    assert reused == 0
    assert errors == []
    assert "valid.jpg" in image_index.load_index()
