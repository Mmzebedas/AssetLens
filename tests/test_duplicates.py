
from PIL import Image

from duplicate_utils import (
    calculate_dhash,
    calculate_hash,
    find_exact_duplicates,
    find_near_duplicates,
    hamming_distance,
)


# TEST 1: Exact duplicate detection
def test_exact_duplicates(tmp_path):

    original = tmp_path / "original.jpg"
    copy = tmp_path / "copy.jpg"
    different = tmp_path / "different.jpg"

    original.write_bytes(b"same image data")
    copy.write_bytes(b"same image data")
    different.write_bytes(b"different image data")

    groups, errors = find_exact_duplicates(
        [original, copy, different]
    )

    assert errors == []
    assert len(groups) == 1
    assert set(groups[0]) == {original, copy}


# TEST 2: Perceptual duplicate detection
def test_perceptual_duplicates(tmp_path):

    image1 = tmp_path / "image1.png"
    image2 = tmp_path / "image2.jpg"

    # Same picture, different file formats
    img = Image.new("RGB", (128, 128), "white")
    img.save(image1)
    img.save(image2)

    # Files have different binary contents
    assert calculate_hash(image1) != calculate_hash(image2)

    # But their visual fingerprints should match
    assert calculate_dhash(image1) == calculate_dhash(image2)

    matches, errors = find_near_duplicates(
        [image1, image2],
        threshold=0
    )

    assert errors == []
    assert len(matches) == 1


# TEST 3: Hamming distance calculation
def test_hamming_distance():

    number1 = 0b1010
    number2 = 0b0011

    assert hamming_distance(number1, number2) == 2
