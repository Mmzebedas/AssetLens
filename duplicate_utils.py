
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image


# 1. Calculate SHA-256 for exact duplicates
def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# 2. Calculate perceptual hash (dHash)
def calculate_dhash(file_path):
    with Image.open(file_path) as img:
        img = img.convert("L")
        img = img.resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        pixels = np.asarray(img)

    differences = pixels[:, :-1] > pixels[:, 1:]

    hash_value = 0

    for bit in differences.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


# 3. Calculate Hamming distance
def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


# 4. Find exact duplicate groups
def find_exact_duplicates(files):
    hashes = defaultdict(list)
    errors = []

    for file in files:
        try:
            file_hash = calculate_hash(file)
            hashes[file_hash].append(file)
        except OSError as error:
            errors.append((file.name, str(error)))

    groups = [
        group for group in hashes.values()
        if len(group) > 1
    ]

    return groups, errors


# 5. Find visually similar pairs
def find_near_duplicates(files, threshold=6):
    image_hashes = {}
    errors = []

    for file in files:
        try:
            image_hashes[file] = calculate_dhash(file)
        except OSError as error:
            errors.append((file.name, str(error)))

    image_list = list(image_hashes)
    matches = []

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):

            file1 = image_list[i]
            file2 = image_list[j]

            distance = hamming_distance(
                image_hashes[file1],
                image_hashes[file2]
            )

            if distance <= threshold:
                matches.append((file1, file2, distance))

    matches.sort(key=lambda item: item[2])

    return matches, errors
