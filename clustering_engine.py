
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from ai_engine import EMBEDDING_DIM
from image_index import IMAGE_FOLDER, load_index


def cluster_images(n_clusters=8):
    # Load the CLIP embeddings already saved by AssetLens
    index = load_index()

    valid_images = []

    for filename, record in sorted(index.items()):
        # Skip images no longer present in the library
        if not (IMAGE_FOLDER / filename).is_file():
            continue

        # Each cached record stores:
        # (file_size, modification_time, embedding)
        vector = np.asarray(record[2], dtype=np.float32)

        if vector.shape != (EMBEDDING_DIM,):
            continue

        if not np.isfinite(vector).all():
            continue

        norm = np.linalg.norm(vector)

        if norm == 0:
            continue

        # Normalize embeddings to unit length
        valid_images.append((filename, vector / norm))

    if len(valid_images) < 2:
        raise ValueError("Not enough valid images to cluster.")

    n_clusters = min(n_clusters, len(valid_images))

    filenames = [name for name, _ in valid_images]
    vectors = np.stack(
        [vector for _, vector in valid_images]
    )

    # Run K-Means on our existing image embeddings
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(vectors)

    groups = defaultdict(list)

    for filename, vector, label in zip(
        filenames, vectors, labels
    ):
        # Measure distance to the group's center
        distance = np.linalg.norm(
            vector - model.cluster_centers_[label]
        )

        groups[int(label)].append((filename, distance))

    # Show the most representative images first
    return {
        group_id: [
            filename
            for filename, _ in sorted(
                members,
                key=lambda item: item[1]
            )
        ]
        for group_id, members in sorted(groups.items())
    }


if __name__ == "__main__":
    groups = cluster_images(n_clusters=8)

    print("\nASSETLENS - AUTOMATIC IMAGE CLUSTERING")
    print("-------------------------------------")

    for group_id, filenames in groups.items():
        print(f"\nGroup {group_id + 1}")
        print(f"Images: {len(filenames)}")

        for filename in filenames:
            print(f"  - {filename}")
