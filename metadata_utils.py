
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".jfif", ".webp", ".avif"
}


def get_image_metadata(image_path):
    path = Path(image_path)

    with Image.open(path) as image:
        width, height = image.size

        # Account for photographs with EXIF rotation.
        orientation = image.getexif().get(274, 1)

        if orientation in (5, 6, 7, 8):
            width, height = height, width

       

    if width > height:
        layout = "Landscape"
    elif height > width:
        layout = "Portrait"
    else:
        layout = "Square"

    return {
        "filename": path.name,
        "path": str(path),
        "format": path.suffix.lstrip(".").upper(),
        "width": width,
        "height": height,
        "orientation": layout,
        "size_mb": round(
            path.stat().st_size / (1024 * 1024), 3
        )
    }


def scan_library(folder):
    folder = Path(folder)

    records = []
    errors = []

    for path in sorted(folder.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue

        try:
            records.append(get_image_metadata(path))
        except (OSError, ValueError) as error:
            errors.append((path.name, str(error)))

    return records, errors


if __name__ == "__main__":
    records, errors = scan_library("images")

    print(f"Images analyzed: {len(records)}")
    print(f"Errors: {len(errors)}")

    for record in records[:5]:
        print(record)
