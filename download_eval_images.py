r"""Create a small, credited local photo dataset for AssetLens evaluation.

Usage (from the AssetLens folder):
    .\.venv\Scripts\python.exe download_eval_images.py
    .\.venv\Scripts\python.exe download_eval_images.py --per-category 15

This educational script retrieves Wikimedia Commons thumbnails; read each
image's source page and licensing terms before publishing it elsewhere.
"""

import argparse
import csv
import html
import io
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
DEST = BASE / "images"
MANIFEST = BASE / "evaluation_sources.csv"
API = "https://commons.wikimedia.org/w/api.php"

CATEGORIES = {
    "dogs": "dog outdoors photograph",
    "cats": "cat animal photograph",
    "mountains": "mountain landscape photograph",
    "oceans": "ocean waves photograph",
    "forests": "forest trees photograph",
    "buildings": "modern building architecture photograph",
    "cars": "car vehicle photograph",
    "flowers": "flower garden photograph",
    "birds": "wild bird photograph",
    "food": "food plate meal photograph",
}

FIELDS = ["filename", "category", "file_title", "author", "license", "source_page"]


def plain_text(value):
    return html.unescape(re.sub(r"<[^>]*>", "", value or "")).strip()


def request_bytes(url, user_agent, max_bytes=12_000_000):
    req = Request(url, headers={"User-Agent": user_agent})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=35) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Image download exceeds size limit")
            return data
        except HTTPError as error:
            if error.code in (429, 503) and attempt < 2:
                delay = error.headers.get("Retry-After", "5")
                time.sleep(min(60, int(delay) if delay.isdigit() else 5))
                continue
            raise
    raise RuntimeError("Request failed repeatedly")


def api_page(term, continuation, user_agent):
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "generator": "search", "gsrsearch": term, "gsrnamespace": "6",
        "gsrlimit": "15", "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "700", "iiextmetadatafilter": "LicenseShortName|Artist",
    }
    if continuation:
        params["gsrcontinue"] = continuation
    data = request_bytes(API + "?" + urlencode(params), user_agent, 4_000_000)
    return json.loads(data)


def allowed_license(label):
    normalized = label.casefold()
    return (
        normalized.startswith("cc by")
        or normalized.startswith("cc0")
        or normalized.startswith("public domain")
        or normalized == "pd"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.per_category <= 25:
        parser.error("--per-category must be between 1 and 25")

    print("Wikimedia Commons requires an identifiable client for API requests.")
    print("Your contact is sent in the User-Agent header, not saved in the dataset.")
    contact = input("Enter your public GitHub profile URL or contact email: ").strip()
    if not contact or ("@" not in contact and not contact.startswith("https://")):
        raise SystemExit("Please enter a valid contact email or public profile URL.")
    user_agent = f"AssetLensEducationBot/0.1 ({contact}) Python-urllib/3"

    DEST.mkdir(exist_ok=True)
    manifest_exists = MANIFEST.exists()
    with MANIFEST.open("a", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=FIELDS)
        if not manifest_exists:
            writer.writeheader()

        for category, search_term in CATEGORIES.items():
            existing = len(list(DEST.glob(f"eval_{category}_*.jpg")))
            count = existing
            continuation = None
            visited = set()
            print(f"\n{category}: {count}/{args.per_category}")

            for _page in range(6):
                if count >= args.per_category:
                    break
                try:
                    payload = api_page(search_term, continuation, user_agent)
                except (HTTPError, URLError, ValueError, json.JSONDecodeError) as error:
                    print(f"  Search error: {error}")
                    break

                pages = payload.get("query", {}).get("pages", [])
                for page in pages:
                    if count >= args.per_category:
                        break
                    info = (page.get("imageinfo") or [{}])[0]
                    url = info.get("thumburl")
                    if not url or info.get("mime") not in ("image/jpeg", "image/png"):
                        continue
                    metadata = info.get("extmetadata", {})
                    license_label = plain_text(metadata.get("LicenseShortName", {}).get("value", ""))
                    if not allowed_license(license_label):
                        continue
                    source_page = info.get("descriptionurl", "")
                    if source_page in visited:
                        continue
                    visited.add(source_page)
                    try:
                        data = request_bytes(url, user_agent)
                        with Image.open(io.BytesIO(data)) as image:
                            image = ImageOps.exif_transpose(image).convert("RGB")
                            image.thumbnail((700, 700), Image.Resampling.LANCZOS)
                            buffer = io.BytesIO()
                            image.save(buffer, format="JPEG", quality=86)
                        filename = f"eval_{category}_{count + 1:02d}.jpg"
                        (DEST / filename).write_bytes(buffer.getvalue())
                        writer.writerow({
                            "filename": filename, "category": category,
                            "file_title": page.get("title", ""),
                            "author": plain_text(metadata.get("Artist", {}).get("value", "")),
                            "license": license_label, "source_page": source_page,
                        })
                        manifest.flush()
                        count += 1
                        print(f"  Saved {filename}")
                        time.sleep(0.5)
                    except (HTTPError, URLError, OSError, ValueError, UnidentifiedImageError) as error:
                        print(f"  Skipped image: {error}")

                continuation = payload.get("continue", {}).get("gsrcontinue")
                if not continuation:
                    break
                time.sleep(1)
            if count < args.per_category:
                print(f"  Only {count} files available; you can rerun later.")

    total = len(list(DEST.glob("eval_*.jpg")))
    print(f"\nFinished. Evaluation images: {total}")
    print(f"Attribution details: {MANIFEST.name}")
    print("Review downloaded images manually before accuracy evaluation.")
    print("Do not publish the downloaded photos without checking their licenses.")


if __name__ == "__main__":
    main()
