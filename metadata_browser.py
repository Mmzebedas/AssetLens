
import math
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from metadata_utils import scan_library


@st.cache_data(show_spinner=False)
def create_metadata_thumbnail(path, modified_time):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return ImageOps.fit(
            image.convert("RGB"),
            (350, 230),
            method=Image.Resampling.LANCZOS,
        )


def render_metadata_browser(folder):
    records, errors = scan_library(folder)

    st.subheader("Smart Library Explorer")

    if errors:
        st.warning(
            f"{len(errors)} images could not be analyzed."
        )

    if not records:
        st.info("No supported photographs found.")
        return

   
    # 1. Filter controls
    formats = sorted({r["format"] for r in records})

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_formats = st.multiselect(
            "Image format",
            options=formats,
            default=formats,
            key="image_format_filter_v2",
        )

    with col2:

        orientation = st.selectbox(
            "Orientation",
            ["Any", "Landscape", "Portrait", "Square"],
        )

    with col3:
        min_width = st.number_input(
            "Minimum width (pixels)",
            min_value=0,
            value=0,
            step=100,
        )

    sort_by = st.selectbox(
        "Sort images by",
        [
            "Filename",
            "Largest resolution",
            "Largest file size",
        ],
    )

    # 2. Apply filters
    filtered = [
        record for record in records
        if record["format"] in selected_formats
        and (
            orientation == "Any"
            or record["orientation"] == orientation
        )
        and record["width"] >= min_width
    ]

    # 3. Sort results
    if sort_by == "Filename":
        filtered.sort(key=lambda r: r["filename"].lower())

    elif sort_by == "Largest resolution":
        filtered.sort(
            key=lambda r: r["width"] * r["height"],
            reverse=True,
        )

    elif sort_by == "Largest file size":
        filtered.sort(
            key=lambda r: r["size_mb"],
            reverse=True,
        )

    st.caption(
        f"Showing {len(filtered)} of {len(records)} images"
    )

    if not filtered:
        st.info("No images match the selected filters.")
        return

    # 4. Paginate for larger libraries
    page_size = 12
    total_pages = math.ceil(len(filtered) / page_size)

    page = st.selectbox(
        "Page",
        options=list(range(1, total_pages + 1)),
    )

    start = (page - 1) * page_size
    visible = filtered[start:start + page_size]

    # 5. Display the image gallery
    for offset in range(0, len(visible), 4):
        columns = st.columns(4)

        for column, record in zip(
            columns, visible[offset:offset + 4]
        ):
            path = Path(record["path"])

            with column:
                thumbnail = create_metadata_thumbnail(
                    str(path),
                    path.stat().st_mtime_ns,
                )

                st.image(thumbnail, width="stretch")

                st.write(f"**{record['filename']}**")

                st.caption(
                    f"{record['width']} × {record['height']} px"
                    f" | {record['format']}"
                    f" | {record['size_mb']} MB"
                )
