# ---------------------------------
# ASSETLENS — PROFESSIONAL DASHBOARD
# ---------------------------------

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from pathlib import Path
from ai_engine import EMBEDDING_DIM, encode_text, load_clip_model
from cluster_browser import render_ai_collections
from duplicate_utils import find_exact_duplicates, find_near_duplicates
from image_index import IMAGE_FOLDER, index_images, load_index
from metadata_browser import render_metadata_browser
from search_dedup import filter_duplicate_results

# 1. APPLICATION CONFIGURATION
st.set_page_config(
    page_title="AssetLens",
    page_icon="🔎",
    layout="wide",
)

# ASSETLENS PREMIUM STYLING
CSS_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"

if CSS_PATH.is_file():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# 2. SHARED HELPERS
@st.cache_resource
def load_model():
    """Load CLIP only when needed and reuse it across Streamlit reruns."""
    torch.set_num_threads(4)
    return load_clip_model()


@st.cache_data(show_spinner=False)
def create_thumbnail(image_path, modified_time):
    """Create a 3:2 thumbnail without changing the original file."""
    with Image.open(image_path) as img:
        corrected = ImageOps.exif_transpose(img)
        return ImageOps.fit(
            corrected.convert("RGB"),
            (420, 280),
            method=Image.Resampling.LANCZOS,
        )



# PREMIUM IMAGE GALLERY
def display_gallery(items, columns_count=3):

    if not items:
        st.info("No images to display.")
        return

    columns = st.columns(columns_count, gap="medium")

    for position, (file, score) in enumerate(items):

        with columns[position % columns_count]:

            with st.container(border=False, key=f"asset_card_{position}"):

                try:
                    thumbnail = create_thumbnail(
                        str(file),
                        file.stat().st_mtime_ns,
                    )

                    st.image(
                        thumbnail,
                        width="stretch",
                    )

                    st.markdown(f"**{file.name}**")

                    if score is not None:
                        st.markdown(
                             f"<span style='color:#F3C969;font-size:13px;'>"
                             f"✦ AI Similarity: {score:.4f}</span>",
                             unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Creative Asset")

                except (OSError, ValueError, RuntimeError) as error:
                    st.warning(
                        f"Could not preview {file.name}: {error}"
                    )

# 3. NAVIGATION AND HEADER
# FIND ALL SUPPORTED IMAGES
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".jfif", ".webp", ".avif"
}

files = sorted(
    file for file in IMAGE_FOLDER.iterdir()
    if file.is_file()
    and file.suffix.lower() in SUPPORTED_EXTENSIONS
)
# 3. NAVIGATION AND HEADER

LOGO_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "assetlens_logo.png"
)

if LOGO_PATH.is_file():
    st.sidebar.image(str(LOGO_PATH), width=170)
else:
    st.sidebar.error("AssetLens logo not found.")

st.sidebar.markdown(
    "<h2 style='color:#F3C969; text-align:center;'>AssetLens</h2>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Search", "Similar Images", "Duplicates", "Library"],
)


# 4. LIBRARY INDEXING — RUN BEFORE READING THE INDEX

# PREMIUM ASSETLENS HEADER
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #20190D, #101014);
        border: 1px solid #8B6A2D;
        border-radius: 18px;
        padding: 35px;
        margin-bottom: 30px;
    ">
        <p style="color:#D4A63A;letter-spacing:3px;">
            CREATIVE INTELLIGENCE PLATFORM
        </p>
        <h1 style="color:#F3C969;font-size:48px;">
            AssetLens
        </h1>
        <p style="color:#C1B9A7;font-size:16px;">
            Discover, search and organize your visual assets
            with artificial intelligence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Library":
    st.markdown("## Creative Asset Library")
    st.caption(
        "Manage, organize and explore your visual collection."
    )

    # Library statistics
    total_images = len(files)
    indexed_images = len(load_index())
    image_formats = len({
        file.suffix.lower() for file in files
    })

    metrics = st.columns(3)

    for column, label, value in zip(
        metrics,
        ["TOTAL ASSETS", "INDEXED IMAGES", "FILE FORMATS"],
        [total_images, indexed_images, image_formats],
    ):
        with column:
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#201B13,#15151A);
                    border:1px solid #69532D;
                    border-radius:15px;
                    padding:24px;
                    margin-bottom:20px;
                ">
                    <div style="
                        color:#BFAF8E;
                        font-size:12px;
                        letter-spacing:1.5px;
                    ">{label}</div>
                    <div style="
                        color:#F3C969;
                        font-size:36px;
                        font-weight:700;
                        margin-top:10px;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Index / Update images", type="primary"):
        with st.spinner("Indexing new or modified photographs..."):
            processed, reused, errors = index_images(files, load_model)

        st.success(
            f"Processed: {processed} | Cached: {reused} | "
            f"Errors: {len(errors)}"
        )
        for error in errors:
            st.warning(error)

    # NEW FEATURE: SMART LIBRARY EXPLORER
    st.divider()
    render_metadata_browser(IMAGE_FOLDER)
    st.divider()
    render_ai_collections()
# 5. LOAD VALID EMBEDDINGS
index = load_index()
valid_images = []

for file in files:
    entry = index.get(file.name)
    if entry is None:
        continue

    try:
        stat = file.stat()
        vector = np.asarray(entry[2], dtype=np.float32)
        if (
            entry[0] == stat.st_size
            and entry[1] == stat.st_mtime_ns
            and vector.shape == (EMBEDDING_DIM,)
            and np.isfinite(vector).all()
        ):
            valid_images.append((file, vector))
    except OSError:
        continue

st.sidebar.divider()
st.sidebar.metric("Images in folder", len(files))
st.sidebar.metric("Indexed images", len(valid_images))



# 6. NATURAL-LANGUAGE SEARCH
if page == "Search":
    st.subheader("Semantic Image Search")

    st.write(
        "Describe what you're looking for in natural language."
    )

    if len(valid_images) < len(files):
        st.info(
            "Some images are not indexed yet. "
            "Open Library to update the index."
        )

    # Search input
    query = st.text_input(
        "Search description",
        placeholder="A large white dog standing outdoors",
        key="semantic_query",
    )

    # Search settings
    st.markdown("#### Search Settings")

    hide_duplicates = st.checkbox(
        "Hide duplicate photographs",
        value=True,
        help=(
            "Remove visually identical or nearly identical "
            "photographs from search results."
        ),
    )

    if valid_images:
        max_results = st.slider(
            "Maximum number of results",
            min_value=1,
            max_value=min(30, len(valid_images)),
            value=min(5, len(valid_images)),
        )
    else:
        max_results = 5

    # Execute semantic search
    if query.strip():

        if not valid_images:
            st.warning(
                "Index your images in the Library section first."
            )

        else:
            with st.spinner("Searching with CLIP..."):

                # Load the pretrained CLIP model
                model, _, tokenizer = load_model()

                # Convert the query into a text embedding
                text_vector = encode_text(
                    model,
                    tokenizer,
                    query.strip(),
                )

                # Collect indexed image embeddings
                image_vectors = np.stack(
                    [
                        vector
                        for _, vector in valid_images
                    ]
                )

                # Calculate cosine similarity
                # Embeddings are already normalized
                scores = image_vectors @ text_vector

                # Rank all images by similarity
                ranking = np.argsort(-scores)

                ranked_results = [
                    (
                        valid_images[int(i)][0],
                        float(scores[i]),
                    )
                    for i in ranking
                ]

                # Apply duplicate filtering if enabled
                if hide_duplicates:

                    display_results = filter_duplicate_results(
                        ranked_results,
                        max_results=max_results,
                        threshold=3,
                    )

                else:
                    display_results = ranked_results[
                        :max_results
                    ]

            # Display search results
            st.divider()

            st.subheader("Search Results")

            st.caption(
                "Similarity scores are not confidence percentages."
            )

            st.write(
                f"Showing {len(display_results)} "
                f"of {len(ranked_results)} indexed photographs."
            )

            if hide_duplicates:
                st.caption(
                    "Duplicate filtering is enabled."
                )

            if display_results:
                display_gallery(
                    display_results,
                    columns_count=3,
                )
            else:
                st.info(
                    "No matching photographs were available."
                )

    else:
        st.info(
            "Enter a description to search "
            "your indexed photographs."
        )



# 7. IMAGE-TO-IMAGE SIMILARITY
elif page == "Similar Images":
    st.subheader("Find Similar Images")
    st.write("Select a reference photograph to find visually related images.")

    if len(valid_images) < 2:
        st.info("Index at least two images in Library first.")
    else:
        image_names = [file.name for file, _ in valid_images]
        selected_name = st.selectbox("Reference image", image_names)
        selected_index = image_names.index(selected_name)
        selected_file, selected_vector = valid_images[selected_index]

        st.image(
            create_thumbnail(
                str(selected_file), selected_file.stat().st_mtime_ns
            ),
            width=300,
            caption=f"Reference: {selected_name}",
        )

        if st.button("Find Similar Images", type="primary"):
            image_vectors = np.stack(
                [vector for _, vector in valid_images]
            )
            similarities = image_vectors @ selected_vector
            ranking = [
                int(i) for i in np.argsort(-similarities)
                if int(i) != selected_index
            ]
            results = [
                (valid_images[i][0], float(similarities[i]))
                for i in ranking
            ]
            st.subheader("Similar Images")
            display_gallery(results)


# 8. DUPLICATE DETECTION
elif page == "Duplicates":
    st.subheader("Duplicate Detection")
    st.write("Detect identical files and photographs with similar visual structure.")

    exact_tab, near_tab = st.tabs(["Exact duplicates", "Near duplicates"])

    with exact_tab:
        st.caption("SHA-256 finds files with matching binary contents.")
        if st.button("Find Exact Duplicates"):
            groups, errors = find_exact_duplicates(files)
            for filename, error in errors:
                st.warning(f"{filename}: {error}")

            if groups:
                st.warning(f"Found {len(groups)} exact duplicate groups.")
                for group_number, group in enumerate(groups, start=1):
                    with st.expander(f"Group {group_number}: {len(group)} files"):
                        display_gallery([(file, None) for file in group])
            else:
                st.success("No exact duplicates found.")

    with near_tab:
        st.caption("dHash finds potentially similar photographs, even if compressed or resized.")
        threshold = st.slider(
            "Maximum perceptual hash distance",
            min_value=0,
            max_value=15,
            value=6,
        )

        if st.button("Find Near Duplicates"):
            with st.spinner("Comparing image fingerprints..."):
                matches, errors = find_near_duplicates(files, threshold)

            for filename, error in errors:
                st.warning(f"{filename}: {error}")

            if matches:
                st.success(f"Found {len(matches)} potentially similar pairs.")
                for file1, file2, distance in matches:
                    with st.expander(
                        f"{file1.name} ↔ {file2.name} | Distance: {distance}"
                    ):
                        left, right = st.columns(2)
                        with left:
                            st.image(str(file1), width=300)
                            st.caption(file1.name)
                        with right:
                            st.image(str(file2), width=300)
                            st.caption(file2.name)
                        if distance == 0:
                            st.info("Identical perceptual fingerprints (not necessarily identical files).")
                        else:
                            st.info(f"Perceptual distance: {distance}")
            else:
                st.info("No near duplicates found at this threshold.")


# 9. COMPLETE LIBRARY GALLERY
elif page == "Library":
    left, right = st.columns(2)
    left.metric("Images in folder", len(files))
    right.metric("Indexed images", len(valid_images))

    if not files:
        st.info("Add JPEG or PNG photographs to the images folder.")
    else:
        st.subheader("Complete Image Gallery")
        display_gallery([(file, None) for file in files])
