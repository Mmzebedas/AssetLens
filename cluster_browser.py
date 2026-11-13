
import streamlit as st

from clustering_engine import cluster_images
from image_index import IMAGE_FOLDER
from metadata_browser import create_metadata_thumbnail


def render_ai_collections():
    st.subheader("AI Collections")

    st.write(
        "Automatically organize photographs using "
        "CLIP embeddings and K-Means clustering."
    )

    n_clusters = st.slider(
        "Number of AI groups",
        min_value=2,
        max_value=15,
        value=9,
    )

    if st.button("Generate AI Groups", type="primary"):
        with st.spinner("Analyzing image relationships..."):
            try:
                groups = cluster_images(
                    n_clusters=n_clusters
                )
                st.session_state["ai_groups"] = groups
            except ValueError as error:
                st.error(str(error))
                return

    groups = st.session_state.get("ai_groups")

    if not groups:
        st.info("Click Generate AI Groups to begin.")
        return

    total = sum(len(images) for images in groups.values())

    st.success(
        f"Organized {total} images into "
        f"{len(groups)} AI groups."
    )

    for group_id, filenames in sorted(groups.items()):

        with st.expander(
            f"Group {group_id + 1} — "
            f"{len(filenames)} images",
            expanded=False,
        ):

            columns = st.columns(4)

            for i, filename in enumerate(filenames):
                path = IMAGE_FOLDER / filename

                if not path.is_file():
                    continue

                with columns[i % 4]:
                    thumbnail = create_metadata_thumbnail(
                        str(path),
                        path.stat().st_mtime_ns,
                    )

                    st.image(
                        thumbnail,
                        width="stretch",
                    )

                    st.caption(filename)
