"""Streamlit interface for VisionScribe."""

from hashlib import sha256
from io import BytesIO
import logging

import streamlit as st

from caption_generator import CaptionGenerationError, generate_caption
from utils import open_uploaded_image

LOGGER = logging.getLogger(__name__)


def clear_upload() -> None:
    """Reset this session's upload and result while keeping shared models cached."""
    st.session_state.upload_number += 1
    st.session_state.pop("image_id", None)
    st.session_state.pop("caption", None)


def main() -> None:
    """Render the upload, generation, and result workflow."""
    st.set_page_config(page_title="VisionScribe – AI Image Caption Generator",
                       page_icon="✍️", layout="centered")
    st.markdown(
        """<style>
        .block-container {max-width: 860px; padding-top: 2.5rem;}
        h1 {letter-spacing: -0.04em;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
        </style>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("About the Model")
        st.markdown("**Image Encoder:** VGG16\n\n"
                    "**Language Model:** LSTM\n\n"
                    "**Dataset:** Flickr8k\n\n"
                    "**Task:** Image Caption Generation")
        st.divider()
        st.markdown("**How it works**")
        st.write("Image → VGG16 features → LSTM decoder → Caption")
        st.caption("VGG16 extracts visual features. The LSTM uses them to "
                   "predict a description one word at a time.")

    st.title("VisionScribe")
    st.subheader("AI-Powered Image Caption Generator")
    st.write("Upload an image and let our deep learning model generate a "
             "natural-language description.")

    if "upload_number" not in st.session_state:
        st.session_state.upload_number = 0

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"],
            key=f"image_upload_{st.session_state.upload_number}",
            help="Supported formats: JPG, JPEG, PNG, and WEBP.",
        )
        image = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_id = sha256(image_bytes).hexdigest()
            if st.session_state.get("image_id") != image_id:
                st.session_state.image_id = image_id
                st.session_state.pop("caption", None)
            try:
                image = open_uploaded_image(BytesIO(image_bytes))
            except (OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.image(image, caption="Your uploaded image", use_container_width=True)
        else:
            st.session_state.pop("image_id", None)
            st.session_state.pop("caption", None)
            st.caption("Start with a photo of people, animals, or an everyday scene.")

        generate_column, clear_column = st.columns([2, 1])
        generate = generate_column.button(
            "Generate Caption", type="primary", disabled=image is None,
            use_container_width=True,
        )
        clear_column.button(
            "Clear / upload another image", on_click=clear_upload,
            disabled=uploaded_file is None, use_container_width=True,
        )

    if generate and image is not None:
        st.session_state.pop("caption", None)
        with st.spinner("Analyzing image and generating caption..."):
            try:
                st.session_state.caption = generate_caption(image)
            except CaptionGenerationError as error:
                st.error(str(error))
            except Exception:
                LOGGER.exception("Unexpected caption generation failure")
                st.error("Something went wrong while generating the caption. "
                         "Please try again or upload another image.")

    if st.session_state.get("caption"):
        with st.container(border=True):
            st.subheader("Generated Caption")
            st.write(st.session_state.caption)
        st.caption("AI-generated descriptions may miss details or make mistakes.")


if __name__ == "__main__":
    main()
