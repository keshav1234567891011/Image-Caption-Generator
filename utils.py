"""Small helpers for opening uploads and presenting generated captions."""

import re
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError


def open_uploaded_image(upload: BinaryIO) -> Image.Image:
    """Fully decode an upload, correct camera orientation, and convert to RGB."""
    try:
        with Image.open(upload) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError) as error:
        raise ValueError(
            "We couldn't open this image. Please upload a valid JPG, JPEG, PNG, or WEBP file."
        ) from error


def format_caption(caption: str) -> str:
    """Remove sequence markers and format a sentence without rewriting model words."""
    caption = re.sub(r"\b(?:startseq|endseq)\b", "", caption, flags=re.IGNORECASE)
    caption = " ".join(caption.split()).strip()
    if not caption:
        return ""
    caption = caption[0].upper() + caption[1:]
    if caption[-1] not in ".!?":
        caption += "."
    return caption
