"""Cached model loading and inference for the original VGG16 + LSTM model."""

import logging
from pathlib import Path
import pickle

from keras_preprocessing.text import Tokenizer
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import format_caption

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_model.h5"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer.pkl"
IMAGE_SIZE = (224, 224)
FEATURE_SIZE = 4096
MAX_CAPTION_LENGTH = 35
START_TOKEN = "startseq"
END_TOKEN = "endseq"
LOGGER = logging.getLogger(__name__)


class CaptionGenerationError(Exception):
    """An inference or model-loading problem that can be explained to the user."""


@st.cache_resource(show_spinner=False)
def load_caption_model() -> Model:
    """Load the saved H5 decoder once, without restoring training state."""
    if not MODEL_PATH.is_file():
        raise CaptionGenerationError(
            "The caption model is missing. Please add best_model.h5 to the project folder."
        )
    try:
        model = load_model(str(MODEL_PATH), compile=False)
        if model.input_shape != [(None, FEATURE_SIZE), (None, MAX_CAPTION_LENGTH)]:
            raise ValueError("Unexpected decoder input shapes")
        return model
    except Exception as error:
        LOGGER.exception("Could not load the caption model")
        raise CaptionGenerationError(
            "The caption model couldn't be loaded. Please check the model file "
            "and use the project's required package versions."
        ) from error


@st.cache_resource(show_spinner=False)
def load_tokenizer() -> Tokenizer:
    """Load the original trusted vocabulary once; never recreate or refit it."""
    if not TOKENIZER_PATH.is_file():
        raise CaptionGenerationError(
            "The tokenizer is missing. Please add tokenizer.pkl to the project folder."
        )
    try:
        with TOKENIZER_PATH.open("rb") as file:
            tokenizer = pickle.load(file)
        if not tokenizer.word_index.get(START_TOKEN):
            raise ValueError("Tokenizer has no start token")
        return tokenizer
    except Exception as error:
        LOGGER.exception("Could not load the tokenizer")
        raise CaptionGenerationError(
            "The tokenizer couldn't be loaded. Please use the original tokenizer.pkl "
            "and install the project's required packages."
        ) from error


@st.cache_resource(show_spinner=False)
def load_image_encoder() -> Model:
    """Cache ImageNet VGG16 with its 4096-value fc2 output, as used in training."""
    try:
        vgg16 = VGG16(weights="imagenet", include_top=True)
        return Model(inputs=vgg16.inputs, outputs=vgg16.get_layer("fc2").output)
    except Exception as error:
        LOGGER.exception("Could not load VGG16")
        raise CaptionGenerationError(
            "The image encoder couldn't be loaded. On first use, VGG16 weights "
            "must be downloaded. Please check the connection and try again."
        ) from error


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Create a float32 VGG16 batch using the notebook's nearest-neighbor resize."""
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    image_array = np.asarray(image, dtype=np.float32)
    return preprocess_input(np.expand_dims(image_array, axis=0))


def predict_caption(model: Model, features: np.ndarray, tokenizer: Tokenizer) -> str:
    """Greedily decode words with the original pre-padding and length limit."""
    words = [START_TOKEN]
    index_to_word = getattr(tokenizer, "index_word", None) or {
        index: word for word, index in tokenizer.word_index.items()
    }
    for _ in range(MAX_CAPTION_LENGTH):
        sequence = tokenizer.texts_to_sequences([" ".join(words)])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LENGTH,
                                 padding="pre", truncating="pre")
        prediction = model.predict([features, sequence], verbose=0)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Decoder returned invalid probabilities")
        word = index_to_word.get(int(np.argmax(prediction)))
        if word is None or word == END_TOKEN:
            break
        if word == START_TOKEN:
            break
        words.append(word)
    return format_caption(" ".join(words))


def generate_caption(image: Image.Image) -> str:
    """Extract image features and generate a display-ready caption."""
    model = load_caption_model()
    tokenizer = load_tokenizer()
    encoder = load_image_encoder()
    try:
        features = encoder.predict(preprocess_image(image), verbose=0)
        if features.shape != (1, FEATURE_SIZE):
            raise ValueError("Unexpected image feature shape")
        caption = predict_caption(model, features, tokenizer)
        if not caption:
            raise CaptionGenerationError(
                "The model couldn't find a description for this image. Please try another photo."
            )
        return caption
    except CaptionGenerationError:
        raise
    except Exception as error:
        LOGGER.exception("Caption prediction failed")
        raise CaptionGenerationError(
            "We couldn't generate a caption for this image. Please try again or upload another photo."
        ) from error
