"""Regression checks for the original decoder contract and upload workflow."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from keras_preprocessing.text import Tokenizer
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

import caption_generator as generator
from utils import format_caption, open_uploaded_image


class SequenceModel:
    """Return known next-word indices and retain decoder inputs for assertions."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = iter(indices)
        self.sequences = []

    def predict(self, inputs, verbose=0):
        self.sequences.append(inputs[1].copy())
        result = np.zeros((1, 100), dtype=np.float32)
        result[0, next(self.indices)] = 1
        return result


class CaptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(["startseq little girl endseq"])
        self.features = np.zeros((1, generator.FEATURE_SIZE), dtype=np.float32)

    def test_decoder_stops_at_end_token_and_preserves_pre_padding(self) -> None:
        indices = [self.tokenizer.word_index[word] for word in
                   ("little", "girl", "endseq")]
        model = SequenceModel(indices)
        caption = generator.predict_caption(model, self.features, self.tokenizer)
        self.assertEqual(caption, "Little girl.")
        self.assertEqual(len(model.sequences), 3)
        self.assertEqual(model.sequences[0].shape, (1, 35))
        self.assertTrue(np.all(model.sequences[0][0, :-1] == 0))
        self.assertEqual(model.sequences[0][0, -1],
                         self.tokenizer.word_index["startseq"])

    def test_unknown_index_returns_partial_caption(self) -> None:
        model = SequenceModel([self.tokenizer.word_index["little"], 99])
        self.assertEqual(generator.predict_caption(model, self.features,
                                                   self.tokenizer), "Little.")

    def test_generation_limit_and_reverse_vocabulary_fallback(self) -> None:
        self.tokenizer.index_word = {}
        model = SequenceModel([self.tokenizer.word_index["little"]] * 35)
        caption = generator.predict_caption(model, self.features, self.tokenizer)
        self.assertEqual(len(model.sequences), 35)
        self.assertEqual(len(caption.split()), 35)
        self.assertNotIn("startseq", caption)

    def test_all_upload_formats_and_rgb_preprocessing(self) -> None:
        for file_format, mode in (("JPEG", "RGB"), ("PNG", "RGBA"),
                                  ("WEBP", "RGBA"), ("PNG", "L")):
            with self.subTest(file_format=file_format, mode=mode):
                upload = BytesIO()
                Image.new(mode, (17, 23)).save(upload, format=file_format)
                upload.seek(0)
                image = open_uploaded_image(upload)
                self.assertEqual(image.mode, "RGB")
                batch = generator.preprocess_image(image)
                self.assertEqual(batch.shape, (1, 224, 224, 3))
                self.assertEqual(batch.dtype, np.float32)
        red = generator.preprocess_image(Image.new("RGB", (1, 1), "red"))
        np.testing.assert_allclose(red[0, 0, 0], [-103.939, -116.779, 131.32],
                                   atol=0.001)

    def test_invalid_upload_has_friendly_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "couldn't open"):
            open_uploaded_image(BytesIO(b"not an image"))

    def test_caption_formatting(self) -> None:
        self.assertEqual(format_caption("startseq little girl endseq"), "Little girl.")
        self.assertEqual(format_caption("STARTSEQ hello! ENDSEQ"), "Hello!")
        self.assertEqual(format_caption("startseq endseq"), "")

    def test_missing_artifacts_have_friendly_errors(self) -> None:
        for path_name, loader, filename in (
            ("MODEL_PATH", generator.load_caption_model, "best_model.h5"),
            ("TOKENIZER_PATH", generator.load_tokenizer, "tokenizer.pkl"),
        ):
            loader.clear()
            with patch.object(generator, path_name, Path("missing-artifact")):
                with self.assertRaisesRegex(generator.CaptionGenerationError, filename):
                    loader()
            loader.clear()

    def test_prediction_failure_has_friendly_error(self) -> None:
        with patch.object(generator, "load_caption_model"), \
             patch.object(generator, "load_tokenizer", return_value=self.tokenizer), \
             patch.object(generator, "load_image_encoder") as encoder:
            encoder.return_value.predict.side_effect = RuntimeError("prediction failed")
            with self.assertRaisesRegex(generator.CaptionGenerationError,
                                        "couldn't generate"):
                generator.generate_caption(Image.new("RGB", (10, 10)))

    def test_models_and_tokenizer_are_cached(self) -> None:
        generator.load_caption_model.clear()
        generator.load_image_encoder.clear()
        generator.load_tokenizer.clear()

        decoder = Mock(input_shape=[(None, 4096), (None, 35)])
        with patch.object(generator, "load_model", return_value=decoder) as load, \
             patch.object(generator, "VGG16") as vgg, \
             patch.object(generator, "Model") as feature_model:
            self.assertIs(generator.load_caption_model(), generator.load_caption_model())
            load.assert_called_once_with(str(generator.MODEL_PATH), compile=False)
            self.assertIs(generator.load_image_encoder(), generator.load_image_encoder())
            vgg.assert_called_once_with(weights="imagenet", include_top=True)
            feature_model.assert_called_once()
            self.assertIs(generator.load_tokenizer(), generator.load_tokenizer())
        generator.load_caption_model.clear()
        generator.load_image_encoder.clear()
        generator.load_tokenizer.clear()

    def test_original_decoder_matches_tokenizer(self) -> None:
        model = generator.load_caption_model()
        tokenizer = generator.load_tokenizer()
        self.assertEqual(model.input_shape, [(None, 4096), (None, 35)])
        self.assertEqual(model.output_shape[-1], len(tokenizer.word_index) + 1)
        sequence = np.zeros((1, 35), dtype=np.int32)
        sequence[0, -1] = tokenizer.word_index["startseq"]
        probabilities = model.predict([self.features, sequence], verbose=0)
        self.assertEqual(probabilities.shape, (1, model.output_shape[-1]))
        self.assertTrue(np.all(np.isfinite(probabilities)))

    def test_streamlit_initial_screen_does_not_load_models(self) -> None:
        with patch.object(generator, "load_caption_model") as loader:
            app = AppTest.from_file("main.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.title[0].value, "VisionScribe")
            self.assertTrue(app.button[0].disabled)
            loader.assert_not_called()

    def test_streamlit_generate_preserve_and_clear(self) -> None:
        upload = BytesIO()
        Image.new("RGB", (10, 10), "red").save(upload, format="PNG")
        with patch("streamlit.file_uploader", return_value=upload), \
             patch.object(generator, "generate_caption", return_value="A red image.") as predict:
            app = AppTest.from_file("main.py").run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            predict.assert_not_called()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["caption"], "A red image.")
            app.run()
            predict.assert_called_once()
            app.button[1].click().run()
            self.assertEqual(app.session_state["upload_number"], 1)
            self.assertNotIn("caption", app.session_state)


if __name__ == "__main__":
    unittest.main()
