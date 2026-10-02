# VisionScribe – AI Image Caption Generator

## Overview

VisionScribe turns an uploaded photo into a natural-language description using a
VGG16 image encoder and a trained LSTM caption decoder. The Streamlit interface
provides an image preview, explicit caption generation, and a clear result card.
The application reuses the supplied trained model and tokenizer; no retraining is
required.

## Live Demo

[Open VisionScribe][(https://YOUR-APP-NAME.streamlit.app](https://image-caption-generator-mvzdrcuzwoepbfhnzmsa4b.streamlit.app/))

Replace this placeholder with your Streamlit Community Cloud URL.

## Features

- Upload JPG, JPEG, PNG, or WEBP images with an immediate preview.
- Convert images to RGB and correct EXIF camera orientation.
- Generate captions on demand with a progress spinner.
- Display sentences without internal `startseq` and `endseq` markers.
- Clear the current image and caption to start again.
- Cache VGG16, the caption model, and tokenizer across Streamlit reruns.
- Show friendly messages for invalid images, missing artifacts, and inference failures.
- Learn about the model from the sidebar.

## How It Works

1. Upload an image and select **Generate Caption**.
2. The application converts it to RGB, resizes it to 224 × 224, and applies
   `tensorflow.keras.applications.vgg16.preprocess_input`.
3. VGG16's `fc2` layer produces a 4096-dimensional feature vector.
4. Starting with `startseq`, the decoder combines image features with the current
   word sequence and predicts the most likely next word.
5. Generation stops at `endseq`, an unmapped vocabulary index, or the 35-step limit.
6. Internal markers are removed, the first letter is capitalized, and sentence
   punctuation is added when needed.

## Architecture

```text
Uploaded Image
      ↓
VGG16 CNN
      ↓
4096-dimensional image features
      ↓
Embedding + LSTM
      ↓
Word-by-word prediction
      ↓
Generated Caption
```

VGG16 retains its fully connected layers through `fc2` and excludes only the final
ImageNet prediction layer. Using `include_top=False` would produce incompatible
features. In the saved decoder, the image branch uses Dropout and a Dense layer;
the text branch uses Embedding, Dropout, and LSTM. Their outputs are added before
the final Dense vocabulary prediction layers.

## Technologies Used

- Python 3.11
- Streamlit for the interface, session state, and resource caching
- TensorFlow 2.15.1 / bundled Keras 2 for legacy H5 inference
- NumPy for image batches and prediction selection
- Pillow for image decoding and conversion
- Keras-Preprocessing 1.1.2 to deserialize the original tokenizer

## Project Structure

```text
VisionScribe/
├── main.py                       # Streamlit interface and session workflow
├── caption_generator.py          # Cached loaders, preprocessing, decoding
├── utils.py                      # Image loading and caption formatting
├── tests/test_caption_generator.py # Inference and UI regression checks
├── best_model.h5                 # Original trained caption model (unchanged)
├── tokenizer.pkl                 # Original vocabulary (unchanged)
├── image-caption-generator.ipynb  # Original training/reference notebook
├── requirements.txt              # Runtime dependencies
├── .streamlit/config.toml        # Interface theme
├── .gitignore                    # Local environment and IDE exclusions
└── README.md
```

## Installation

Install Python **3.11**, download or clone this repository, and open a terminal in
the project directory. Keep `best_model.h5` and `tokenizer.pkl` beside the Python
files.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Locally

```bash
python -m streamlit run main.py
```

Open the URL shown in your terminal. Model resources load on the first generation
request and remain cached for subsequent interactions. The first request downloads
the ImageNet VGG16 weights if they are not already in the Keras cache, so internet
access is needed initially. A server restart clears the in-memory cache.

Run the regression checks without downloading VGG16 weights:

```bash
python -m unittest discover -s tests -v
```

For **Streamlit Community Cloud**, push the project to GitHub, select `main.py` as
the entry point, and choose **Python 3.11** in the deployment's advanced settings.
Ensure both trained artifacts are included in the repository. Community Cloud
installs `requirements.txt` and reads `.streamlit/config.toml` automatically.

## Model Information

- **Dataset:** Flickr8k, a collection of approximately 8,000 photographs with
  multiple human-written captions per image.
- **Encoder:** VGG16 with ImageNet weights, using `(1, 4096)` `fc2` features.
- **Decoder:** Embedding and LSTM with a merged image feature branch.
- **Sequence length:** 35, confirmed by the original notebook's saved output.
- **Decoding:** Greedy next-word selection with pre-padding, matching the original
  implementation.
- **Artifacts:** The supplied `best_model.h5` and `tokenizer.pkl` are reused
  unchanged. The model loads with `compile=False` because training state is not
  needed for inference.

The notebook is retained as historical training documentation. Running the app
does not execute it or train the model. The app dependencies cover inference;
running every notebook cell may require additional research packages and the dataset.

## Example

For a photo of a girl in a pink dress, an illustrative result might be:

> Little girl in pink dress.

A raw sequence such as `startseq little girl in pink dress endseq` becomes the
sentence above. Formatting preserves predicted words; it does not invent missing
articles or rewrite grammar. Actual results depend on the photo and trained model.

## Limitations

- The Flickr8k vocabulary and training scenes limit generalization to unfamiliar
  objects, abstract art, specialized images, and complex situations.
- Captions can omit details, repeat words, or describe objects incorrectly.
- Greedy decoding is fast but may produce less fluent sentences than other methods.
- Large models require memory and make the first generation request slower,
  especially on shared hosting.
- The app describes images; it does not reliably read text or provide a complete
  accessibility description.

## Future Improvements

- Compare beam search with greedy decoding using the existing model.
- Add downloadable captions and batch image uploads.
- Evaluate caption quality on a held-out Flickr8k test set.
- Explore richer captioning models as a separate, optional extension.

## Credits

VisionScribe was adapted and extended from the open-source image-captioning
implementation supplied with this repository. The original training notebook,
VGG16 + LSTM architecture, and trained artifacts provide its foundation; this
version adds a modular inference layer, cached resources, a redesigned interface,
and clearer deployment documentation.

Acknowledgements to the Flickr8k dataset creators, the VGG researchers, and the
TensorFlow/Keras and Streamlit communities.
