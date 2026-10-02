import streamlit as st
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
# Load the pre-trained model
model_path = "best_model.h5"  # Replace with the actual path
model = load_model(model_path)

# Load the tokenizer
tokenizer_path = "tokenizer.pkl"  # Replace with the actual path
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Set the maximum length for captions
max_length = 35  # Replace with the actual max length

# Function to generate captions
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'

    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)

        # Predict next word
        yhat = model.predict([image, sequence], verbose=0)

        # Get index with high probability
        yhat = np.argmax(yhat)

        # Convert index to word
        word = idx_to_word(yhat, tokenizer)

        # Stop if word not found
        if word is None:
            break

        # Append word as input for generating the next word
        in_text += " " + word

        # Stop if we reach end tag
        if word == 'endseq':
            break

    return in_text

# Streamlit app
vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs,
                  outputs=vgg_model.layers[-2].output)


# ...

def main():
    st.title("Image Caption Generator")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Display the uploaded image with reduced width
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.markdown(
            f'<style>img{{max-width: 300px; max-height: 300px;margin: auto;}}</style>',
            unsafe_allow_html=True
        )

        # Preprocess the image for model prediction
        image = Image.open(uploaded_file)
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        image_array = preprocess_input(image_array)

        # Generate feature vector using the VGG model
        feature = vgg_model.predict(image_array, verbose=0)

        # Generate caption
        caption = predict_caption(model, feature, tokenizer, max_length)

        # Display the generated caption
        st.subheader("Generated Caption:")
        st.write(caption)


# ...


if __name__ == "__main__":
    main()