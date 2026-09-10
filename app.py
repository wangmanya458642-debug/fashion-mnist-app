import numpy as np
import streamlit as st
from PIL import Image

from src.predict_cnn import predict_cnn


LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

st.title("Fashion-MNIST Classifier (CNN)")
st.write("Upload a fashion-item image to classify it into one of 10 Fashion-MNIST categories.")
st.caption(
    "For best results, use a centered, Fashion-MNIST-style image with a light item "
    "on a dark background. The image is converted to grayscale and resized to 28×28 pixels."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Preprocessed 28×28 grayscale image", width=280)

    prediction = predict_cnn(np.asarray(image))
    st.success(f"Prediction: {LABELS[prediction]}")
