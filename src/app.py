import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from predict_cnn import predict_cnn  # 从 src 导入预测函数


labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

st.title("Fashion-MNIST Classifier (CNN)")
st.write("Upload an image of a fashion item, and the model will predict its category.")

file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file).convert("L").resize((28,28))
    st.image(img, caption="Uploaded Image")

    arr = np.array(img)
    pred = predict_cnn(arr)
    st.write("### Prediction:", labels[pred])