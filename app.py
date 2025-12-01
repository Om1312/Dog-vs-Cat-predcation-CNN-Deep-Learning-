import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --------------------
# Load your trained CNN model
# --------------------
model = load_model("cat_dog_cnn_model.keras")    
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image and the model will classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.write("Classifying...")

    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # Normalization

    # Prediction
    result = model.predict(img_array)[0][0]

    # Show prediction
    if result > 0.5:
        st.success(f"Prediction: 🐶 **Dog** (Probability: {result:.5f})")
    else:
        st.success(f"Prediction: 🐱 **Cat** (Probability: {1 - result:.5f})")
