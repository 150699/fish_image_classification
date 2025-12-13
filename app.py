import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "final_fish_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
st.title(" Fish Image Classification App")

DATASET_TRAIN_DIR = "Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data/train"

if os.path.exists(DATASET_TRAIN_DIR):
    class_names = sorted(os.listdir(DATASET_TRAIN_DIR))
else:
    st.warning(" Could not find the dataset folder. Using placeholder names.")
    class_names = ["class_1", "class_2", "class_3"]

st.write("Detected Classes:", class_names)

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f" Prediction: **{class_names[index]}**")
    st.info(f" Confidence: `{confidence:.2f}`")