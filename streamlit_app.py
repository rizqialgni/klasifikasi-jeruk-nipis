import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Load model
model = tf.keras.models.load_model("model_jeruk_nipis.h5")
import json

# Kelas
class_names = ["Matang", "Mentah", "Setengah Matang"]

# Load label map
with open("label_map.json", "r") as f:
    class_indices = json.load(f)

# Balikkan dictionary-nya biar index jadi key
label_map = {v: k for k, v in class_indices.items()}

st.title("🍋 Klasifikasi Tingkat Kematangan Jeruk Nipis")
st.markdown("Upload gambar buah jeruk nipis, dan sistem akan memprediksi tingkat kematangannya.")

# Upload file dari komputer
uploaded_file = st.file_uploader("Pilih gambar jeruk...", type=["jpg", "jpeg", "png"])
# Upload dari kamera langsung
camera_image = st.camera_input("Atau ambil gambar langsung dari kamera:")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

if image is not None:
    st.image(image, caption='Gambar yang dipilih', use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index] * 100

    label = class_names[pred_index]
    st.success(f"**Prediksi: {label} ({confidence:.2f}%)**")


def set_bg_local(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local("bg.jpg")  # Gambar background di folder yang sama
