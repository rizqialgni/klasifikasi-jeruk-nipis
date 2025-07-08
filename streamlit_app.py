import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model_jeruk_nipis.h5")
class_names = ["Mentah", "Setengah Matang", "Matang"]

st.title("🍋 Klasifikasi Kematangan Jeruk Nipis")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index] * 100

    st.success(f"**Prediksi: {class_names[pred_index]} ({confidence:.2f}%)**")
