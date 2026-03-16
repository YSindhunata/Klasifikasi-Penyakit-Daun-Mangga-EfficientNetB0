import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from rembg import remove
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Deteksi Penyakit Daun Mangga", layout="centered")
st.title(" Sistem Klasifikasi Penyakit Daun Mangga")
st.write("Arsitektur: **EfficientNet B0**")

@st.cache_resource
def load_model():
    model_path = 'best_efficientnet_mango_CLEAN.keras'
    return tf.keras.models.load_model(model_path)
try:
    model = load_model()
    st.success("Model EfficientNet B0 berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .keras berada di folder yang sama. Error: {e}")

CLASS_NAMES = [
    'Anthracnose', 
    'Bacterial Canker', 
    'Cutting Weevil', 
    'Die Back', 
    'Gall Midge', 
    'Healthy', 
    'Powdery Mildew', 
    'Sooty Mould'
]

uploaded_file = st.file_uploader("Unggah citra daun mangga (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar asli
    original_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="1. Citra Asli", use_container_width=True)

    with st.spinner("Sedang menghapus latar belakang..."):
        output_rgba = remove(original_image)
        black_canvas = Image.new("RGB", output_rgba.size, (0, 0, 0))
        black_canvas.paste(output_rgba, mask=output_rgba.split()[3])
        with col2:
            st.image(black_canvas, caption="2. Hasil Segmentasi (Latar Hitam)", use_container_width=True)

    with st.spinner("Sedang memprediksi penyakit..."):
        img_resized = black_canvas.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array) 
        predictions = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100

        st.markdown("---")
        st.subheader("Hasil Analisis:")
        
        if predicted_class_name == 'Healthy':
            st.success(f"**Diagnosis:** {predicted_class_name} (Akurasi: {confidence:.2f}%)")
        else:
            st.error(f"**Diagnosis:** Terindikasi penyakit **{predicted_class_name}** (Akurasi: {confidence:.2f}%)")