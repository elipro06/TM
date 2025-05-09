import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import platform

# Mostrar la versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título de la aplicación
st.title("Reconocimiento Inteligente de Imágenes")

# Imagen de ejemplo
image = Image.open('OIG5.jpg')
st.image(image, caption="Imagen de ejemplo", width=350)

# Instrucciones en la barra lateral
with st.sidebar:
    st.subheader("Este modelo fue entrenado usando Teachable Machine")
    st.write("Tómate una foto y el modelo predecirá una de las siguientes posiciones: izquierda o arriba.")

# Captura de imagen desde la cámara
img_file_buffer = st.camera_input("Haz clic para tomar una foto")

if img_file_buffer is not None:
    # Preparar la imagen para predicción
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))  # Redimensionar

    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model.predict(data)

    # Mostrar resultados
    st.subheader("Resultado de la predicción:")
    if prediction[0][0] > 0.5:
        st.markdown(f"**Izquierda** con una probabilidad de: {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.markdown(f"**Arriba** con una probabilidad de: {prediction[0][1]:.2f}")


