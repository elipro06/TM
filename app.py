import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Carga el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título de la aplicación
st.title("Reconocimiento Inteligente de Imágenes")

# Imagen inicial para mostrar
image = Image.open('OIG5.jpg')
st.image(image, caption="Imagen de ejemplo", width=350)

# Sidebar con instrucciones
with st.sidebar:
    st.subheader("Este modelo fue entrenado usando Teachable Machine")
    st.write("Tómate una foto y el modelo predirá una de las siguientes posiciones: izquierda o arriba.")

# Entrada de cámara para capturar imagen
img_file_buffer = st.camera_input("Haz clic para tomar una foto")

if img_file_buffer is not None:
    # Se prepara la imagen para la predicción
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Abrir la imagen desde el buffer de la cámara
    img = Image.open(img_file_buffer)

    # Cambiar el tamaño de la imagen
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir la imagen PIL a numpy array
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la predicción con el modelo
    prediction = model.predict(data)
    
    # Mostrar los resultados según las probabilidades de las clases
    st.subheader("Resultado de la predicción:")
    if prediction[0][0] > 0.5:
        st.markdown(f"**Izquierda** con una probabilidad de: {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.markdown(f"**Arriba** con una probabilidad de: {prediction[0][1]:.2f}")
    # Si hay más clases, se pueden añadir aquí con una estructura similar
    # if prediction[0][2] > 0.5:
    #    st.markdown(f"**Derecha** con una probabilidad de: {prediction[0][2]:.2f}")


