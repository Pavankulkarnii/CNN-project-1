import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model.h5')

st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

class_names = ['class_0','class_1']   # CHANGE THIS

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    if st.button("Predict"):
        pred = model.predict(img)
        result = class_names[np.argmax(pred)]
        st.success(result)