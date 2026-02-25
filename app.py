import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model.keras")

st.title("🐶 Dogs vs Cats Classifier")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show image (convert BGR → RGB for display)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image")

    # Preprocess for model
    img = cv2.resize(img, (256, 256))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("🐶 Dog")
    else:
        st.success("🐱 Cat")