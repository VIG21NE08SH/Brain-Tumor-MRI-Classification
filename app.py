import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import gdown

model_path = Path(__file__).parent / "brain_tumor_model.keras"
file_id = "1vSae7ofweP0wylvaiuIZAXrnF3sVyAHX"

if not model_path.exists():
    with st.spinner("Downloading model..."):
        gdown.download(
            id=file_id,
            output=str(model_path),
            quiet=False
        )

model = tf.keras.models.load_model(model_path)

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("Brain Tumor MRI Classification")
st.write("Educational classification demonstration only.")

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction Result")
    st.write(predicted_class)

st.warning("Educational use only. This app cannot diagnose a medical condition.")
