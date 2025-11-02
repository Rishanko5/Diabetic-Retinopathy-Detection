import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import time

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #00C4CC;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #BBBBBB;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    .stImage {
        display: flex;
        justify-content: center;
    }
    .result-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 0px 20px rgba(0, 196, 204, 0.2);
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 60px;
        font-size: 14px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🩺 Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload a retinal image to predict the stage of Diabetic Retinopathy</p>", unsafe_allow_html=True)

@st.cache_resource
def load_dr_model():
    return load_model('DR_Detection_Model.keras')

model = load_dr_model()
class_names = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferative DR', 'Severe DR']
img_height, img_width = 180, 180

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    st.image(img, caption="🩻 Uploaded Retinal Image", width=350)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    with st.spinner("Analyzing image... Please wait ⏳"):
        time.sleep(1)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.progress(float(confidence) / 100)
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("👆 Upload a retinal image to get started.")

