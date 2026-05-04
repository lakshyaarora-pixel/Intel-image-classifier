# import streamlit as st
# import tensorflow
# import numpy as np
# from PIL import Image
#
# # ----------------------------
# # Page Config
# # ----------------------------
# st.set_page_config(page_title="Intel Image Classifier", layout="centered")
#
# st.title("🌍 Intel Image Classification App")
# st.write("Upload an image and the model will predict the scene type.")
#
# # ----------------------------
# # Load Model
# # ----------------------------
# @st.cache_resource
# def load_model():
#     model = tensorflow.keras.models.load_model("intel_model2.keras")
#     return model
#
# model = load_model()
#
# # ----------------------------
# # Class Names
# # ----------------------------
# class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
#
# # ----------------------------
# # Image Upload
# # ----------------------------
# uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
#
# if uploaded_file is not None:
#     # Show image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)
#
#     from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#
#     image = image.convert("RGB")
#
#     input_shape = model.input_shape[1:3]
#     img = image.resize(input_shape)
#
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#
#
#     # Prediction
#     prediction = model.predict(img_array)
#     score = prediction[0]
#
#     st.write("Prediction shape:", prediction.shape)
#     st.write("Score length:", len(prediction[0]))
#
#     predicted_class = class_names[np.argmax(score)]
#     confidence = np.max(score) * 100
#
#     # Output
#     st.success(f"Prediction: **{predicted_class}**")
#     st.info(f"Confidence: **{confidence:.2f}%**")
#     st.write(prediction.shape)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Intel Image Classifier", layout="wide")

# ----------------------------
# TITLE
# ----------------------------
st.title("🌍 Intel Scene Classification")
st.markdown("Upload an image and the model will predict the scene category with confidence.")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Settings")

show_top3 = st.sidebar.checkbox("Show Top 3 Predictions", True)
show_chart = st.sidebar.checkbox("Show Confidence Chart", True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("intel_model2.keras")

model = load_model()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # ----------------------------
    # LEFT: IMAGE
    # ----------------------------
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # ----------------------------
    # RIGHT: PREDICTION
    # ----------------------------
    with col2:

        with st.spinner("🔍 Analyzing image..."):

            # Preprocess
            input_shape = model.input_shape[1:3]
            img = image.resize(input_shape)

            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediction
            prediction = model.predict(img_array)
            score = prediction[0]

            idx = np.argmax(score)
            confidence = np.max(score) * 100

            predicted_class = class_names[idx]

        # ----------------------------
        # RESULT
        # ----------------------------
        st.success(f"🎯 Prediction: **{predicted_class.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # ----------------------------
        # TOP 3
        # ----------------------------
        if show_top3:
            st.subheader("🔝 Top 3 Predictions")

            top_3_idx = np.argsort(score)[-3:][::-1]

            for i in top_3_idx:
                st.write(f"{class_names[i]} → {score[i]*100:.2f}%")

        # ----------------------------
        # CHART
        # ----------------------------
        if show_chart:
            st.subheader("📊 Confidence Distribution")

            chart_data = {class_names[i]: score[i] for i in range(len(class_names))}
            st.bar_chart(chart_data)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using TensorFlow & Streamlit")