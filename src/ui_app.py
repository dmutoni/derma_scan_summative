# src/ui_app.py

import streamlit as st
import requests
from PIL import Image
import io
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DermaScan – AI Skin Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

# ------------------------------
# Custom Styling
# ------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #2E86C1;
        }
        .sub-title {
            font-size: 20px;
            font-weight: 500;
            color: #5D6D7E;
        }
        .section-divider {
            margin-top: 30px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# HEADER
# ------------------------------
st.markdown('<p class="main-title">🩺 DermaScan – AI Skin Disease Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Deep Learning • Dermatology • Model Retraining • Visual Insights</p>',
            unsafe_allow_html=True)

st.markdown("---")

# ------------------------------
# UPTIME CHECK
# ------------------------------
st.markdown("### 🔌 API & Model Status")

try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        st.success(f"API Running ✔  — Uptime: {data['uptime_seconds']} sec")
        st.info(f"Model supports **{data['num_classes']} classes**:\n{data['classes']}")
    else:
        st.error("API reachable but health check failed ❌")
except:
    st.error("API not reachable. Start FastAPI on port 8000.")

st.markdown("---")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Select Page:",
    ["🔍 Predict", "📦 Bulk Upload", "♻️ Retrain Model", "📊 Visualizations"]
)

# ------------------------------
# PAGE 1: PREDICT
# ------------------------------
if page == "🔍 Predict":

    st.markdown("## 🔍 Single Image Prediction")

    uploaded_img = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", width=350)

        if st.button("Predict Diagnosis"):
            with st.spinner("Running prediction..."):
                files = {"file": uploaded_img.getvalue()}
                try:
                    r = requests.post(f"{API_URL}/predict", files=files)
                    result = r.json()

                    if "class_name" in result:
                        st.success(f"Prediction: **{result['class_name']}**")
                        st.info(f"Confidence: **{result['confidence']:.2f}**")
                    else:
                        st.error(f"Error: {result}")
                except:
                    st.error("Failed to connect to API.")


# ------------------------------
# PAGE 2: BULK UPLOAD
# ------------------------------
elif page == "📦 Bulk Upload":

    st.markdown("## 📦 Upload New Training Images (Bulk)")

    files = st.file_uploader("Upload multiple images for retraining",
                             type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)

    if files:
        st.info(f"{len(files)} images selected.")

        if st.button("Upload to Server"):
            with st.spinner("Uploading..."):
                try:
                    upload_files = [("files", (f.name, f.getvalue(), "image/jpeg")) for f in files]
                    r = requests.post(f"{API_URL}/upload-bulk", files=upload_files)
                    st.success(f"Uploaded {r.json()['files_saved']} files successfully!")
                except:
                    st.error("Upload failed. API unreachable.")


# ------------------------------
# PAGE 3: RETRAIN MODEL
# ------------------------------
elif page == "♻️ Retrain Model":

    st.markdown("## ♻️ Retrain the Model")

    st.warning("⚠️ This may take several minutes.")

    if st.button("Start Retraining"):
        with st.spinner("Retraining model..."):
            try:
                r = requests.post(f"{API_URL}/retrain")
                result = r.json()

                if result["status"] == "no_new_data":
                    st.warning("No new data found for retraining.")
                else:
                    st.success("Retraining Complete 🎉")
                    st.write(f"Epochs: {result['epochs']}")
                    st.write(f"Final Train Accuracy: {result['final_train_acc']:.3f}")
                    st.write(f"Final Val Accuracy: {result['final_val_acc']:.3f}")
            except:
                print("Retraining failed (API unreachable).")


# ------------------------------
# PAGE 4: VISUALIZATIONS
# ------------------------------
elif page == "📊 Visualizations":

    st.markdown("## 📊 Dataset Insights")

    col1, col2 = st.columns(2)

    # Example Visualization 1 – Class Distribution
    with col1:
        st.markdown("### #️⃣ Class Distribution")
        fig, ax = plt.subplots()
        class_counts = {
            "eczema": 1677,
            "melanoma": 3140,
            "dermatitis": 1257,
            "bcc": 3323,
            "nevus": 7970,
            "keratosis": 2079
        }
        ax.bar(class_counts.keys(), class_counts.values(), color="#5DADE2")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Visualization 2 – Sample Image
    with col2:
        st.markdown("### 🖼 Example Image")
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Eczema.jpg",
                 caption="Sample Eczema Image",
                 use_column_width=True)

    # Visualization 3 – Image Shape
    st.markdown("### 📐 Typical Image Dimensions")
    fig2, ax2 = plt.subplots()
    ax2.bar(["Height", "Width", "Channels"], [256, 256, 3], color="#48C9B0")
    st.pyplot(fig2)
