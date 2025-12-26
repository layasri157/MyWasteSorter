import streamlit as st
from fastai.vision.all import *
import pandas as pd
from datetime import datetime
import os
import time

# Load model once
@st.cache_resource
def load_model():
    # use the Linux‑friendly exported learner
    return load_learner('waste_sorter_model_export.pkl')

learn = load_model()

# Animated and styled title
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50; font-family: 'Courier New', Courier, monospace;">
    🍀 <span style='animation: rainbow 2s infinite;'>Waste Sorter</span> 🍀
    </h1>

    <style>
    @keyframes rainbow {
        0%{color: #ff1744}
        20%{color: #f50057}
        40%{color: #d500f9}
        60%{color: #651fff}
        80%{color: #2979ff}
        100%{color: #00b0ff}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("Upload a waste image and watch the magic happen ✨")

# Load or create history
@st.cache_data
def load_history():
    if os.path.exists('prediction_history.csv'):
        return pd.read_csv('prediction_history.csv')
    return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])

history_df = load_history()

# Control uploader with session state for clearing
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def clear_file():
    st.session_state.upload_key += 1

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"], key=st.session_state.upload_key
)

if st.button("Clear Image"):
    clear_file()

if uploaded_file is not None:
    with st.spinner('Analyzing image... 🔍'):
        image = PILImage.create(uploaded_file)
        time.sleep(1)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Predicting waste category... 🎯'):
        pred, pred_idx, probs = learn.predict(image)
        time.sleep(1)

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {probs[pred_idx]:.4f}")

    infos = {
        "Plastic": "♻️ Includes plastic bottles, bags, packaging.",
        "Glass": "🔮 Includes bottles and jars.",
        "Metal": "🛠️ Includes cans, tins, foils.",
        "Paper": "📄 Includes newspapers, cardboard, journals.",
        "Organic": "🍂 Includes food scraps and leaves."
    }

    if str(pred) in infos:
        st.info(infos[str(pred)])

    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": uploaded_file.name,
        "Prediction": str(pred),
        "Confidence": round(float(probs[pred_idx]), 4)
    }
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv('prediction_history.csv', index=False)

# Separate section with glowing header
st.markdown("---")
st.markdown("""
<h2 style="text-align:center; color:#673AB7; text-shadow: 0 0 8px #673AB7;">
📜 Prediction History
</h2>
""", unsafe_allow_html=True)

st.dataframe(history_df)
