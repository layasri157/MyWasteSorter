import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time
from PIL import Image

import onnx_infer  # uses preprocess_image + predict_onnx from your ONNX file

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
    if os.path.exists("prediction_history.csv"):
        try:
            return pd.read_csv("prediction_history.csv")
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])

history_df = load_history()

# Control uploader with session state for clearing
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

def clear_file():
    st.session_state.upload_key += 1

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"], key=st.session_state.upload_key
)

if st.button("Clear Image"):
    clear_file()
    st.rerun()

if uploaded_file is not None:
    # Analyze image
    with st.spinner("Analyzing image... 🔍"):
        image = Image.open(uploaded_file).convert("RGB")
        time.sleep(1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict via ONNX
    with st.spinner("Predicting waste category... 🎯"):
        img_array = onnx_infer.preprocess_image(image)
        pred, conf = onnx_infer.predict_onnx(img_array)  # conf is in percent (0–100)
        time.sleep(1)

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf/100:.4f}")

    # Category info with nice emoji
    infos = {
        "Plastic": "♻️ Includes plastic bottles, bags, packaging.",
        "Glass": "🔮 Includes bottles and jars.",
        "Metal": "🛠️ Includes cans, tins, foils.",
        "Paper": "📄 Includes newspapers, cardboard, journals.",
        "Organic": "🍂 Includes food scraps and leaves.",
        "Cardboard": "📦 Includes boxes and packaging cardboard.",
    }

    if str(pred) in infos:
        st.info(infos[str(pred)])

    # Save to history
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": uploaded_file.name,
        "Prediction": str(pred),
        "Confidence": round(float(conf / 100), 4),  # store like old fastai (0–1)
    }
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    history_df.to_csv("prediction_history.csv", index=False)

# Separate section with glowing header
st.markdown("---")
st.markdown(
    """
<h2 style="text-align:center; color:#673AB7; text-shadow: 0 0 8px #673AB7;">
📜 Prediction History
</h2>
""",
    unsafe_allow_html=True,
)

st.dataframe(history_df)
