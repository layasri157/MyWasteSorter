import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time
from PIL import Image

from onnx_infer import predict_image  # ONNX backend

HISTORY_FILE = "prediction_history.csv"

# ---------- History helpers ----------

@st.cache_data
def load_history(dummy: int = 0):
    """Load prediction history from CSV."""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])

def append_history(filename: str, pred: str, conf: float):
    """Append one prediction row and save to CSV."""
    df = load_history()  # current history
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Prediction": str(pred),
        "Confidence": round(float(conf), 4),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.tail(100).to_csv(HISTORY_FILE, index=False)

# ---------- UI ----------

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

# ---------- Prediction ----------

if uploaded_file is not None:
    # Analyze image
    with st.spinner("Analyzing image... 🔍"):
        image = Image.open(uploaded_file).convert("RGB")
        time.sleep(1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict via ONNX (predict_image from onnx_infer.py)
    with st.spinner("Predicting waste category... 🎯"):
        pred, pred_idx, probs, conf = predict_image(image)
        time.sleep(1)

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf:.4f}")

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
    append_history(uploaded_file.name, pred, conf)

# ---------- History section ----------

st.markdown("---")
st.markdown(
    """
<h2 style="text-align:center; color:#673AB7; text-shadow: 0 0 8px #673AB7;">
📜 Prediction History
</h2>
""",
    unsafe_allow_html=True,
)

# Force cache refresh every 5 seconds so new rows appear
history_df = load_history(int(time.time() // 5))
st.dataframe(history_df, use_container_width=True)
