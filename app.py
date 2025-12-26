import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time
from PIL import Image

import onnx_infer  # ONNX inference module

# ------------ Page setup & hero header ------------

st.set_page_config(page_title="MyWasteSorter", layout="wide")

st.markdown("""
    <h1 style="text-align: center; color: #22c55e; font-family: 'SF Mono', 'Courier New', monospace;">
      🍀 <span style='animation: rainbow 2s linear infinite;'>MyWasteSorter</span>
    </h1>
    <p style="text-align: center; color: #9ca3af; font-size: 1.05rem;">
      AI‑powered waste classification • ONNX Runtime • Streamlit Cloud
    </p>
    <style>
      @keyframes rainbow {
        0%{color:#f97316}
        25%{color:#ec4899}
        50%{color:#6366f1}
        75%{color:#22c55e}
        100%{color:#eab308}
      }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("**Upload a waste image and get an instant, clean prediction. ✨**")

# ------------ History helpers ------------

@st.cache_data(ttl=300)
def load_history():
    """Load last predictions from CSV or create empty."""
    if os.path.exists("prediction_history.csv"):
        try:
            df = pd.read_csv("prediction_history.csv")
            needed = ["Timestamp", "Filename", "Prediction", "Confidence"]
            if all(c in df.columns for c in needed):
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])

def append_history(filename: str, pred: str, conf: float):
    """Append one row and keep last 100 entries."""
    df = load_history()
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Prediction": pred,
        "Confidence": round(float(conf), 2),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.tail(100).to_csv("prediction_history.csv", index=False)
    return df

# ------------ Session state for clear button ------------

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

def clear_file():
    st.session_state.upload_key += 1

# ------------ Layout ------------

left, right = st.columns([2, 1], vertical_alignment="top")

# ------------ Left: upload & prediction ------------

with left:
    st.subheader("📸 Upload image")

    uploaded_file = st.file_uploader(
        "Drop a JPG/PNG here",
        type=["jpg", "jpeg", "png"],
        key=st.session_state.upload_key,
        help="Try plastic, paper, glass, metal, organic or cardboard items.",
    )

    if st.button("Clear image", type="secondary"):
        clear_file()
        st.rerun()

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        with st.spinner("Analyzing image..."):
            time.sleep(0.4)
            img_array = onnx_infer.preprocess_image(image)
            label, confidence = onnx_infer.predict_onnx(img_array)

        # Pretty result card
        st.markdown("")
        st.markdown(
            f"""
            <div style="
                background: radial-gradient(circle at top left, #22c55e, #16a34a 40%, #111827);
                padding: 1.2rem 1.4rem;
                border-radius: 1rem;
                color: white;
                box-shadow: 0 18px 45px rgba(0,0,0,0.45);
                text-align: left;
            ">
              <div style="font-size: 0.9rem; opacity: 0.85;">Prediction</div>
              <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.2rem;">
                {label}
              </div>
              <div style="font-size: 0.95rem; margin-top: 0.3rem; opacity: 0.9;">
                Confidence: {confidence:.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Small helper text per class
        infos = {
            "Plastic": "Bottles, bags, containers, wrappers and packaging films.",
            "Glass": "Bottles, jars and clean glass containers.",
            "Metal": "Cans, tins, foil and other metal packaging.",
            "Paper": "Newspapers, notebooks, printing paper and magazines.",
            "Organic": "Food scraps, vegetable peels, coffee grounds and leaves.",
            "Cardboard": "Shipping boxes, cereal boxes and thick paperboard.",
        }
        if str(label) in infos:
            st.markdown(f"**Tip:** {infos[str(label)]}")

        # Persist to history
        updated = append_history(uploaded_file.name, str(label), confidence)

# ------------ Right: history panel ------------

with right:
    st.subheader("📜 Recent predictions")

    hist = load_history()
    if not hist.empty:
        # Only last 10 rows, newest at top
        hist_view = hist.tail(10).iloc[::-1].reset_index(drop=True)

        st.dataframe(
            hist_view,
            use_container_width=True,
            hide_index=True,
        )

        # Compact stats
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total predictions", len(hist))
        with c2:
            st.metric("Avg confidence", f"{hist['Confidence'].mean():.1f}%")
    else:
        st.info("No history yet. Upload an image to see your predictions here.")

# ------------ Footer ------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#6b7280; font-size:0.85rem; padding:0.5rem 0 0.2rem;">
      Built with <span style="color:#f97316;">Streamlit</span> · Optimized with <span style="color:#22c55e;">ONNX Runtime</span>
    </div>
    """,
    unsafe_allow_html=True,
)
