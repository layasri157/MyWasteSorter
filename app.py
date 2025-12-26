import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time
from PIL import Image
import numpy as np

import onnx_infer  # Correct ONNX import

# Animated and styled title
st.set_page_config(page_title="MyWasteSorter", layout="wide")
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50; font-family: 'Courier New', Courier, monospace;">
    🍀 <span style='animation: rainbow 2s infinite;'>Waste Sorter</span> 🍀
    </h1>
    <p style="text-align: center; color: #666; font-size: 1.2em;">
    AI-Powered Waste Classification • ONNX Runtime • Production Ready
    </p>
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

st.markdown("---")
st.markdown("**Upload a waste image and watch the magic happen ✨**")

# Load or create history with caching
@st.cache_data(ttl=300)
def load_history():
    if os.path.exists('prediction_history.csv'):
        try:
            df = pd.read_csv('prediction_history.csv')
            if not all(col in df.columns for col in ["Timestamp", "Filename", "Prediction", "Confidence"]):
                return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])
            return df
        except:
            return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])
    return pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Confidence"])

# Session state for uploader
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def clear_file():
    st.session_state.upload_key += 1

# Two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"], 
        key=st.session_state.upload_key,
        help="Upload plastic, paper, glass, metal, organic, or cardboard"
    )
    
    if st.button("🗑️ Clear Image", use_container_width=True):
        clear_file()
        st.rerun()

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        # Prediction with spinners - CORRECT ONNX CALLS
        with st.spinner('🔍 Preprocessing image...'):
            time.sleep(0.5)
        
        with st.spinner('🎯 Predicting waste category...'):
            # CORRECT ONNX WORKFLOW
            img_array = onnx_infer.preprocess_image(image)
            prediction, confidence = onnx_infer.predict_onnx(img_array)
            pred = prediction
            conf = confidence
            time.sleep(0.8)

        # Results
        st.markdown("---")
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #4CAF50, #81C784); 
                    padding: 1rem; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">🎯 {pred}</h2>
            <h3 style="color: white; margin: 0; font-size: 1.2em;">
                Confidence: <span style="font-size: 1.5em;">{conf:.1f}%</span>
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Category info
        infos = {
            "Plastic": "♻️ Bottles, bags, containers, packaging materials",
            "Glass": "🔮 Bottles, jars, broken glass",
            "Metal": "🛠️ Cans, tins, aluminum foil, steel",
            "Paper": "📄 Newspapers, magazines, office paper",
            "Organic": "🍂 Food scraps, vegetable peels, leaves",
            "Cardboard": "📦 Boxes, packaging cardboard"
        }
        
        if str(pred) in infos:
            st.markdown(f"**ℹ️ {infos[str(pred)]}**")

        # Save to history
        history_df = load_history()
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": uploaded_file.name,
            "Prediction": str(pred),
            "Confidence": round(float(conf), 4)
        }
        new_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        new_df.tail(100).to_csv('prediction_history.csv', index=False)

with col2:
    st.markdown("---")
    st.markdown("""
    <h3 style="text-align:center; color:#673AB7; text-shadow: 0 0 8px #673AB7;">
    📜 Recent History
    </h3>
    """, unsafe_allow_html=True)
    
    history_df = load_history()
    if not history_df.empty:
        st.markdown("""
        <style>
        .dataframe th { background-color: #4CAF50; color: white; }
        .dataframe td { padding: 8px; }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            history_df.tail(10), 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.1f%%"
                )
            }
        )
        
        # Stats
        st.markdown("---")
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("Total Predictions", len(history_df))
        with col_stats2:
            avg_conf = history_df["Confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    else:
        st.info("👆 Upload an image to see history populate!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>Built with ❤️ using <strong>ONNX Runtime</strong> | 
    Deployed on <strong>Streamlit Cloud</strong></p>
</div>
""", unsafe_allow_html=True)
