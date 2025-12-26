import pandas as pd
import streamlit as st
from datetime import datetime
import os

@st.cache_data(ttl=300)
def load_history():
    if os.path.exists('prediction_history.csv'):
        return pd.read_csv('prediction_history.csv')
    return pd.DataFrame(columns=['timestamp', 'filename', 'prediction', 'confidence'])

def add_prediction(filename, prediction, confidence):
    df = load_history()
    new_row = pd.DataFrame({
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'filename': [filename],
        'prediction': [prediction],
        'confidence': [f"{confidence:.1f}%"]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.tail(100).to_csv('prediction_history.csv', index=False)
    return df.tail(10)

def get_history():
    return load_history().tail(10)
