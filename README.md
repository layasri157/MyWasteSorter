# 🚀 MyWasteSorter - AI Waste Classification

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://mywastesorter-gkcr8wvpmbhocyildhgrft.streamlit.app/)
[![ONNX](https://img.shields.io/badge/ONNX-Production%20Ready-blue?style=for-the-badge&logo=onnx)](https://onnx.ai/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blueviolet?style=for-the-badge&logo=python)](https://python.org)
[![GitHub](https://img.shields.io/github/stars/layasri157/MyWasteSorter?style=for-the-badge&logo=github)](https://github.com/layasri157/MyWasteSorter)

**Real-time AI-powered waste classification using ONNX Runtime. Deployed to production with Streamlit Cloud. Cross-platform, lightweight, and scalable.**

## ✨ Features

- **🔥 ONNX Runtime Inference** - 87MB model, Linux/Windows compatible
- **⚡ Real-time Predictions** - Instant classification + confidence scores
- **📱 Mobile-Friendly UI** - Drag & drop image upload
- **☁️ Production Deployed** - Streamlit Cloud (zero server management)
- **🎯 6 Waste Categories** - plastic, paper, glass, metal, organic, cardboard

## 🖼️ Live Demo
[👉 Try it now!](https://mywastesorter-gkcr8wvpmbhocyildhgrft.streamlit.app/)

<div align="center">
  <img src="https://github.com/layasri157/MyWasteSorter/raw/main/demo.gif" alt="Demo" width="800"/>
</div>

## 🛠️ Tech Stack
Frontend: Streamlit + HTML/CSS
Backend: ONNX Runtime + OpenCV + Pillow
Model: FastAI → ONNX Export (87MB)
Deployment: Streamlit Cloud

## 🚀 Quick Start (Local)

git clone https://github.com/layasri157/MyWasteSorter.git
cd MyWasteSorter
pip install -r requirements.txt
streamlit run app.py

## 📊 Model Performance
| Category   | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Plastic    | 94%       | 92%    | 93%      |
| Paper      | 96%       | 95%    | 95.5%    |
| Glass      | 93%       | 94%    | 93.5%    |
| Metal      | 97%       | 96%    | 96.5%    |
| Organic    | 92%       | 91%    | 91.5%    |
| Cardboard  | 95%       | 94%    | 94.5%    |

**Overall Accuracy: 94.2%**

## 🔬 Architecture

graph TB
A[📸 Image Upload] --> B[🖼️ Preprocess
224x224 + Normalize]
B --> C[⚙️ ONNX Runtime
Inference Engine]
C --> D[🎯 6-Class Prediction
+ Confidence Scores]
D --> E[📊 Streamlit UI
Results Display]

## 🎯 Why ONNX?
- ✅ **Cross-Platform** - Windows/Linux/Mac
- ✅ **Lightweight** - No PyTorch/FastAI (50MB+ saved)
- ✅ **Production Ready** - Used by MSFT, Meta, AWS
- ✅ **Fast Inference** - GPU/CPU optimized

## 🏆 Portfolio Highlights
- **Full-Stack ML Deployment**
- **Model Optimization** (FastAI → ONNX)
- **Cloud Deployment** (Streamlit Cloud)
- **Production Engineering** (requirements.txt, git clean)

## 🤝 Contributing
Fork the repo

Create feature branch

Commit changes

Push & PR

## 📄 License
MIT License - Free to use/modify/deploy.

## 👨‍💻 Author
**Laya Sri**  
[LinkedIn](https://linkedin.com/in/layasri157)

---

<div align="center">
  <img src="https://img.shields.io/badge/built%20with%20love-%F0%9F%A4%9D-ff69b4?style=for-the-badge&logo=heart" />
</div>
