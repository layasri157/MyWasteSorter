from fastai.vision.all import *

# 1) Load your existing model trained on Windows
learn = load_learner("waste_sorter_model.pkl")

# 2) Export in fastai's standard format (Linux-friendly)
learn.export("waste_sorter_model_export.pkl")

print("Saved portable learner to waste_sorter_model_export.pkl")
