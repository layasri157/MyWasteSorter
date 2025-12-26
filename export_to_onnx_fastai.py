from fastai.vision.all import *
import torch
from pathlib import Path

PKL_PATH = Path("waste_sorter_model.pkl")   # your current fastai file
ONNX_PATH = Path("waste_sorter_model.onnx")

# 1) load fastai learner
learn = load_learner(PKL_PATH)

# 2) get underlying torch model
model = learn.model
model.eval()

# 3) dummy input (adjust 224 if you trained with different size)
dummy = torch.randn(1, 3, 224, 224)

# 4) export to ONNX
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH.as_posix(),
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)

print("Saved", ONNX_PATH)
