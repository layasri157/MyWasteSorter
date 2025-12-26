import onnxruntime as ort
import numpy as np
from PIL import Image

# path to onnx model
ONNX_MODEL_PATH = "waste_sorter_model.onnx"

# same class order you used in fastai (update if different)
CLASS_NAMES = ["Plastic", "Glass", "Metal", "Paper", "Organic"]

# create ONNX session
_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

def preprocess(pil_img: Image.Image) -> np.ndarray:
    # resize to same size used for training
    img = pil_img.convert("RGB").resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0  # scale 0–1

    # if fastai used imagenet normalization, you can add it here:
    # mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    # std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    # arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)  # 1x3x224x224
    return arr

def predict_image(pil_img: Image.Image):
    x = preprocess(pil_img)
    outputs = _session.run(None, {"input": x})
    logits = outputs[0][0]              # shape: (num_classes,)
    probs = np.exp(logits) / np.exp(logits).sum()
    idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    return pred_class, idx, probs, confidence
