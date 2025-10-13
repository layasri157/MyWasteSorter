from fastai.vision.all import *

# Load your trained model
learn = load_learner('waste_sorter_model.pkl')

# Path to the image you want to test (keep this image in the same folder)
img_path = 'test_image.jpg'

# Load the image
img = PILImage.create(img_path)

# Predict the category and confidence
pred, pred_idx, probs = learn.predict(img)

print(f'Prediction: {pred}, Confidence: {probs[pred_idx]:.4f}')
