from model import model, device
from transformation import Transform
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import joblib

# Load transformation pipeline
transform = Transform()

# Load label encoder
le = joblib.load("label_encoder.pkl")

async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes and convert to PIL
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()
            class_label = le.inverse_transform([class_idx])[0]

        return {"prediction": class_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
