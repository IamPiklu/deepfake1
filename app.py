import os
import io
import zipfile
import base64
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import pipeline

# # Optionally extract examples.zip if needed.
# if os.path.exists("examples.zip"):
#     with zipfile.ZipFile("examples.zip", "r") as zip_ref:
#         zip_ref.extractall(".")

# Initialize the deepfake detection pipeline.
pipe = pipeline(model="./deepfake", trust_remote_code=True)

app = FastAPI(title="Deepfake Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...), true_label: str = Form(...)):
    """
    Expects:
      - file: An image file.
      - true_label: A text field representing the true label.
    
    Returns a JSON response with:
      - confidences: The prediction confidences.
      - true_label: The provided label.
      - face_with_explainability: The explainability image encoded in base64.
    """
    # Read the uploaded image file.
    image_bytes = await file.read()
    
    # Save the image bytes to a temporary file so the pipeline can open it.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    # Run prediction using the pipeline by passing the file path.
    out = pipe.predict(tmp_path)
    confidences = out["confidences"]
    face_with_mask = out["face_with_mask"]

    # Delete the temporary file.
    os.remove(tmp_path)
    
    # Convert the explainability image to base64.
    buffered = io.BytesIO()
    if isinstance(face_with_mask, Image.Image):
        face_with_mask.save(buffered, format="PNG")
    else:
        face_with_mask = Image.fromarray(face_with_mask)
        face_with_mask.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "confidences": confidences,
        "true_label": true_label,
        "face_with_explainability": img_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
