from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.inference.predict_risk import run_pipeline

app = FastAPI(
    title="Industrial Visual Inspection API",
    version="1.0"
)

UPLOAD_DIR = "temp"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Root endpoint (health check).
# Used to verify that the FastAPI service is running correctly.
# Returns basic information about the deployed model and API status.
# Accessible through: http://127.0.0.1:8000/

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "YOLO11s"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = run_pipeline(file_path)

    return result