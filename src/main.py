from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.tasks import process_detection
import os
import shutil

app = FastAPI()

# Create necessary directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/results", exist_ok=True)

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    file_path = os.path.join("data/uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Trigger the Chef
    process_detection.delay(file_path)
    
    # Return the expected result name so the user knows what to look for
    result_filename = f"detected_{file.filename}"
    return {
        "status": "Processing",
        "original_file": file.filename,
        "result_file": result_filename,
        "location": f"data/results/{result_filename}"
    }

# New Endpoint: To actually download/see the result in the browser
@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join("data/results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found. The Chef might still be cooking!"}