from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from backend.model import get_image_embedding
from backend.faiss_index import add_embedding, search

app = FastAPI()

@app.post("/add_image/")
async def add_image(file: UploadFile = File(...)):
    """Adds an image to FAISS"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    embedding = get_image_embedding(image)
    add_embedding(embedding)
    return {"message": "Image added to databse"}

@app.post("/search/")
async def search(file: UploadFile = File(...)):
    """Searches for similar images in FAISS"""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    query_embedding = get_image_embedding(image)
    distances, indices = search(query_embedding)
    return {"distances": distances[0].tolist(), "indices": indices[0].tolist()}