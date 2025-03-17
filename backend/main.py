from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from backend.model import get_image_embedding
from backend.faiss_index import add_image, search_similar_images

app = FastAPI()

@app.post("/add_image/")
async def add_image_endpoint(file: UploadFile = File(...)):
    """Adds an X-ray image to FAISS."""
    try:
        # Process Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_embedding = get_image_embedding(image)

        # Store in FAISS
        add_image(image_embedding, file.filename)

        return {"message": "Image added successfully!"}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}

@app.post("/search_by_image/")
async def search_by_image(file: UploadFile = File(...)):
    """Finds similar cases using an image query."""
    try:
        # ✅ Ensure file bytes are properly read
        image_bytes = await file.read()

        # ✅ Debugging: Check if the file has content
        if not image_bytes:
            raise ValueError("Uploaded file is empty!")

        # ✅ Convert bytes to an image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print(f"✅ Image received: {image.size}")  # Debugging

        # Generate CLIP embedding
        query_embedding = get_image_embedding(image)

        # Search FAISS for similar images
        similar_images, distances = search_similar_images(query_embedding)

        return {"similar_images": similar_images, "distances": distances}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"error": str(e)}