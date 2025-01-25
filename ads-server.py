import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import easyocr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Step 1: Visual Feature Extraction using CLIP
def extract_visual_features(image_path, prompts):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Process the image and prompts
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    # Get similarity scores
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Shape: [1, len(prompts)]
    probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

    # Map prompts to probabilities
    prompt_probs = {prompt: float(probs[0][i]) for i, prompt in enumerate(prompts)}
    return prompt_probs

# Step 2: OCR for Text Extraction using EasyOCR
def extract_text_from_image(image_path):
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    results = reader.readtext(image_path, detail=0)  # Extract only the text, no bounding boxes
    return results

# FastAPI application
app = FastAPI()

@app.post("/analyze-image/")
async def analyze_image_api(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Define prompts for visual features
        prompts = ["leaves", "nature", "green tones", "eco-friendly symbols", "logos", "plastic"]

        # Step 1: Extract visual features
        visual_features = extract_visual_features(temp_file_path, prompts)

        # Step 2: Extract text from image
        extracted_text = extract_text_from_image(temp_file_path)

        # Return results as JSON
        return JSONResponse(content={
            "visual_features": visual_features,
            "extracted_text": extracted_text
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Example usage if run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
