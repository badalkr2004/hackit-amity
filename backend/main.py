from typing import Union
import torch
from fastapi import FastAPI, HTTPException,File,UploadFile
from transformers import pipeline
from apptypes import Text
from lib.model import load_model
from lib.audio_utils import preprocess_audio,get_sentiment_label




app = FastAPI()
# Load trained model
classifier = pipeline("text-classification", model="./models/sentiment_model",device=0)
model, device = load_model()  # device is either "cuda" or "cpu"
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
def read_item(data: Text):
   
    try:
        with torch.no_grad():  # Ensure no gradients are calculated for inference
            res = classifier(data.text)
            return {"label": res[0]["label"], "score": res[0]["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Preprocess audio file
        audio_tensor = preprocess_audio(file, device)
        
        # Perform inference
        with torch.no_grad():
            output = model(audio_tensor)
        
        # Get predicted sentiment
        sentiment = get_sentiment_label(output)

        return {"sentiment": sentiment, "raw_output": output.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")