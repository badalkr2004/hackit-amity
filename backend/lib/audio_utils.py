import torch
import librosa
from io import BytesIO
from fastapi import UploadFile, HTTPException

# Function to preprocess audio
def preprocess_audio(file: UploadFile, device):
    try:
        # Read audio file as bytes
        audio_bytes = file.file.read()
        audio_stream = BytesIO(audio_bytes)
        
        # Load and convert audio to mono & 16kHz
        waveform, sample_rate = librosa.load(audio_stream, sr=16000, mono=True)
        
        # Convert to tensor & add batch dimension
        tensor_audio = torch.tensor(waveform).unsqueeze(0).to(device)
        return tensor_audio
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")

# Function to get sentiment label from model output
def get_sentiment_label(output):
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_class = torch.argmax(output, dim=1).item()
    return sentiment_labels.get(predicted_class, "Unknown")
