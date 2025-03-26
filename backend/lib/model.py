import torch
import os
from lib.model_architecture import AudioSentimentModel
# from lib.cuda_avail import cudaCheck


class Config:
    # Data parameters
    data_path = "datasets/audio/TRAIN"  # Path to audio files
    csv_path = "datasets/audio/TRAIN.csv"  # Path to CSV file
    sample_rate = 22050
    duration = 3  # seconds
    n_mfcc = 40
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    
    # Training parameters
    batch_size = 32
    epochs = 30
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # Model parameters
    num_classes = 3  # positive, negative, neutral
    
    # Mixed precision training to optimize GPU memory usage for RTX 3050
    use_mixed_precision = True

config = Config()
# Load the model
def load_model(model_filename="best_audio_sentiment_model.pt"):

     # Get absolute path of the model file in the lib directory
    model_path = os.path.join(os.path.dirname(__file__), model_filename)

    # Check if file exists before loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_channels = config.n_mfcc + config.n_mels + 12  # MFCC + Mel + Chroma
    model = AudioSentimentModel(config.num_classes, input_channels).to(device)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()  # Set to evaluation mode
    return model, device
