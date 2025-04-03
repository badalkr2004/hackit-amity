import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Enhanced GPU detection
print("\n=== GPU DIAGNOSTICS ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))
        print(f"GPU {i} capability:", torch.cuda.get_device_capability(i))
    
    # Force current device
    torch.cuda.set_device(0)
    print("Current device:", torch.cuda.current_device())
    print("Device being used:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("WARNING: CUDA is not available. Training will be on CPU only.")
    print("If you have a GPU, please check your PyTorch installation.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

# Test tensor allocation on device
test_tensor = torch.zeros((2, 2)).to(device)
print(f"Test tensor device: {test_tensor.device}")
print("=== END DIAGNOSTICS ===\n")

# Configuration
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

# Feature extraction functions
def extract_features(file_path):
    """Extract MFCCs, mel spectrograms and chroma features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=config.sample_rate)
        
        # Ensure consistent audio length
        if len(y) > config.sample_rate * config.duration:
            y = y[:config.sample_rate * config.duration]
        else:
            y = np.pad(y, (0, max(0, config.sample_rate * config.duration - len(y))), 'constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        # Extract Mel spectrograms
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length
        )
        
        # Normalize features
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
        chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)
        
        # Combine features
        features = np.vstack([mfccs, mel_spec_db, chroma])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        # Determine the fixed size for all spectrograms
        self.fixed_length = config.sample_rate * config.duration // config.hop_length + 1
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = extract_features(file_path)
        
        if features is None:
            # Return a placeholder if extraction failed
            features = np.zeros((config.n_mfcc + config.n_mels + 12, self.fixed_length))
        else:
            # Ensure all features have the same size
            if features.shape[1] > self.fixed_length:
                features = features[:, :self.fixed_length]
            elif features.shape[1] < self.fixed_length:
                # Pad with zeros
                padding = np.zeros((features.shape[0], self.fixed_length - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        if self.transform:
            features = self.transform(features)
            # After transformation, ensure size is still consistent
            if features.shape[1] != self.fixed_length:
                # Resize to fixed length
                if features.shape[1] > self.fixed_length:
                    features = features[:, :self.fixed_length]
                else:
                    padding = torch.zeros((features.shape[0], self.fixed_length - features.shape[1]))
                    features = torch.cat([features, padding], dim=1)
        
        label = self.labels[idx]
        
        return features, label

# Model architecture - CNN with attention mechanism
class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.scale = torch.sqrt(torch.tensor(in_features, dtype=torch.float32))
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attention = F.softmax(attention, dim=2)
        
        output = torch.bmm(attention, v)
        
        return output

class AudioSentimentModel(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size of the feature maps after convolution and pooling
        # This is an approximation and may need adjustment based on your input size
        height = (input_channels // 16)  # After 4 pooling operations
        width = (config.sample_rate * config.duration // config.hop_length + 1) // 16
        
        # Adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((height, width))
        
        # Attention mechanism
        self.attention = AttentionBlock(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * height * width, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Add channel dimension if not present (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Ensure fixed size output
        x = self.adaptive_pool(x)
        
        # Reshape for attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        
        # Apply attention
        x = self.attention(x)
        
        # Flatten and pass through fully connected layers
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Data augmentation transforms
class TimeShift:
    def __init__(self, shift_max=0.2):
        self.shift_max = shift_max
        
    def __call__(self, x):
        shift = int(np.random.uniform(-self.shift_max, self.shift_max) * x.shape[1])
        return torch.roll(x, shift, dims=1)

class PitchShift:
    def __init__(self, max_steps=2):
        self.max_steps = max_steps
        
    def __call__(self, x):
        # Simplified pitch shift simulation - not for production use
        steps = np.random.randint(-self.max_steps, self.max_steps + 1)
        if steps == 0:
            return x
        
        # Apply a simple frequency shift simulation
        # Handle 2D tensor (features x time frames)
        if steps > 0:
            return x[:, steps:]  # Shift right
        else:
            return x[:, :steps]  # Shift left

# Main training pipeline
def main():
    # 1. Data preparation
    print("Preparing data...")
    
    # Load data from CSV file
    df = pd.read_csv(config.csv_path)
    
    # Use the correct column names from the CSV
    file_paths = [os.path.join(config.data_path, file_name) for file_name in df['Filename'].values]
    emotions = df['Class'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_emotions = label_encoder.fit_transform(emotions)
    
    # Train-validation-test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        file_paths, encoded_emotions, test_size=0.2, random_state=SEED, stratify=encoded_emotions
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
    )
    
    # Create augmentation transform
    train_transform = transforms.Compose([
        TimeShift(shift_max=0.2),
        PitchShift(max_steps=2)
    ])
    
    # Create datasets
    train_dataset = AudioDataset(X_train, y_train, transform=train_transform)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # 2. Model initialization
    print("Initializing model...")
    input_channels = config.n_mfcc + config.n_mels + 12  # MFCC + Mel + Chroma
    model = AudioSentimentModel(config.num_classes, input_channels).to(device)
    
    # 3. Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 4. Initialize mixed precision training if available and enabled
    scaler = None
    if config.use_mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    
    # 5. Training loop
    print("Starting training...")
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    # Training function
    def train_epoch():
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                     f"Acc: {100 * correct/total:.2f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    # Validation function
    def validate():
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        return val_loss, val_acc
    
    # Main training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model...")
            torch.save(model.state_dict(), 'best_audio_sentiment_model.pt')
    
    # 6. Evaluate on test set
    print("\nEvaluating model on test set...")
    
    # Load best model
    model.load_state_dict(torch.load('best_audio_sentiment_model.pt'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=label_encoder.classes_,
        digits=4
    )
    
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Classification Report:")
    print(class_report)
    
    # 7. Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed! Model saved as 'best_audio_sentiment_model.pt'")

if __name__ == "__main__":
    main()