import torch
import torch.nn as nn
from torch.nn import functional as F


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