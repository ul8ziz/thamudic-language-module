import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = torch.relu(out)
        
        return out

class ThamudicRecognitionModel(nn.Module):
    def __init__(self, num_classes=26):
        super(ThamudicRecognitionModel, self).__init__()
        
        # First convolutional block with residual connection
        self.conv1 = ResidualBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second convolutional block with residual connection
        self.conv2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Third convolutional block with residual connection
        self.conv3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1280, 128)  
        self.bn4 = nn.BatchNorm1d(128)  
        self.dropout4 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, num_classes)  
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Dense layers
        x = self.flatten(x)
        # Ensure the flattened size matches the input size of dense1
        if x.shape[1] != 1280:
            print(f"Warning: Flattened size {x.shape[1]} does not match expected size 1280")
            x = torch.nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 1280).squeeze(1)
        
        x = self.dense1(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        
        return x
