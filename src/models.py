"""
Neural network model for Thamudic character recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from torch.optim import AdamW

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ThamudicRecognitionModel(nn.Module):
    """Thamudic character recognition model with SE-ResNet architecture"""
    def __init__(self, num_classes=28, input_channels=3):
        super(ThamudicRecognitionModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and final dense layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling and final dense layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def save_checkpoint(self, save_path, **kwargs):
        """Save model checkpoint with additional metadata"""
        # Create parent directories if they don't exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'metadata': kwargs
        }
        
        try:
            torch.save(checkpoint, save_path)
            logging.info(f"Checkpoint saved successfully to {save_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and return metadata with partial loading support"""
        try:
            # Use a context manager for safe loading
            with torch.serialization.safe_globals([AdamW]):
                checkpoint = torch.load(checkpoint_path, 
                                     map_location=torch.device('cpu'),
                                     weights_only=False)
            
            # Determine checkpoint state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metadata = checkpoint.get('metadata', {})
            else:
                state_dict = checkpoint
                metadata = {}
            
            # Get current model's state dict
            model_state_dict = self.state_dict()
            
            # Create a new state dict for loading
            new_state_dict = model_state_dict.copy()
            
            # Partially load state dict, handling class number mismatch
            for name, param in state_dict.items():
                if name in model_state_dict:
                    # Handle final layer mismatch
                    if 'fc.4' in name:  # Last layer weights/bias
                        current_shape = model_state_dict[name].shape
                        checkpoint_shape = param.shape
                        
                        if current_shape != checkpoint_shape:
                            logging.warning(f"Mismatch in {name}: checkpoint shape {checkpoint_shape}, current model shape {current_shape}")
                            
                            # For weights
                            if len(current_shape) == 2 and len(checkpoint_shape) == 2:
                                min_classes = min(current_shape[0], checkpoint_shape[0])
                                new_state_dict[name][:min_classes, :] = param[:min_classes, :]
                            
                            # For biases
                            elif len(current_shape) == 1 and len(checkpoint_shape) == 1:
                                min_classes = min(current_shape[0], checkpoint_shape[0])
                                new_state_dict[name][:min_classes] = param[:min_classes]
                            
                            continue
                    
                    # Load other parameters normally
                    new_state_dict[name] = param
            
            # Load the modified state dict
            self.load_state_dict(new_state_dict, strict=False)
            
            logging.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            return metadata
        
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            raise
