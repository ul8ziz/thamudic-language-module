import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple
import numpy as np

class ThamudicRecognitionModel(nn.Module):
    def __init__(self, num_classes: int):
        super(ThamudicRecognitionModel, self).__init__()
        print(f"Initializing model with {num_classes} classes")
        
        # Use ResNet18 as backbone with pretrained weights
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-8]:
            param.requires_grad = False
        
        # Modify first layer to handle grayscale images
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final layers with custom classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize the new layers
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ThamudicTranslator:
    def __init__(self, model_path: str, char_mapping: Dict[int, str]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Character mapping size: {len(char_mapping)}")
        
        # Initialize model with correct number of classes
        self.model = ThamudicRecognitionModel(len(char_mapping))
        
        # Load model weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        self.char_mapping = char_mapping
        
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict a single character from an image."""
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Return both prediction and confidence
            return self.char_mapping[predicted.item()], confidence.item()
