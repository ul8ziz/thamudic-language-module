# Thamudic Language Recognition Module

## Project Overview
A deep learning project for recognizing and classifying Thamudic characters using advanced computer vision and neural network techniques. The project implements a custom architecture with attention mechanisms specifically designed for ancient script recognition.

## Technical Components

### Neural Network Architecture
- **Base Model**: ResNet18 with transfer learning
- **Custom Features**:
  - Attention mechanism for focusing on important character features
  - Custom ThamudicFeatureExtractor for local and global feature extraction
  - Label smoothing for better generalization
  - Advanced data augmentation techniques

### Image Processing Pipeline
- Pre-processing steps:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Noise reduction using Non-local Means Denoising
  - Edge enhancement using Canny detection
  - Aspect ratio preservation during resizing
  - Dynamic padding for uniform input size

### Model Architecture Details
- Input: Grayscale images (224x224)
- Feature Extraction Layers:
  - Local feature extraction (32->64 channels)
  - Global feature extraction (64->128 channels)
  - Attention mechanism for feature weighting
- Classification Head:
  - Adaptive pooling
  - Dropout layers (0.5, 0.3)
  - Fully connected layers (256->128->num_classes)

## Prerequisites
- Python 3.9+
- CUDA 11.8
- NVIDIA GPU (GeForce RTX 3050 or better)
- Key Dependencies:
  - PyTorch
  - OpenCV
  - NumPy
  - Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thamudic-language-module.git
cd thamudic-language-module
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
thamudic-language-module/
│
├── src/
│   ├── train.py              # Training script with custom loss functions
│   ├── model.py              # Neural network architecture definition
│   ├── image_processor.py    # Image preprocessing pipeline
│   └── utils/                # Utility functions
│       ├── augmentation.py   # Data augmentation techniques
│       └── metrics.py        # Performance metrics
│
├── data/
│   ├── letters/             # Training dataset
│   └── mappings/            # Character mapping files
│
├── models/                  # Saved model checkpoints
├── logs/                   # Training logs
└── requirements.txt        # Project dependencies
```

## Training the Model

### Dataset Preparation
1. Organize your dataset in the following structure:
```
data/
└── letters/
    ├── letter_1/
    │   ├── img1.png
    │   └── img2.png
    ├── letter_2/
    └── ...
```

2. Create character mappings in `data/mappings/char_mapping.json`

### Training Configuration
```bash
python src/train.py \
    --data_dir data/letters \
    --mapping_file data/mappings/char_mapping.json \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_augmentation True
```

### Training Features
- Label smoothing for better generalization
- Progressive learning rate adjustment
- Early stopping with model checkpointing
- Automatic mixed precision training
- Wandb integration for experiment tracking

## Model Performance
- **Accuracy**: 94.5% on validation set
- **Model Size**: 87MB
- **Inference Time**: ~15ms per image on RTX 3050

## Usage
```python
from src.model import ThamudicRecognitionModel
from src.image_processor import ThamudicImageProcessor

# Initialize processor and model
processor = ThamudicImageProcessor()
model = ThamudicRecognitionModel.load_from_checkpoint('models/latest.pth')

# Process and predict
image = processor.process_image('path/to/image.jpg')
prediction = model.predict(image)
```

## Troubleshooting
- **CUDA Out of Memory**: Reduce batch size or image size
- **Training Instability**: Adjust learning rate or enable gradient clipping
- **Poor Recognition**: Check image preprocessing parameters

## Future Improvements
- [ ] Implement attention visualization
- [ ] Add support for sequence recognition
- [ ] Optimize model for mobile deployment
- [ ] Create web interface for testing
- [ ] Add support for batch processing

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
- Thamudic Language Research Team
- Ancient Script Recognition Community
- Open-source Computer Vision Community
