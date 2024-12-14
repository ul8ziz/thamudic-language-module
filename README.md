# Thamudic OCR System (نظام التعرف على النصوص الثمودية)

## Overview
Advanced deep learning system for recognizing ancient Thamudic inscriptions, utilizing state-of-the-art computer vision and neural network techniques.

## Key Features
- ✨ Accurate Thamudic character recognition
- 🖼️ Advanced image preprocessing
- 🚀 Real-time processing capabilities
- 📊 Detailed analysis and visualization
- 🔍 Support for both individual characters and full texts

## System Requirements

### Software Requirements
- Python 3.10
- CUDA Toolkit 11.2+ (for GPU support)
- Operating System: Windows/Linux/macOS

### Core Dependencies
```bash
pip install -r requirements.txt
```

Key libraries:
- TensorFlow 2.6+
- OpenCV 4.5+
- NumPy 1.19+
- scikit-learn 0.24+
- Albumentations 1.1+

## Project Structure
```plaintext
thamudic_ocr/
├── datasets/                    # All dataset related files
│   ├── raw_images/             # Original inscription images
│   ├── processed_images/       # Preprocessed images
│   └── metadata/               # Dataset metadata files
│       └── character_map.json  # Character mapping information
│
├── models/                     # Model related files
│   ├── checkpoints/           # Saved model checkpoints
│   ├── final/                 # Production models
│   └── configs/               # Model configurations
│
├── src/                       # Source code
│   ├── core/                 # Core functionality
│   │   ├── cnn_architecture.py     # Neural network architecture
│   │   ├── model_trainer.py        # Training system
│   │   └── inference_engine.py     # Prediction engine
│   │
│   ├── data/                 # Data processing
│   │   ├── data_pipeline.py        # Data processing pipeline
│   │   ├── image_augmentation.py   # Image enhancement
│   │   └── dataset_manager.py      # Dataset management
│   │
│   ├── evaluation/           # Evaluation tools
│   │   ├── model_evaluator.py      # Model evaluation
│   │   └── performance_metrics.py  # Performance analysis
│   │
│   ├── utils/               # Utility functions
│   │   ├── training_helpers.py     # Training utilities
│   │   └── model_config.py         # Configuration
│   │
│   └── interface/           # User interfaces
│       └── thamudic_interface.py   # GUI application
│
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/ul8ziz/thamudic-ocr.git
cd thamudic-ocr
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .thamudic_env
.\thamudic_env\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .thamudic_env
source .thamudic_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Data Preparation
```bash
# Process and organize images
python src/data/dataset_manager.py --input_dir datasets/raw_images --output_dir datasets/processed_images
```

### 2. Model Training
```bash
# Train the model
python src/core/model_trainer.py --config models/configs/training_config.json
```

### 3. Model Evaluation
```bash
# Evaluate model performance
python src/evaluation/model_evaluator.py --model_path models/final/best_model.h5
```

### 4. Using the GUI
```bash
# Launch the graphical interface
python src/interface/thamudic_interface.py
```

### 5. Making Predictions
```bash
# Recognize text in an image
python src/core/inference_engine.py --image path/to/image.png
```

## Core Components Description

### Data Processing (`src/data/`)
- `data_pipeline.py`: Streamlined data processing workflow
  - Image loading and preprocessing
  - Data augmentation pipeline
  - Batch processing capabilities

- `image_augmentation.py`: Advanced image enhancement
  - Contrast enhancement
  - Noise reduction
  - Geometric transformations
  - Custom augmentation techniques

- `dataset_manager.py`: Dataset organization
  - Data splitting (train/val/test)
  - Metadata management
  - Dataset statistics

### Model Core (`src/core/`)
- `cnn_architecture.py`: Neural network design
  - MobileNetV2 backbone
  - Attention mechanisms
  - Custom layers for Thamudic recognition

- `model_trainer.py`: Training system
  - Custom training loops
  - Learning rate scheduling
  - Model checkpointing
  - Training visualization

- `inference_engine.py`: Prediction system
  - Real-time inference
  - Batch processing
  - Result visualization

### Evaluation Tools (`src/evaluation/`)
- `model_evaluator.py`: Evaluation suite
  - Performance metrics
  - Error analysis
  - Confusion matrix generation

### Interface (`src/interface/`)
- `thamudic_interface.py`: GUI application
  - User-friendly interface
  - Real-time processing
  - Result visualization
  - Batch processing support

## Model Architecture
The system uses a hybrid architecture:
- MobileNetV2 backbone for efficient feature extraction
- Custom attention mechanisms for character focus
- Specialized layers for Thamudic script recognition
- Advanced data augmentation pipeline

## Contributing
We welcome contributions in:
1. Model Improvements
   - Architecture enhancements
   - Performance optimizations
   - New feature implementations

2. Dataset Expansion
   - New character samples
   - Inscription images
   - Metadata enrichment

3. Documentation
   - Usage examples
   - API documentation
   - Training guides

## Support and Contact
- Technical Issues: Create an [issue](https://github.com/ul8ziz/thamudic-ocr/issues)
- Contributions: Submit a [pull request](https://github.com/ul8ziz/thamudic-ocr/pulls)
- Questions: Check our [discussions](https://github.com/ul8ziz/thamudic-ocr/discussions)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Advanced Training Guide

### Model Parameter Tuning
```python
# Example configuration for advanced training
{
    "model_params": {
        "backbone": "mobilenetv2",
        "attention_type": "self_attention",
        "dropout_rate": 0.5,
        "learning_rate": 0.001
    },
    "training_params": {
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5
    }
}
```

### Performance Optimization Strategies
1. Data Preprocessing
   - Image normalization and standardization
   - Advanced augmentation techniques
   - Class balancing strategies

2. Model Architecture
   - Feature extraction optimization
   - Attention mechanism tuning
   - Custom loss functions

3. Training Process
   - Learning rate scheduling
   - Gradient clipping
   - Mixed precision training

### Advanced Data Processing
```python
# Example of advanced data processing pipeline
from src.data.data_pipeline import AdvancedImageProcessor

processor = AdvancedImageProcessor(
    normalize=True,
    augment=True,
    denoise=True
)

# Apply advanced processing
processed_image = processor.process(
    image,
    enhance_contrast=True,
    remove_background=True,
    sharpen=True
)
```

## Troubleshooting Guide

### Common Issues and Solutions

1. Training Issues
   - **Problem**: Model not converging
     - **Solution**: Check learning rate, batch size, and data normalization
   
   - **Problem**: Overfitting
     - **Solution**: Increase dropout, add regularization, or use data augmentation
   
   - **Problem**: GPU memory errors
     - **Solution**: Reduce batch size or use gradient accumulation

2. Data Processing Issues
   - **Problem**: Poor image quality
     - **Solution**: Use advanced preprocessing techniques
   
   - **Problem**: Unbalanced dataset
     - **Solution**: Apply class weights or use data augmentation
   
   - **Problem**: Slow data pipeline
     - **Solution**: Optimize data loading and preprocessing

3. Inference Issues
   - **Problem**: Slow prediction speed
     - **Solution**: Use model quantization or batch processing
   
   - **Problem**: Poor accuracy on new data
     - **Solution**: Fine-tune model or enhance preprocessing

## Utility Tools

### Dataset Management Tools
```bash
# Generate dataset statistics
python src/utils/dataset_analyzer.py --data_dir datasets/processed_images

# Clean and validate dataset
python src/utils/dataset_cleaner.py --data_dir datasets/raw_images

# Convert dataset format
python src/utils/dataset_converter.py --input_format jpg --output_format png
```

### Model Analysis Tools
```bash
# Analyze model performance
python src/utils/model_analyzer.py --model_path models/final/best_model.h5

# Generate confusion matrix
python src/utils/confusion_matrix_generator.py --test_data datasets/test

# Profile model inference
python src/utils/model_profiler.py --batch_sizes 1,8,16,32
```

## Advanced Evaluation Guide

### Performance Metrics
1. Character-Level Metrics
   - Per-character accuracy
   - Confusion matrix analysis
   - Error pattern analysis

2. Text-Level Metrics
   - Word error rate (WER)
   - Character error rate (CER)
   - BLEU score for sequence evaluation

3. System Performance
   - Inference speed
   - Memory usage
   - GPU utilization

### Results Interpretation
```python
# Example of detailed evaluation
from src.evaluation.advanced_metrics import DetailedEvaluator

evaluator = DetailedEvaluator(model_path='models/final/best_model.h5')

# Get comprehensive evaluation results
results = evaluator.evaluate(
    test_data,
    metrics=['accuracy', 'precision', 'recall', 'f1'],
    generate_plots=True,
    save_results=True
)
```

### Best Practices
1. Evaluation Setup
   - Use consistent test sets
   - Implement cross-validation
   - Monitor system resources

2. Results Reporting
   - Document all parameters
   - Include confidence intervals
   - Provide example outputs

3. Performance Optimization
   - Profile bottlenecks
   - Optimize critical paths
   - Monitor resource usage
