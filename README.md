# Thamudic OCR System

## Overview
The Thamudic OCR System is an advanced deep learning application designed for the accurate recognition of ancient Thamudic inscriptions. It leverages cutting-edge computer vision and neural network methodologies to deliver high-performance character recognition.

## Key Features
- **Accurate Character Recognition**: Employs state-of-the-art models for precise identification of Thamudic characters.
- **Advanced Image Preprocessing**: Utilizes sophisticated techniques to enhance image quality and improve recognition accuracy.
- **Real-Time Processing**: Capable of processing images in real-time, making it suitable for dynamic applications.
- **Comprehensive Analysis and Visualization**: Provides detailed insights and visual representations of recognition results.
- **Support for Full Texts**: Handles both individual character recognition and complete text analysis.

## System Requirements

### Software Requirements
- **Python**: Version 3.10
- **CUDA Toolkit**: Version 11.2 or higher for GPU acceleration
- **Operating System**: Compatible with Windows, Linux, and macOS

### Core Dependencies
Install the necessary libraries using:
```bash
pip install -r requirements.txt
```
Key libraries include:
- TensorFlow 2.6+
- OpenCV 4.5+
- NumPy 1.19+
- scikit-learn 0.24+
- Albumentations 1.1+

## Project Structure
```plaintext
thamudic_ocr/
├── datasets/                    # Dataset files
│   ├── raw_images/             # Original images
│   ├── processed_images/       # Preprocessed images
│   └── metadata/               # Metadata files
│
├── models/                     # Model files
│   ├── checkpoints/           # Model checkpoints
│   ├── final/                 # Final models
│   └── configs/               # Configuration files
│
├── src/                       # Source code
│   ├── core/                 # Core modules
│   ├── data/                 # Data processing
│   ├── evaluation/           # Evaluation tools
│   ├── utils/               # Utility functions
│   └── interface/           # User interfaces
│
├── tests/                     # Tests
├── docs/                      # Documentation
├── requirements.txt          # Dependencies
└── README.md                # Documentation
```

## Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ul8ziz/thamudic-ocr.git
cd thamudic-ocr
```

### 2. Create a Virtual Environment
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
Prepare and organize images using:
```bash
python src/data/dataset_manager.py --input_dir datasets/raw_images --output_dir datasets/processed_images
```

### 2. Model Training
Initiate the training process to develop a robust model capable of recognizing Thamudic inscriptions. This step involves configuring the model parameters and training it on the prepared dataset. Execute the following command to start training:

```bash
python src/core/model_trainer.py --config models/configs/training_config.json
```

#### Prerequisites:
- Ensure that your data is properly prepared and located in the `datasets/processed_images` directory.
- Review and adjust the `training_config.json` file in `models/configs/` to fit your specific training requirements, such as batch size, learning rate, and number of epochs.

#### Expected Outcome:
- The model will be trained using the specified configurations, and checkpoints will be saved in the `models/checkpoints/` directory for later evaluation or further training.

#### Error Handling:
- If you encounter memory errors, consider reducing the batch size or using a machine with more GPU memory.
- For convergence issues, experiment with different learning rates or data augmentation techniques.
- Ensure that all dependencies are installed and compatible with your system configuration.

### 3. Model Evaluation
Evaluate the model's performance:
```bash
python src/evaluation/model_evaluator.py --model_path models/final/best_model.h5
```

### 4. Using the GUI
Launch the graphical interface:
```bash
python src/interface/thamudic_interface.py
```

### 5. Making Predictions
Recognize text in an image:
```bash
python src/core/inference_engine.py --image path/to/image.png
```

## Contributing
We welcome contributions in model improvements, dataset expansion, and documentation enhancements. Please refer to our [contributing guidelines](https://github.com/ul8ziz/thamudic-ocr/blob/main/CONTRIBUTING.md) for more information.

## Support and Contact
For technical issues, please create an [issue](https://github.com/ul8ziz/thamudic-ocr/issues). For contributions, submit a [pull request](https://github.com/ul8ziz/thamudic-ocr/pulls). For questions, check our [discussions](https://github.com/ul8ziz/thamudic-ocr/discussions).

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/ul8ziz/thamudic-ocr/blob/main/LICENSE) file for details.
