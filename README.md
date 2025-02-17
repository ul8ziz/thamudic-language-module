# Thamudic Language Module

## Overview
The Thamudic Language Module is an advanced deep learning application designed for the accurate recognition and analysis of ancient Thamudic inscriptions. It leverages cutting-edge computer vision and neural network methodologies to deliver high-performance character recognition with a user-friendly web interface.

## Key Features
- **Accurate Character Recognition**: Employs state-of-the-art deep learning models for precise identification of Thamudic characters
- **Interactive Web Interface**: Built with Streamlit for easy interaction and visualization
- **Advanced Image Preprocessing**: Sophisticated techniques to enhance image quality
- **Real-Time Processing**: Process images in real-time with immediate feedback
- **Comprehensive Analysis**: Detailed insights and visual representations of recognition results
- **Multi-Platform Support**: Works across different operating systems

## Key Libraries

The Thamudic Language Module leverages several powerful libraries to deliver high-performance character recognition and analysis:

### Deep Learning & Neural Networks
- **PyTorch (`torch`)**: Primary deep learning framework for model development and training
- **TensorFlow**: Provides additional neural network capabilities and model optimization
- **torchvision**: Computer vision extensions for PyTorch
- **Weights & Biases (`wandb`)**: Experiment tracking and model management

### Image Processing
- **OpenCV (`cv2`)**: Advanced image processing and computer vision operations
- **Pillow (`PIL`)**: Image manipulation and enhancement
- **Albumentations**: Advanced image augmentation library for machine learning
- **scikit-image**: Additional image processing utilities

### Data Science & Machine Learning
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and utilities
- **SciPy**: Scientific and technical computing

### Web Application & Visualization
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization
- **python-dotenv**: Environment variable management

### Utility Libraries
- **tqdm**: Progress bar for long-running operations
- **PyYAML**: YAML configuration file parsing
- **pathlib**: Cross-platform path manipulation

### Development & Testing
- **pytest**: Unit testing framework
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Static type checking

Each library plays a crucial role in the project's functionality, from data processing and model training to web interface development and code quality assurance.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel/AMD)
- **RAM**: Minimum 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, but recommended)
- **Storage**: 2GB minimum free space

### Software Requirements
- **Python**: Version 3.10 or higher
- **CUDA Toolkit**: Version 11.2 or higher (for GPU acceleration)
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ul8ziz/thamudic-language-module.git
cd thamudic-language-module
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload an image containing Thamudic inscriptions

4. View the recognition results and analysis

## Project Structure
```
thamudic-language-module/
├── data/                      # Data files
│   ├── letter_mapping.json    # Character mapping definitions
│   └── models/               # Trained model files
│
├── src/                      # Source code
│   ├── app.py               # Main Streamlit application
│   ├── models/              # Model definitions
│   └── utils/               # Utility functions
│
├── tests/                    # Test files
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions and support, please open an issue in the GitHub repository.
