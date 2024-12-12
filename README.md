# Thamudic Text Recognition Project

## Project Overview
An advanced optical character recognition (OCR) system for ancient Thamudic inscriptions, leveraging state-of-the-art image processing and machine learning techniques.

## Project Objectives
- Collect and organize Thamudic text images
- Develop robust image preprocessing pipelines
- Create a specialized machine learning model for Thamudic character recognition
- Contribute to archaeological research and linguistic preservation

## Project Structure
```
thamudic_env/
│
├── data/
│   └── letters/
│       ├── thamudic_letters/     # Raw and processed letter images
│       └── letter_mapping.json   # Mapping of Thamudic letter metadata
│
├── Scripts/
│   ├── manage_thamudicletters_images.py  # Image collection and organization
│   └── data_preprocessing.py     # Advanced image preprocessing
│
└── README.md
```

## Technical Requirements
### Software
- Python 3.8+
- Operating System: Windows/Linux/macOS

### Python Dependencies
- Image Processing
  * OpenCV (4.5+)
  * NumPy (1.19+)
  * Albumentations (1.1+)

- Machine Learning
  * scikit-learn (0.24+)
  * TensorFlow (2.6+) or PyTorch (1.10+)

- Additional Tools
  * json
  * typing

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/thamudic-text-recognition.git
cd thamudic-text-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv thamudic_env
source thamudic_env/bin/activate  # On Windows: thamudic_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Features
- Advanced image preprocessing for ancient inscriptions
- Character segmentation and recognition
- Web interface for easy interaction
- Support for both single characters and full inscriptions
- Visualization of recognition results

## Usage

### Image Collection and Organization
```bash
python Scripts/manage_thamudicletters_images.py
```
This script validates, cleans, and organizes Thamudic letter images.

### Data Preprocessing
```bash
python Scripts/data_preprocessing.py
```
Preprocesses images, applies augmentation, and prepares data for model training.

### Training the Model
```bash
python src/train.py
```

### Using the Web Interface
1. Start the web application:
```bash
streamlit run src/app.py
```
2. Open your web browser and navigate to the displayed URL
3. Upload an image of a Thamudic inscription
4. View the recognition results and visualization

### Using the Command Line Interface
```bash
python src/predict.py --image_path path/to/your/image.png
```

## Project Components

### 1. Image Preprocessing (`image_preprocessing.py`)
- Image enhancement
- Character segmentation
- Data augmentation

### 2. Model Architecture (`model.py`)
- CNN-based recognition model
- Support for multiple character classes
- Training and validation pipelines

### 3. Prediction System (`predict.py`)
- Single character recognition
- Full inscription processing
- Confidence scoring

### 4. Web Interface (`app.py`)
- User-friendly interface
- Real-time processing
- Result visualization

## Project Challenges
- Limited Thamudic language datasets
- Complex image preprocessing requirements
- Variations in ancient inscription styles
- Need for specialized OCR training

## Contribution Guidelines
We welcome contributions in the following areas:
1. Image Dataset Expansion
   - Add high-quality Thamudic inscription images
   - Improve image annotation
   - Validate and verify image metadata

2. Algorithm Development
   - Enhance image preprocessing techniques
   - Develop advanced feature extraction methods
   - Improve character segmentation algorithms

3. Machine Learning Models
   - Experiment with different neural network architectures
   - Implement transfer learning techniques
   - Optimize model performance and accuracy

4. Documentation
   - Improve project documentation
   - Add usage examples
   - Create tutorial guides

## Research and Collaboration
This project aims to bridge computational linguistics with archaeological research. We encourage collaboration with:
- Linguists
- Archaeologists
- Machine Learning Researchers
- Open-source Contributors
