"""
Thamudic Character Recognition Web Application
Streamlit-based web interface for Thamudic character recognition
"""

import os
import torch
import numpy as np
import json
import logging
from pathlib import Path
from PIL import Image, ImageEnhance
import streamlit as st
from models import ThamudicRecognitionModel, ModelCheckpoint
from torchvision import transforms

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

class ThamudicApp:
    """Thamudic Character Recognition Application"""
    
    def __init__(self, model_path: str, mapping_path: str):
        """Initialize the application"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load letter mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
            self.num_classes = len(self.mapping_data['thamudic_letters'])
            self.class_to_symbol = {
                item['name']: item['symbol'] 
                for item in self.mapping_data['thamudic_letters']
            }
        
        # Initialize the model
        self.model = ThamudicRecognitionModel(num_classes=self.num_classes)
        self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load the trained Thamudic recognition model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image, 
                        contrast: float = 2.0, 
                        brightness: float = 1.2,
                        sharpness: float = 1.5) -> torch.Tensor:
        """
        Preprocess the uploaded image
        
        Args:
            image: Uploaded image file
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment factor
            sharpness: Sharpness adjustment factor
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance the image
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        
        # Resize and convert to tensor
        image = image.resize((224, 224))
        tensor = torch.from_numpy(np.array(image)).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor / 255.0
        
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image, 
                contrast: float = 2.0,
                brightness: float = 1.2,
                sharpness: float = 1.5) -> list:
        """
        Make prediction on the preprocessed image
        
        Args:
            image: Uploaded image file
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment factor
            sharpness: Sharpness adjustment factor
        Returns:
            list: List of predicted characters with confidence scores
        """
        try:
            # Preprocess the image
            tensor = self.preprocess_image(
                image, 
                contrast=contrast,
                brightness=brightness,
                sharpness=sharpness
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=3)
            
            # Gather results
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                letter_data = self.mapping_data['thamudic_letters'][idx.item()]
                predictions.append({
                    'letter': letter_data['name'],
                    'symbol': letter_data['symbol'],
                    'confidence': prob.item()
                })
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise

def setup_page():
    """Setup the application page"""
    st.set_page_config(
        page_title="Thamudic Character Recognition",
        page_icon="🔍",
        layout="wide"
    )
    
    # CSS for RTL design
    st.markdown("""
    <style>
        .element-container, .stMarkdown, .stButton, .stText {
            direction: rtl;
            text-align: right;
        }
        .st-emotion-cache-16idsys p {
            direction: rtl;
            text-align: right;
        }
        .stMetricValue, .stMetricLabel {
            direction: rtl;
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    setup_page()
    st.title("Thamudic Character Recognition 🔍")
    st.markdown("---")
    
    try:
        # Initialize paths
        base_dir = Path(__file__).parent
        model_path = base_dir / 'models' / 'checkpoints' / 'best_model.pt'
        mapping_path = base_dir / 'models' / 'configs' / 'letter_mapping.json'
        
        # Initialize the application
        app = ThamudicApp(str(model_path), str(mapping_path))
        st.success("Model loaded successfully!")
        
        # Image processing settings
        with st.expander("⚙️ Image Processing Settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                contrast = st.slider("Contrast", 0.5, 3.0, 2.0, 0.1)
            with col2:
                sharpness = st.slider("Sharpness", 0.5, 3.0, 1.5, 0.1)
            with col3:
                brightness = st.slider("Brightness", 0.5, 2.0, 1.2, 0.1)
        
        # Upload image
        uploaded_file = st.file_uploader(
            "Choose an image for analysis",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image")
            
            with col2:
                st.subheader("Analysis Results")
                with st.spinner('Analyzing image...'):
                    predictions = app.predict(
                        image,
                        contrast=contrast,
                        brightness=brightness,
                        sharpness=sharpness
                    )
                    
                    if predictions:
                        for i, pred in enumerate(predictions, 1):
                            confidence = pred['confidence']
                            
                            # Color the result based on confidence score
                            if confidence >= 0.8:
                                confidence_color = "🟢"
                            elif confidence >= 0.5:
                                confidence_color = "🟡"
                            else:
                                confidence_color = "🔴"
                            
                            st.metric(
                                f"Prediction #{i} {confidence_color}",
                                f"{pred['letter']} ({pred['symbol']})",
                                f"Confidence: {confidence:.2%}"
                            )
                        
                        # Display the full text of the top prediction
                        best_prediction = predictions[0]
                        st.markdown("### Recognized Text")
                        st.markdown(f"""
                        <div style='direction: rtl; text-align: right; font-size: 24px; 
                                    padding: 20px; background-color: #1e3d59; border-radius: 10px;
                                    color: #ffc13b; font-weight: bold; margin: 10px 0;
                                    border: 2px solid #ffc13b;'>
                            <div style='margin-bottom: 10px;'>Arabic Letter: {best_prediction['letter']}</div>
                            <div>Thamudic Symbol: {best_prediction['symbol']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No results found")
                        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in application: {str(e)}")

if __name__ == '__main__':
    main()
