"""
Thamudic Character Recognition Web Application
Streamlit-based web interface for Thamudic character recognition
"""

import os
import logging
import json
import torch
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import streamlit as st
from models import ThamudicRecognitionModel
from torchvision import transforms
from pathlib import Path
import numpy as np
import cv2

# Setup logging
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
            self.letter_mapping = {
                item['index']: {'letter': item['letter'], 'symbol': item['symbol']}
                for item in self.mapping_data['thamudic_letters']
            }
        
        # Initialize the model
        self.model = ThamudicRecognitionModel(num_classes=self.num_classes)
        self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """Load the trained Thamudic recognition model"""
        try:
            checkpoint = self.model.load_checkpoint(model_path)
            self.model = self.model.to(self.device)  # Add this line
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image, 
                        contrast: float = 2.0,
                        brightness: float = 1.2,
                        sharpness: float = 1.5,
                        resize: bool = True,
                        rotate: int = 0) -> torch.Tensor:
        """
        Preprocess the uploaded image
        
        Args:
            image: Uploaded image file
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment factor
            sharpness: Sharpness adjustment factor
            resize: Whether to resize the image
            rotate: Angle to rotate the image
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance the image
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        
        # Resize if needed
        if resize:
            image = image.resize((224, 224))
        
        # Rotate if specified
        if rotate:
            image = image.rotate(rotate)
        
        # Convert to tensor
        tensor = torch.from_numpy(np.array(image)).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Normalize with ImageNet stats
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        tensor = normalize(tensor)
        
        return tensor.to(self.device)
    
    def draw_result_on_image(self, image: Image.Image, boxes, predictions) -> Image.Image:
        """Draw bounding box and Arabic letter on the image"""
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        width, height = image.size
        
        for box, prediction in zip(boxes, predictions):
            # استخدم إحداثيات كل حرف لرسم المربع
            x, y, w, h = box
            box_left = x
            box_right = x + w
            box_top = y
            box_bottom = y + h
            
            # رسم المربع الأخضر
            box_color = (0, 255, 0)  # Green color
            box_width = max(2, min(width, height) // 200)  # تقليل سمك الخط
            draw.rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                outline=box_color,
                width=box_width
            )
            
            try:
                # إضافة الحرف العربي بدون خلفية
                font_size = min(box_right - box_left, height) // 4  # زيادة حجم الخط أكثر
                font_path = str(Path(__file__).parent / 'assets' / 'fonts' / 'NotoSansArabic-Regular.ttf')
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
                
                # حساب موقع النص
                text = self.letter_mapping[prediction]['letter']
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # وضع النص في وسط المربع
                text_x = box_left + (box_right - box_left - text_width) // 2
                text_y = box_top - text_height - 10  # إضافة مسافة صغيرة فوق المربع
                
                # رسم النص بخط أسود بدون خلفية
                draw.text(
                    (text_x, text_y),
                    text,
                    font=font,
                    fill=(0, 0, 0)  # لون النص أسود
                )
                
            except Exception as e:
                logging.error(f"Error drawing text: {str(e)}")
        
        return result_image 

    def predict(self, image: Image.Image,
                contrast: float = 2.0,
                brightness: float = 1.2,
                sharpness: float = 1.5,
                num_predictions: int = 5) -> tuple:
        """
        Make prediction on the preprocessed image
        """
        try:
            # Convert image to numpy array
            image_np = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Ignore background (component 0)
            boxes = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Ignore very small or very large components
                min_area = 100
                max_area = (image.size[0] * image.size[1]) / 8
                if min_area < area < max_area:
                    # Add margin around letter
                    margin = 5
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(image.size[0] - x, w + 2 * margin)
                    h = min(image.size[1] - y, h + 2 * margin)
                    boxes.append((x, y, w, h))
            
            # Sort boxes from left to right
            boxes.sort(key=lambda box: box[0])
            
            # Prepare letter images for prediction
            letter_images = []
            for box in boxes:
                x, y, w, h = box
                letter_image = image.crop((x, y, x+w, y+h))
                letter_images.append(letter_image)
            
            # Predict letters
            predictions = []
            confidences = []  # تأكد من تعريف المتغير هنا
            for letter_image in letter_images:
                processed_image = self.preprocess_image(letter_image)
                
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                    predictions.append(prediction.item())
                    confidences.append(confidence.item())
            
            # Draw result on image
            result_image = self.draw_result_on_image(image, boxes, predictions)
            
            return predictions, result_image
            
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
    
    # CSS for design
    st.markdown("""
    <style>
        .element-container, .stMarkdown, .stButton, .stText {
            text-align: center;
        }
        .stImage > img {
            max-width: 400px;
            margin: auto;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #1e3d59;
            color: white;
            margin: 10px 0;
            border: 2px solid #ffc13b;
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
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / 'models' / 'checkpoints' / 'best_model.pt'
        mapping_path = base_dir / 'data' / 'letter_mapping.json'
        
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
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image")
            
            with col2:
                st.subheader("Analysis Results")
                with st.spinner('Analyzing image...'):
                    predictions, result_image = app.predict(
                        image,
                        contrast=contrast,
                        brightness=brightness,
                        sharpness=sharpness
                    )
                    
                    # Display result image
                    st.image(result_image, caption="Recognition Result")
                    
                    if predictions:
                        for i, pred in enumerate(predictions, 1):
                            confidence = confidences[i-1]
                            
                            # Color the result based on confidence score
                            if confidence >= 0.8:
                                confidence_color = "🟢"
                            elif confidence >= 0.5:
                                confidence_color = "🟡"
                            else:
                                confidence_color = "🔴"
                            
                            st.metric(
                                f"Prediction #{i} {confidence_color}",
                                f"{app.letter_mapping[pred]['letter']} ({app.letter_mapping[pred]['symbol']})",
                                f"Confidence: {confidence:.2%}"
                            )
                        
                        # Display the full text of the top prediction
                        best_prediction = predictions[0]
                        st.markdown("### Recognized Text")
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div style='margin-bottom: 10px;'>Arabic Letter: {app.letter_mapping[best_prediction]['letter']}</div>
                            <div>Thamudic Symbol: {app.letter_mapping[best_prediction]['symbol']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No results found")
                        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in application: {str(e)}")

if __name__ == '__main__':
    main()
