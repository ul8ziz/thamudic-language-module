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
        self.model = ThamudicRecognitionModel(num_classes=len(self.mapping_data['thamudic_letters']))
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
            x, y, w, h = box
            box_left = x
            box_right = x + w
            box_top = y
            box_bottom = y + h
            
            # Draw a thinner green box
            box_color = (0, 255, 0)  # Green color
            box_width = 1  # Thinner line
            draw.rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                outline=box_color,
                width=box_width
            )
            
            try:
                # Add the Arabic letter with adjusted position
                font_size = min(h, w) // 2  # Adjusted font size
                font_path = str(Path(__file__).parent / 'assets' / 'fonts' / 'NotoSansArabic-Regular.ttf')
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
                
                # Calculate text position
                text = self.letter_mapping[prediction]['letter']
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Center text above the box
                text_x = box_left + (w - text_width) // 2
                text_y = box_top - text_height - 5  # Small gap above box
                
                # Draw text
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            except Exception as e:
                logging.error(f"Error drawing text: {str(e)}")
                continue
        
        return result_image 

    def predict(self, image: Image.Image,
                contrast: float = 2.0,
                brightness: float = 1.2,
                sharpness: float = 1.5,
                num_predictions: int = 5,
                image_type: str = 'white_background'):  
        """
        Make prediction on the preprocessed image
        
        Args:
            image: Input image
            contrast: Contrast adjustment
            brightness: Brightness adjustment
            sharpness: Sharpness adjustment
            num_predictions: Number of predictions to return
            image_type: Type of image processing to apply:
                       'white_background': For clean images with white background
                       'dark_background': For images with dark background
                       'inscription': For inscription/rock carving images
        """
        try:
            # Convert to grayscale
            gray_image = image.convert('L')
            
            if image_type == 'white_background':
                # Process images with white background
                enhanced = ImageEnhance.Contrast(gray_image).enhance(contrast)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
                
                img_array = np.array(enhanced)
                _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = np.ones((2,2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Invert the image to make letters white
                binary = 255 - binary
                
            elif image_type == 'dark_background':
                # Process images with dark background
                enhanced = ImageEnhance.Contrast(gray_image).enhance(contrast * 1.5)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness * 1.3)
                
                img_array = np.array(enhanced)
                # Use adaptive thresholding for dark backgrounds
                binary = cv2.adaptiveThreshold(
                    img_array, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    21, 10
                )
                
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
            else:  # image_type == 'inscription'
                # Process inscription images
                # Apply local contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(np.array(gray_image))
                
                # Apply a filter to reduce noise
                enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
                
                # Use adaptive thresholding with custom values for inscriptions
                binary = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    25, 15
                )
                
                # Apply morphological operations to clean the image
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Apply an additional filter to smooth edges
                binary = cv2.medianBlur(binary, 3)
            
            # Find connected components with adjusted connectivity
            connectivity = 4 if image_type == 'white_background' else 8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
            
            boxes = []
            valid_components = []
            
            # Adjust filtering criteria based on image type
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Adjust criteria based on image type
                if image_type == 'white_background':
                    min_area = 50
                    max_area = (image.size[0] * image.size[1]) / 6
                    min_aspect = 0.1
                    max_aspect = 10
                    min_density = 0.05
                    max_density = 0.95
                    margin = 3
                elif image_type == 'dark_background':
                    min_area = 40
                    max_area = (image.size[0] * image.size[1]) / 5
                    min_aspect = 0.08
                    max_aspect = 12
                    min_density = 0.03
                    max_density = 0.97
                    margin = 4
                else:  # inscription
                    min_area = 30
                    max_area = (image.size[0] * image.size[1]) / 4
                    min_aspect = 0.05
                    max_aspect = 15
                    min_density = 0.02
                    max_density = 0.98
                    margin = 5
                
                aspect_ratio = w / h if h > 0 else 0
                
                if (min_area < area < max_area and 
                    min_aspect < aspect_ratio < max_aspect):
                    
                    component_mask = (labels == i).astype(np.uint8)
                    black_pixels = np.sum(component_mask)
                    density = black_pixels / (w * h)
                    
                    if min_density < density < max_density:
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(image.size[0] - x, w + 2 * margin)
                        h = min(image.size[1] - y, h + 2 * margin)
                        boxes.append((x, y, w, h))
                        valid_components.append(i)
            
            # Sort boxes from right to left
            boxes.sort(key=lambda box: -(box[0] + box[2]))
            
            # Prepare letter images for prediction
            letter_images = []
            final_boxes = []
            for box in boxes:
                x, y, w, h = box
                letter_image = image.crop((x, y, x+w, y+h))
                letter_images.append(letter_image)
                final_boxes.append(box)
            
            # Adjust confidence threshold based on image type
            confidence_threshold = {
                'white_background': 0.4,
                'dark_background': 0.35,
                'inscription': 0.3
            }.get(image_type, 0.4)
            
            # Predict letters
            predictions = []
            confidences = []
            for letter_image in letter_images:
                processed_image = self.preprocess_image(letter_image)
                
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                    
                    if confidence.item() > confidence_threshold:
                        predictions.append(prediction.item())
                        confidences.append(confidence.item())
                    else:
                        predictions.append(-1)
                        confidences.append(0.0)
            
            # Filter out low-confidence predictions
            final_predictions = []
            final_confidences = []
            filtered_boxes = []
            for pred, conf, box in zip(predictions, confidences, final_boxes):
                if pred != -1:
                    final_predictions.append(pred)
                    final_confidences.append(conf)
                    filtered_boxes.append(box)
            
            # Draw result on image
            result_image = self.draw_result_on_image(image, filtered_boxes, final_predictions)
            
            return final_predictions, final_confidences, result_image
            
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
        model_path = base_dir / 'models' / 'checkpoints' / 'best_model.pth'
        mapping_path = base_dir / 'data' / 'letter_mapping.json'
        
        # Initialize the application
        app = ThamudicApp(str(model_path), str(mapping_path))
        st.success("Model loaded successfully!")
        
        # Create two columns: left for settings, right for results
        col_settings, col_results = st.columns([1, 2])
        
        with col_settings:
            st.subheader("⚙️ Recognition Settings")
            # إضافة اختيار نوع الصورة
            image_type = st.selectbox(
                "نوع الصورة",
                options=['white_background', 'dark_background', 'inscription'],
                format_func=lambda x: {
                    'white_background': 'خلفية بيضاء (صور نظيفة)',
                    'dark_background': 'خلفية سوداء',
                    'inscription': 'نقوش صخرية'
                }[x]
            )
            
            # Image processing settings
            contrast = st.slider("Contrast", 0.5, 3.0, 2.0, 0.1)
            brightness = st.slider("Brightness", 0.5, 3.0, 1.2, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 3.0, 1.5, 0.1)
        
        with col_results:
            st.subheader("Recognition Results")
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
            
            # Create placeholder for image and results
            image_placeholder = st.empty()
            letters_container = st.container()
            
            if uploaded_file is not None:
                # Read the image
                image = Image.open(uploaded_file)
                
                # Process image automatically
                with st.spinner('Processing...'):
                    # Make prediction
                    predictions, confidences, result_image = app.predict(
                        image,
                        contrast=contrast,
                        brightness=brightness,
                        sharpness=sharpness,
                        image_type=image_type
                    )
                    
                    # Display result image
                    image_placeholder.image(result_image, use_container_width=True)
                    
                    # Display detected letters
                    if predictions:
                        with letters_container:
                            st.markdown("### Detected Letters")
                            # Use columns to display letters in a grid
                            letter_cols = st.columns(3)
                            for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                                letter_info = app.letter_mapping[pred]
                                with letter_cols[idx % 3]:
                                    st.markdown(
                                        f"""
                                        <div style="
                                            padding: 10px;
                                            border-radius: 5px;
                                            background-color: rgba(30, 61, 89, 0.9);
                                            color: white;
                                            margin: 5px 0;
                                            text-align: center;
                                            border: 1px solid #ffc13b;">
                                            <h4 style="margin:0;font-size:1.5em;">{letter_info['letter']}</h4>
                                            <p style="margin:0;font-size:0.9em;">({letter_info['symbol']})</p>
                                            <p style="margin:0;font-size:0.8em;color:#ffc13b;">
                                                {conf:.1%}
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                    else:
                        with letters_container:
                            st.warning("No letters detected in the image.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in application: {str(e)}")

if __name__ == '__main__':
    main()
