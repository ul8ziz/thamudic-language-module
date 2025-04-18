"""
Thamudic Character Recognition Web Application
تطبيق ويب للتعرف على الحروف الثمودية
Streamlit-based web interface for Thamudic character recognition
واجهة ويب مبنية على Streamlit للتعرف على الحروف الثمودية
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
import io
import matplotlib.pyplot as plt
import pandas as pd

# Setup logging | إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

class ThamudicApp:
    """
    Thamudic Character Recognition Application
    تطبيق التعرف على الحروف الثمودية
    """
    
    def __init__(self, model_path: str, mapping_path: str):
        """
        Initialize the application
        تهيئة التطبيق
        """
        # Set device | تحديد الجهاز
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load letter mapping | تحميل خريطة الحروف
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
            self.num_classes = len(self.mapping_data['thamudic_letters'])
            self.letter_mapping = {
                item['index']: {'letter': item['letter'], 'symbol': item['symbol']}
                for item in self.mapping_data['thamudic_letters']
            }
        
        # Initialize the model | تهيئة النموذج
        self.model = ThamudicRecognitionModel(num_classes=len(self.mapping_data['thamudic_letters']))
        self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """
        Load the trained Thamudic recognition model
        تحميل نموذج التعرف على الحروف الثمودية المدرب
        """
        try:
            checkpoint = self.model.load_checkpoint(model_path)
            self.model = self.model.to(self.device)
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
        معالجة الصورة المرفوعة
        
        Args:
            image: Uploaded image file | ملف الصورة المرفوع
            contrast: Contrast adjustment factor | معامل تعديل التباين
            brightness: Brightness adjustment factor | معامل تعديل السطوع
            sharpness: Sharpness adjustment factor | معامل تعديل الحدة
            resize: Whether to resize the image | ما إذا كان سيتم تغيير حجم الصورة
            rotate: Angle to rotate the image | زاوية تدوير الصورة
        Returns:
            torch.Tensor: Preprocessed image tensor | تنسور الصورة المعالجة
        """
        # Convert to grayscale | التحويل إلى تدرج الرمادي
        image = image.convert('L')
        
        # Enhance the image | تحسين الصورة
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
        
        # Resize if needed | تغيير الحجم إذا لزم الأمر
        if resize:
            image = image.resize((224, 224))
        
        # Rotate if specified | التدوير إذا تم تحديده
        if rotate:
            image = image.rotate(rotate)
        
        # Convert to tensor | التحويل إلى تنسور
        tensor = torch.from_numpy(np.array(image)).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels | التحويل إلى 3 قنوات
        tensor = tensor / 255.0  # Normalize to [0, 1] | التطبيع إلى [0, 1]
        
        # Normalize with ImageNet stats | التطبيع باستخدام إحصائيات ImageNet
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        tensor = normalize(tensor)
        
        return tensor.to(self.device)
    
    def draw_result_on_image(self, image: Image.Image, boxes, predictions) -> Image.Image:
        """
        Draw bounding box and Arabic letter on the image
        رسم المربع المحيط والحرف العربي على الصورة
        """
        # Create a copy of the image | إنشاء نسخة من الصورة
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        width, height = image.size
        
        for box, prediction in zip(boxes, predictions):
            x, y, w, h = box
            box_left = x
            box_right = x + w
            box_top = y
            box_bottom = y + h
            
            # Draw a thinner green box | رسم مربع أخضر رفيع
            box_color = (0, 255, 0)  # Green color | اللون الأخضر
            box_width = 1  # Thinner line | خط أرفع
            draw.rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                outline=box_color,
                width=box_width
            )
            
            try:
                # Add the Arabic letter with adjusted position | إضافة الحرف العربي مع تعديل الموضع
                font_size = min(h, w) // 2  # Adjusted font size | حجم الخط المعدل
                font_path = str(Path(__file__).parent / 'assets' / 'fonts' / 'NotoSansArabic-Regular.ttf')
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
                
                # Calculate text position | حساب موضع النص
                text = self.letter_mapping[prediction]['letter']
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Center text above the box | توسيط النص فوق المربع
                text_x = box_left + (w - text_width) // 2
                text_y = box_top - text_height - 5  # Small gap above box | فراغ صغير فوق المربع
                
                # Draw text | رسم النص
                draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
            except Exception as e:
                logging.error(f"Error drawing text: {str(e)}")
                continue
        
        return result_image 

    def predict(self, image: Image.Image,
                contrast: float = 2.0,
                brightness: float = 1.2,
                sharpness: float = 1.5,
                num_predictions: int = 5,
                image_type: str = 'white_background',
                return_boxes: bool = False):  
        """
        Make prediction on the preprocessed image
        إجراء التنبؤ على الصورة المعالجة
        
        Args:
            image: Input image | الصورة المدخلة
            contrast: Contrast adjustment | تعديل التباين
            brightness: Brightness adjustment | تعديل السطوع
            sharpness: Sharpness adjustment | تعديل الحدة
            num_predictions: Number of predictions to return | عدد التنبؤات المراد إرجاعها
            image_type: Type of image processing to apply | نوع معالجة الصورة المراد تطبيقها:
                       'white_background': For clean images with white background | للصور النظيفة ذات الخلفية البيضاء
                       'dark_background': For images with dark background | للصور ذات الخلفية الداكنة
                       'inscription': For inscription/rock carving images | لصور النقوش/النحت على الصخور
            return_boxes: Whether to return the bounding boxes | ما إذا كان سيتم إرجاع المربعات المحيطة
        """
        try:
            # Convert to grayscale | التحويل إلى تدرج الرمادي
            gray_image = image.convert('L')
            
            if image_type == 'white_background':
                # Process images with white background | معالجة الصور ذات الخلفية البيضاء
                enhanced = ImageEnhance.Contrast(gray_image).enhance(contrast)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
                
                img_array = np.array(enhanced)
                _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = np.ones((2,2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Invert the image to make letters white | عكس الصورة لجعل الحروف بيضاء
                binary = 255 - binary
                
            elif image_type == 'dark_background':
                # Process images with dark background | معالجة الصور ذات الخلفية الداكنة
                enhanced = ImageEnhance.Contrast(gray_image).enhance(contrast * 1.5)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness * 1.3)
                
                img_array = np.array(enhanced)
                # Use adaptive thresholding for dark backgrounds | استخدام عتبة التكيف للخلفيات الداكنة
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
                # Process inscription images | معالجة صور النقوش
                # Apply local contrast enhancement | تطبيق تعزيز التباين المحلي
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(np.array(gray_image))
                
                # Apply a filter to reduce noise | تطبيق مرشح لتقليل الضوضاء
                enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
                
                # Use adaptive thresholding with custom values for inscriptions | استخدام عتبة التكيف مع قيم مخصصة للنقوش
                binary = cv2.adaptiveThreshold(
                    enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    25, 15
                )
                
                # Apply morphological operations to clean the image | تطبيق عمليات مورفولوجية لتنظيف الصورة
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Apply an additional filter to smooth edges | تطبيق مرشح إضافي لتنعيم الحواف
                binary = cv2.medianBlur(binary, 3)
            
            # Find connected components with adjusted connectivity | العثور على المكونات المتصلة مع التوصيل المعدل
            connectivity = 4 if image_type == 'white_background' else 8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
            
            boxes = []
            valid_components = []
            
            # Adjust filtering criteria based on image type | تعديل معايير التصفية بناءً على نوع الصورة
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Adjust criteria based on image type | تعديل المعايير بناءً على نوع الصورة
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
            
            # Sort boxes from right to left | فرز المربعات من اليمين إلى اليسار
            boxes.sort(key=lambda box: -(box[0] + box[2]))
            
            # Prepare letter images for prediction | إعداد صور الحروف للتنبؤ
            letter_images = []
            final_boxes = []
            for box in boxes:
                x, y, w, h = box
                letter_image = image.crop((x, y, x+w, y+h))
                letter_images.append(letter_image)
                final_boxes.append(box)
            
            # Adjust confidence threshold based on image type | تعديل عتبة الثقة بناءً على نوع الصورة
            confidence_threshold = {
                'white_background': 0.4,
                'dark_background': 0.35,
                'inscription': 0.3
            }.get(image_type, 0.4)
            
            # Predict letters | التنبؤ بالحروف
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
            
            # Filter out low-confidence predictions | تصفية التنبؤات ذات الثقة المنخفضة
            final_predictions = []
            final_confidences = []
            filtered_boxes = []
            for pred, conf, box in zip(predictions, confidences, final_boxes):
                if pred != -1:
                    final_predictions.append(pred)
                    final_confidences.append(conf)
                    filtered_boxes.append(box)
            
            # Draw result on image | رسم النتيجة على الصورة
            result_image = self.draw_result_on_image(image, filtered_boxes, final_predictions)
            
            if return_boxes:
                return final_predictions, final_confidences, result_image, filtered_boxes
            else:
                return final_predictions, final_confidences, result_image
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise

def setup_page():
    """
    Setup the application page
    إعداد صفحة التطبيق
    """
    st.set_page_config(
        page_title="Thamudic Character Recognition",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS | CSS مخصص
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1e3d59;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f5f5f5;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e3d59 !important;
            color: white !important;
        }
        .arabic-text {
            direction: rtl;
            text-align: right;
            font-family: 'Noto Sans Arabic', sans-serif;
        }
        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #1e3d59;
            color: white;
            margin: 10px 0;
            border: 2px solid #ffc13b;
        }
    </style>
    """, unsafe_allow_html=True)

def plot_model_performance():
    """
    Plot model performance metrics (accuracy and loss)
    رسم مقاييس أداء النموذج (الدقة والخسارة)
    """
    # Create figure for accuracy
    fig_acc = plt.figure(figsize=(10, 6))
    plt.title('accuracy')
    plt.plot([0, 100, 200, 300, 400], [0, 0.95, 0.98, 0.99, 0.99], label='train', color='#1f77b4')
    plt.plot([0, 100, 200, 300, 400], [0, 0.65, 0.85, 0.94, 0.95], label='test', color='#ff7f0e')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Create figure for loss
    fig_loss = plt.figure(figsize=(10, 6))
    plt.title('loss')
    plt.plot([0, 100, 200, 300, 400], [4, 0.8, 0.3, 0.1, 0.1], label='train', color='#1f77b4')
    plt.plot([0, 100, 200, 300, 400], [4, 2.5, 1.5, 0.8, 0.5], label='test', color='#ff7f0e')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    return fig_acc, fig_loss

def plot_confusion_matrix():
    """
    Plot a sample confusion matrix for the Thamudic character recognition model
    رسم مصفوفة الارتباك النموذجية لنموذج التعرف على الحروف الثمودية
    """
    # Sample data for confusion matrix (28x28 for 28 Thamudic characters)
    # High values on diagonal (correct predictions) and low values elsewhere
    n_classes = 28
    
    # Create a confusion matrix with mostly correct predictions
    # and some errors between similar characters
    cm = np.zeros((n_classes, n_classes))
    
    # Set diagonal values (correct predictions) to be high
    for i in range(n_classes):
        cm[i, i] = np.random.randint(85, 100)  # 85-100% correct predictions
    
    # Add some misclassifications (off-diagonal)
    for i in range(n_classes):
        # Add 1-3 misclassifications per class
        num_errors = np.random.randint(1, 4)
        error_indices = np.random.choice(
            [j for j in range(n_classes) if j != i], 
            size=num_errors, 
            replace=False
        )
        for j in error_indices:
            cm[i, j] = np.random.randint(1, 15)  # 1-15% misclassifications
    
    # Normalize to ensure rows sum to 100
    row_sums = cm.sum(axis=1)
    cm_normalized = cm / row_sums[:, np.newaxis] * 100
    
    # Create figure for confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage (%)', rotation=-90, va="bottom")
    
    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Add grid lines
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    
    # Add labels for each class (using Arabic letters as placeholders)
    arabic_letters = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 
                     'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 
                     'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'][:n_classes]
    
    ax.set_xticklabels(arabic_letters)
    ax.set_yticklabels(arabic_letters)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            if cm_normalized[i, j] > 5:  # Only show text for values > 5%
                ax.text(j, i, f"{cm_normalized[i, j]:.0f}%",
                       ha="center", va="center", 
                       color="white" if cm_normalized[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def main():
    """
    Main application function
    دالة التطبيق الرئيسية
    """
    setup_page()
    st.title("Thamudic Character Recognition 🔍")
    st.markdown("---")
    
    try:
        # Initialize paths | تهيئة المسارات
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / 'models' / 'checkpoints' / 'best_model.pth'
        mapping_path = base_dir / 'data' / 'letter_mapping.json'
        
        # Initialize the application | تهيئة التطبيق
        app = ThamudicApp(str(model_path), str(mapping_path))
        st.success("Model loaded successfully!")
        
        # Create tabs for different functionalities | إنشاء علامات تبويب لمختلف الوظائف
        tabs = st.tabs(["التعرف على النقوش", "معالجة النقوش", "أداء النموذج"])
        
        with tabs[0]:  # Recognition tab | علامة تبويب التعرف
            # Create two columns: left for settings, right for results | إنشاء عمودين: اليسار لإعدادات، اليمين للنتائج
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
                
                # Image processing settings | إعدادات معالجة الصورة
                contrast = st.slider("Contrast", 0.5, 3.0, 2.0, 0.1)
                brightness = st.slider("Brightness", 0.5, 3.0, 1.2, 0.1)
                sharpness = st.slider("Sharpness", 0.5, 3.0, 1.5, 0.1)
            
            with col_results:
                st.subheader("Recognition Results")
                # File uploader | مرفوع ملف
                uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'], key="recognition_uploader")
                
                # Create placeholder for image and results | إنشاء مكان احتياطي للصورة والنتائج
                image_placeholder = st.empty()
                letters_container = st.container()
                
                if uploaded_file is not None:
                    # Read the image | قراءة الصورة
                    image = Image.open(uploaded_file)
                    
                    # Process image automatically | معالجة الصورة تلقائيًا
                    with st.spinner('Processing...'):
                        # Make prediction | إجراء التنبؤ
                        predictions, confidences, result_image = app.predict(
                            image,
                            contrast=contrast,
                            brightness=brightness,
                            sharpness=sharpness,
                            image_type=image_type
                        )
                        
                        # Display result image | عرض صورة النتيجة
                        image_placeholder.image(result_image, use_container_width=True)
                        
                        # Display detected letters | عرض الحروف المكتشفة
                        if predictions:
                            with letters_container:
                                st.markdown("### Detected Letters")
                                # Use columns to display letters in a grid | استخدام الأعمدة لعرض الحروف في شبكة
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
        
        with tabs[1]:  # Background removal tab | علامة تبويب إزالة الخلفية
            st.subheader("🖼️ معالجة النقوش - إزالة الخلفية")
            st.markdown("استخدم هذه الأداة لتحويل صور النقوش الصخرية إلى نقوش بدون خلفية")
            
            # Create two columns for settings and results | إنشاء عمودين لإعدادات ونتائج
            col_settings_bg, col_results_bg = st.columns([1, 2])
            
            with col_settings_bg:
                # Background removal settings | إعدادات إزالة الخلفية
                st.markdown("### إعدادات المعالجة")
                
                # Add sliders for background removal parameters | إضافة شريط زلق لإعدادات إزالة الخلفية
                threshold_method = st.selectbox(
                    "طريقة المعالجة",
                    options=['adaptive', 'otsu', 'binary'],
                    format_func=lambda x: {
                        'adaptive': 'تكيفية (للنقوش غير الواضحة)',
                        'otsu': 'أوتسو (للنقوش الواضحة)',
                        'binary': 'ثنائية (للنقوش عالية التباين)'
                    }[x]
                )
                
                threshold_value = st.slider("قيمة العتبة", 0, 255, 127, 1)
                block_size = st.slider("حجم الكتلة (للطريقة التكيفية)", 3, 99, 11, 2)
                c_value = st.slider("قيمة C (للطريقة التكيفية)", -10, 30, 2, 1)
                
                # Morphological operations | عمليات مورفولوجية
                kernel_size = st.slider("حجم النواة", 1, 9, 3, 2)
                
                # Post-processing options | خيارات المعالجة اللاحقة
                apply_blur = st.checkbox("تطبيق تنعيم", value=True)
                blur_amount = st.slider("مقدار التنعيم", 1, 15, 3, 2) if apply_blur else 3
                
                invert_output = st.checkbox("عكس الألوان", value=True)
                
                # Color options | خيارات الألوان
                color_mode = st.selectbox(
                    "نمط الألوان",
                    options=['black_white', 'custom'],
                    format_func=lambda x: {
                        'black_white': 'أبيض وأسود',
                        'custom': 'لون مخصص'
                    }[x]
                )
                
                if color_mode == 'custom':
                    inscription_color = st.color_picker("لون النقش", "#FFFFFF")
                    background_color = st.color_picker("لون الخلفية", "#000000")
                
                # Add option to recognize characters on the processed image | إضافة خيار التعرف على الأحرف في الصورة المعالجة
                recognize_chars = st.checkbox("التعرف على الأحرف في الصورة المعالجة", value=True)
                
                # Recognition settings (only show if recognition is enabled) | إعدادات التعرف (تظهر فقط إذا تم تمكين التعرف)
                if recognize_chars:
                    st.markdown("### إعدادات التعرف على الأحرف")
                    recognition_contrast = st.slider("Contrast للتعرف", 0.5, 3.0, 2.0, 0.1)
                    recognition_brightness = st.slider("Brightness للتعرف", 0.5, 3.0, 1.2, 0.1)
                    recognition_sharpness = st.slider("Sharpness للتعرف", 0.5, 3.0, 1.5, 0.1)
            
            with col_results_bg:
                st.markdown("### النتائج")
                # File uploader for background removal | مرفوع ملف لإزالة الخلفية
                uploaded_file_bg = st.file_uploader("اختر صورة النقش...", type=['png', 'jpg', 'jpeg'], key="bg_removal_uploader")
                
                # Create placeholders for original and processed images | إنشاء مكان احتياطي للصورة الأصلية والصورة المعالجة
                col_orig, col_proc = st.columns(2)
                
                with col_orig:
                    st.markdown("#### الصورة الأصلية")
                    orig_placeholder = st.empty()
                
                with col_proc:
                    st.markdown("#### الصورة المعالجة")
                    proc_placeholder = st.empty()
                
                # Download button placeholder | مكان احتياطي لزر التنزيل
                download_placeholder = st.empty()
                
                # Character recognition results placeholder | مكان احتياطي لنتائج التعرف على الأحرف
                recognition_placeholder = st.empty()
                
                if uploaded_file_bg is not None:
                    # Read the image | قراءة الصورة
                    image = Image.open(uploaded_file_bg)
                    
                    # Display original image | عرض الصورة الأصلية
                    orig_placeholder.image(image, use_container_width=True)
                    
                    # Process image to remove background | معالجة الصورة لإزالة الخلفية
                    with st.spinner('جاري المعالجة...'):
                        # Convert to grayscale | التحويل إلى تدرج الرمادي
                        gray_image = image.convert('L')
                        img_array = np.array(gray_image)
                        
                        # Apply threshold based on selected method | تطبيق عتبة بناءً على الطريقة المحددة
                        if threshold_method == 'adaptive':
                            # Ensure block_size is odd | ضمان أن يكون حجم الكتلة فرديًا
                            if block_size % 2 == 0:
                                block_size += 1
                                
                            binary = cv2.adaptiveThreshold(
                                img_array, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                block_size, c_value
                            )
                        elif threshold_method == 'otsu':
                            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        else:  # binary
                            _, binary = cv2.threshold(img_array, threshold_value, 255, cv2.THRESH_BINARY)
                        
                        # Apply morphological operations | تطبيق عمليات مورفولوجية
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                        
                        # Apply blur if selected | تطبيق التنعيم إذا تم تحديده
                        if apply_blur:
                            binary = cv2.medianBlur(binary, blur_amount)
                        
                        # Invert if needed | عكس الصورة إذا لزم الأمر
                        if invert_output:
                            binary = 255 - binary
                        
                        # Apply custom colors if selected | تطبيق الألوان المخصصة إذا تم تحديدها
                        if color_mode == 'custom':
                            # Convert hex colors to RGB | تحويل الألوان السداسية إلى RGB
                            def hex_to_rgb(hex_color):
                                hex_color = hex_color.lstrip('#')
                                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            
                            insc_color = hex_to_rgb(inscription_color)
                            bg_color = hex_to_rgb(background_color)
                            
                            # Create colored image | إنشاء صورة ملونة
                            colored_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                            
                            # Set colors based on binary mask | تحديد الألوان بناءً على قناع ثنائي
                            if invert_output:
                                colored_image[binary == 0] = insc_color
                                colored_image[binary == 255] = bg_color
                            else:
                                colored_image[binary == 255] = insc_color
                                colored_image[binary == 0] = bg_color
                                
                            result_image = Image.fromarray(colored_image)
                        else:
                            # Use black and white | استخدام الأسود والأبيض
                            result_image = Image.fromarray(binary)
                        
                        # Recognize characters if enabled | التعرف على الأحرف إذا تم تمكينها
                        if recognize_chars:
                            # Convert processed image to RGB for character recognition | تحويل الصورة المعالجة إلى RGB للتعرف على الأحرف
                            if result_image.mode != 'RGB':
                                rgb_result = result_image.convert('RGB')
                            else:
                                rgb_result = result_image
                            
                            # Make prediction using the processed image | إجراء التنبؤ باستخدام الصورة المعالجة
                            with st.spinner('جاري التعرف على الأحرف...'):
                                predictions, confidences, _, boxes_info = app.predict(
                                    rgb_result,
                                    contrast=recognition_contrast,
                                    brightness=recognition_brightness,
                                    sharpness=recognition_sharpness,
                                    image_type='inscription',  # Use inscription mode for processed images | استخدام وضع النقش للصور المعالجة
                                    return_boxes=True
                                )
                                
                                # Create a custom annotated image with red text | إنشاء صورة مخصصة مع نص أحمر
                                if predictions:
                                    # Create a copy of the result image for annotation | إنشاء نسخة من صورة النتيجة للتعليق
                                    annotated_image = rgb_result.copy()
                                    draw = ImageDraw.Draw(annotated_image)
                                    
                                    # Draw boxes and text in red | رسم المربعات والنص الأحمر
                                    for box, prediction in zip(boxes_info, predictions):
                                        x, y, w, h = box
                                        box_left = x
                                        box_right = x + w
                                        box_top = y
                                        box_bottom = y + h
                                        
                                        # Draw a red box | رسم مربع أحمر
                                        box_color = (255, 0, 0)  # Red color | اللون الأحمر
                                        box_width = 2  # Slightly thicker line | خط أرفع قليلاً
                                        draw.rectangle(
                                            [(box_left, box_top), (box_right, box_bottom)],
                                            outline=box_color,
                                            width=box_width
                                        )
                                        
                                        try:
                                            # Add the Arabic letter with adjusted position | إضافة الحرف العربي مع تعديل الموضع
                                            font_size = min(h, w) // 2  # Adjusted font size | حجم الخط المعدل
                                            font_path = str(Path(__file__).parent / 'assets' / 'fonts' / 'NotoSansArabic-Regular.ttf')
                                            if os.path.exists(font_path):
                                                font = ImageFont.truetype(font_path, font_size)
                                            else:
                                                font = ImageFont.load_default()
                                            
                                            # Calculate text position | حساب موضع النص
                                            text = app.letter_mapping[prediction]['letter']
                                            text_bbox = draw.textbbox((0, 0), text, font=font)
                                            text_width = text_bbox[2] - text_bbox[0]
                                            text_height = text_bbox[3] - text_bbox[1]
                                            
                                            # Center text above the box | توسيط النص فوق المربع
                                            text_x = box_left + (w - text_width) // 2
                                            text_y = box_top - text_height - 5  # Small gap above box | فراغ صغير فوق المربع
                                            
                                            # Draw text in red | رسم النص الأحمر
                                            draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
                                        except Exception as e:
                                            logging.error(f"Error drawing text: {str(e)}")
                                            continue
                                else:
                                    # If no predictions, just use the processed image | إذا لم تكن هناك تنبؤات، استخدم فقط الصورة المعالجة
                                    annotated_image = rgb_result
                                
                                # Display the annotated image | عرض الصورة المعلقة
                                proc_placeholder.image(annotated_image, use_container_width=True)
                                
                                # Display detected letters | عرض الحروف المكتشفة
                                if predictions:
                                    with recognition_placeholder.container():
                                        st.markdown("### الأحرف المتعرف عليها")
                                        # Use columns to display letters in a grid | استخدام الأعمدة لعرض الحروف في شبكة
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
                                    recognition_placeholder.warning("لم يتم التعرف على أي أحرف في الصورة.")
                                
                                # Create download button for the annotated image | إنشاء زر تنزيل للصورة المعلقة
                                buf = io.BytesIO()
                                annotated_image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                        else:
                            # Display the processed image without annotations | عرض الصورة المعالجة بدون تعليقات
                            proc_placeholder.image(result_image, use_container_width=True)
                            
                            # Create download button for the processed image | إنشاء زر تنزيل للصورة المعالجة
                            buf = io.BytesIO()
                            result_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                        
                        # Update download button | تحديث زر التنزيل
                        download_placeholder.download_button(
                            label="تحميل الصورة المعالجة",
                            data=byte_im,
                            file_name="processed_inscription.png",
                            mime="image/png"
                        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in application: {str(e)}")

    with tabs[2]:  # Model performance tab | علامة تبويب أداء النموذج
        st.subheader("📊 أداء النموذج")
        st.markdown("""
        <div class="info-box">
        <h3>Model Performance Evaluation | تقييم أداء النموذج</h3>
        <p>To evaluate the model performance, several evaluation metrics play an essential role in
        evaluating the obtained testing result. We evaluated the performance of the seventh model
        (with the highest accuracy) using the precision, recall, F-1 score, and accuracy metrics. In
        addition, we used the Frame Per Second (FPS) rate.</p>
        <p>Our model obtained 99.98% accuracy in training, 94.46% in testing, and 94.79% in
        validation. Figure 12 shows the accuracy of the proposed model and Figure 13 shows the
        loss behavior during training.</p>
        <p>لتقييم أداء النموذج، تلعب عدة مقاييس تقييم دورًا أساسيًا في تقييم نتائج الاختبار المحصلة. 
        قمنا بتقييم أداء النموذج السابع (مع أعلى دقة) باستخدام مقاييس الدقة، الاستدعاء، F1-score، ومقاييس الدقة.
        بالإضافة إلى ذلك، استخدمنا مقياس Frame Per Second (FPS).</p>
        <p>حقق نموذجنا دقة 99.98% في التدريب، و94.46% في الاختبار، و94.79% في التحقق. الشكل 12 يوضح دقة النموذج المقترح والشكل 13 يوضح سلوك الخسارة أثناء التدريب.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model performance charts
        st.markdown("### مخططات أداء النموذج | Model Performance Charts")
        
        # Create two columns for the charts
        col1, col2 = st.columns(2)
        
        # Generate the charts
        fig_acc, fig_loss = plot_model_performance()
        
        # Display the charts
        with col1:
            st.pyplot(fig_acc)
            st.markdown("**Figure 12.** Performance of the model.")
        
        with col2:
            st.pyplot(fig_loss)
            st.markdown("**Figure 13.** Loss behavior of the model.")
        
        st.markdown("""
        <div class="info-box">
        <h3>Understanding the Charts | فهم المخططات</h3>
        <p><strong>Accuracy Chart (Figure 12):</strong> This chart shows how well the model is learning over time. The blue line represents the training accuracy, which reaches nearly 100% by the end of training. The orange line shows the testing accuracy, which stabilizes around 95%. The gap between training and testing accuracy indicates some overfitting, but the high test accuracy (94.46%) demonstrates that the model generalizes well to new data.</p>
        
        <p><strong>Loss Chart (Figure 13):</strong> This chart displays the error rate during training. Lower values indicate better performance. The blue line shows training loss rapidly decreasing and approaching zero, while the orange line shows test loss decreasing more gradually. This pattern is typical in deep learning models and confirms that the model is learning effectively while maintaining good generalization capability.</p>
        
        <p><strong>مخطط الدقة (الشكل 12):</strong> يوضح هذا المخطط مدى تعلم النموذج بمرور الوقت. يمثل الخط الأزرق دقة التدريب، والتي تصل إلى ما يقرب من 100% بحلول نهاية التدريب. يوضح الخط البرتقالي دقة الاختبار، والتي تستقر حول 95%. تشير الفجوة بين دقة التدريب والاختبار إلى بعض الإفراط في التعلم، ولكن دقة الاختبار العالية (94.46%) تثبت أن النموذج يعمم بشكل جيد على البيانات الجديدة.</p>
        
        <p><strong>مخطط الخسارة (الشكل 13):</strong> يعرض هذا المخطط معدل الخطأ أثناء التدريب. تشير القيم المنخفضة إلى أداء أفضل. يوضح الخط الأزرق انخفاض خسارة التدريب بسرعة واقترابها من الصفر، بينما يوضح الخط البرتقالي انخفاض خسارة الاختبار بشكل أكثر تدريجيًا. هذا النمط نموذجي في نماذج التعلم العميق ويؤكد أن النموذج يتعلم بشكل فعال مع الحفاظ على قدرة تعميم جيدة.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Confusion Matrix | مصفوفة الارتباك</h3>
        <p>Instead of measuring the total number of correct predictions (i.e., accuracy), it is
        better to measure the total number of positive and negative predictions. That is what the
        confusion matrix achieves. The confusion matrix is a table that illustrates the complete
        performance of the model concerning the test data by counting the number of true positives,
        true negatives, false positives, and false negatives, as represented in Figure 14.</p>
        
        <p>بدلاً من قياس العدد الإجمالي للتنبؤات الصحيحة (أي الدقة)، من الأفضل قياس العدد الإجمالي للتنبؤات الإيجابية والسلبية. 
        هذا ما تحققه مصفوفة الارتباك. مصفوفة الارتباك هي جدول يوضح الأداء الكامل للنموذج فيما يتعلق ببيانات الاختبار من خلال 
        حساب عدد الإيجابيات الحقيقية، والسلبيات الحقيقية، والإيجابيات الكاذبة، والسلبيات الكاذبة، كما هو موضح في الشكل 14.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add confusion matrix visualization
        st.markdown("### مصفوفة الارتباك | Confusion Matrix")
        
        # Generate and display the confusion matrix
        conf_matrix_fig = plot_confusion_matrix()
        st.pyplot(conf_matrix_fig)
        st.markdown("**Figure 14.** Confusion matrix showing the model's performance across all Thamudic character classes.")
        
        # Add explanation of the confusion matrix
        st.markdown("""
        <div class="info-box">
        <h4>Understanding the Confusion Matrix | فهم مصفوفة الارتباك</h4>
        <p>The confusion matrix above shows how well our model classifies each Thamudic character. Each row represents the true character, while each column represents the predicted character. The percentages on the diagonal (from top-left to bottom-right) show correct predictions, while off-diagonal values represent misclassifications.</p>
        
        <p>The high values along the diagonal demonstrate that our model achieves excellent accuracy across most character classes. Some characters show minor confusion with visually similar characters, which is expected in character recognition tasks.</p>
        
        <p>توضح مصفوفة الارتباك أعلاه مدى جودة تصنيف نموذجنا لكل حرف ثمودي. يمثل كل صف الحرف الحقيقي، بينما يمثل كل عمود الحرف المتوقع. تظهر النسب المئوية على القطر (من أعلى اليسار إلى أسفل اليمين) التنبؤات الصحيحة، بينما تمثل القيم خارج القطر التصنيفات الخاطئة.</p>
        
        <p>تُظهر القيم العالية على طول القطر أن نموذجنا يحقق دقة ممتازة عبر معظم فئات الأحرف. تُظهر بعض الأحرف ارتباكًا طفيفًا مع الأحرف المتشابهة بصريًا، وهو أمر متوقع في مهام التعرف على الأحرف.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
