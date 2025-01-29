import streamlit as st
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from thamudic_model import ThamudicRecognitionModel
import logging
import cv2
from PIL import ImageFont

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model():
    """
    تحميل نموذج التعرف على الحروف الثمودية
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'best_model.pth')
        mapping_file = os.path.join(base_dir, 'data', 'mapping.json')
        letters_file = os.path.join(base_dir, 'data', 'thamudic_to_arabic.json')
        
        if not os.path.exists(model_path):
            st.error("نموذج التعرف غير موجود. يرجى تدريب النموذج أولاً.")
            return None, None, None
            
        # تحميل التعيين
        with open(mapping_file, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
            
        # تحميل تعيين الحروف العربية
        with open(letters_file, 'r', encoding='utf-8') as f:
            letters_mapping = json.load(f)
            
        # تهيئة النموذج
        model = ThamudicRecognitionModel(num_classes=len(class_mapping))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, class_mapping, letters_mapping
        
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None, None, None

def preprocess_image(image):
    """
    معالجة الصورة للتنبؤ
    """
    # تحويل الصورة إلى مصفوفة numpy
    image_np = np.array(image)
    
    # تحويل الصورة إلى تدرج رمادي
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np
        
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_enhanced = clahe.apply(image_gray)
    
    # تطبيق عتبة تكيفية
    binary = cv2.adaptiveThreshold(
        image_enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # تحويل الصورة الثنائية إلى RGB
    image_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # تحويلات الصورة
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # تطبيق التحويلات
    transformed = transform(image=image_rgb)
    image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
    
    # إضافة بُعد الدفعة
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def detect_letters(image):
    """
    اكتشاف مواقع الحروف في الصورة
    """
    # تحويل الصورة إلى تدرج رمادي
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # تطبيق عتبة تكيفية
    binary = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # تطبيق عمليات مورفولوجية
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # البحث عن المكونات المتصلة
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # تجاهل الخلفية (المكون الأول)
    boxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # تجاهل المكونات الصغيرة جداً أو الكبيرة جداً
        min_area = 100
        max_area = (image.size[0] * image.size[1]) / 8
        if min_area < area < max_area:
            # إضافة هامش حول الحرف
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.size[0] - x, w + 2 * margin)
            h = min(image.size[1] - y, h + 2 * margin)
            boxes.append((x, y, w, h))
    
    # ترتيب الصناديق من اليمين إلى اليسار
    boxes.sort(key=lambda box: box[0], reverse=True)
    
    return boxes

def map_letter_to_thamudic(letter_name, class_mapping):
    """
    Map generic letter names to their corresponding Thamudic character index
    """
    # Reverse the class_mapping to get index to letter_name mapping
    index_to_letter = {v: k for k, v in class_mapping.items()}
    
    # Extract the number from the letter name
    try:
        index = int(letter_name.split('_')[1]) - 1
        return index_to_letter.get(index, letter_name)
    except (IndexError, ValueError):
        return letter_name

def draw_boxes(image, boxes, predictions, letters_mapping, class_mapping):
    """
    Draw bounding boxes around detected letters with predictions
    
    Args:
        image: Input image
        boxes: List of bounding boxes
        predictions: List of predicted classes
        letters_mapping: Mapping from Thamudic to Arabic letters
        class_mapping: Mapping from class names to indices
    """
    try:
        # Create a copy of the image
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Create reverse mappings
        reverse_class_mapping = {v: k for k, v in class_mapping.items()}
        
        # Font settings
        font_size = 20
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw boxes and predictions
        for box, pred in zip(boxes, predictions):
            x1, y1, x2, y2 = box
            
            # Draw rectangle with green color
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)
            
            # Get letter names
            thamudic_letter = reverse_class_mapping[pred]
            arabic_letter = letters_mapping.get(thamudic_letter, "?")
            
            # Calculate text position
            text_width = max(x2 - x1, font_size * 3)
            text_x = x1
            text_y = y1 - font_size - 5
            
            # Draw background rectangle for text
            draw.rectangle([text_x, text_y, text_x + text_width, text_y + font_size],
                         fill='white', outline='lime')
            
            # Draw text
            text = f"{arabic_letter} | {thamudic_letter}"
            draw.text((text_x + 2, text_y), text, fill='black', font=font)
        
        return result_image
        
    except Exception as e:
        logging.error(f"Error in draw_boxes: {str(e)}")
        return image

def main():
    st.set_page_config(
        page_title="مودل التعرف على النصوص الثمودية",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # تطبيق CSS مخصص
    st.markdown("""
    <style>
        /* التنسيق العام */
        .stApp {
            direction: rtl;
            background-color: #f8f9fa;
        }
        
        /* العنوان الرئيسي */
        .main-title {
            text-align: center;
            color: #1f4287;
            padding: 20px 0;
            margin: 0 auto 30px auto;
            max-width: 800px;
            border-bottom: 3px solid #1f4287;
            font-size: 2.2em;
            font-weight: bold;
            background: linear-gradient(45deg, #1f4287, #4776b9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* العناوين الفرعية */
        h3 {
            color: #2d5a77;
            margin: 15px 0;
            padding: 10px 15px;
            border-right: 4px solid #1f4287;
            background: linear-gradient(to left, #f8f9fa, transparent);
            border-radius: 0 5px 5px 0;
        }
        
        /* الحاويات */
        .custom-container {
            background-color: white;
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .custom-container:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* جدول النتائج */
        .results-table {
            font-size: 14px;
            margin: 15px 0;
        }
        
        .dataframe {
            direction: rtl;
            text-align: right !important;
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            text-align: right !important;
            background-color: #1f4287 !important;
            color: white !important;
            padding: 12px !important;
            font-weight: 600;
        }
        
        .dataframe td {
            text-align: right !important;
            padding: 12px !important;
            border-bottom: 1px solid #e6e6e6;
            background-color: white;
        }
        
        .dataframe tr:hover td {
            background-color: #f8f9fa;
        }
        
        /* زر التحميل */
        .stUploadButton>button {
            width: 100%;
            max-width: 300px;
            margin: 10px auto;
            padding: 10px 20px;
            background-color: #1f4287;
            color: white;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .stUploadButton>button:hover {
            background-color: #2d5a77;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* الرسائل */
        .stAlert {
            direction: rtl;
            text-align: right;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        /* التعليقات التوضيحية */
        .caption {
            text-align: center;
            color: #666;
            margin: 8px 0;
            font-size: 0.9em;
            font-style: italic;
        }
        
        /* تنسيق الصور */
        .stImage {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* شريط التقدم */
        .stProgress {
            direction: rtl;
        }
        
        .stProgress > div > div {
            background: linear-gradient(45deg, #1f4287, #4776b9);
            border-radius: 10px;
        }
        
        /* تنسيق الأقسام */
        .section {
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* معلومات إضافية */
        .info-text {
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
            padding: 0 10px;
        }
        
        /* نسبة الثقة */
        .confidence {
            color: #1f4287;
            font-weight: bold;
        }
        
        /* تنسيق الحروف */
        .letter {
            font-size: 1.2em;
            font-weight: bold;
            color: #2d5a77;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # العنوان الرئيسي مع أيقونة
    st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1 class='main-title'>🔍 مودل التعرف على النصوص الثمودية</h1>
            <p class='info-text'>نظام ذكي للتعرف على الحروف الثمودية وتحويلها إلى النص العربي</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model and mapping
    model, class_mapping, letters_mapping = load_model()
    
    if model is None or class_mapping is None or letters_mapping is None:
        return
        
    # إنشاء صفين
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3>تحميل صورة للتعرف عليها</h3>", unsafe_allow_html=True)
        st.markdown("<p class='info-text'>يمكنك تحميل صورة بصيغة PNG أو JPG أو JPEG</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("اختر صورة...", type=['png', 'jpg', 'jpeg'])
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # اكتشاف مواقع الحروف
            with st.spinner('جاري تحليل الصورة...'):
                original_image = Image.open(uploaded_file).convert('RGB')
                boxes = detect_letters(original_image)
            
            if not boxes:
                st.error("لم يتم العثور على حروف في الصورة")
                return
                
            # تحويل الصناديق إلى الشكل المطلوب للتنبؤ
            converted_boxes = []
            letter_images = []
            
            for box in boxes:
                x, y, w, h = box
                letter_image = original_image.crop((x, y, x+w, y+h))
                letter_images.append(letter_image)
                converted_boxes.append((x, y, x+w, y+h))
            
            # التنبؤ بالحروف
            predictions = []
            confidences = []
            
            # إضافة شريط تقدم
            progress_text = st.markdown("<p class='info-text'>جاري التعرف على الحروف...</p>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            
            for i, letter_image in enumerate(letter_images):
                # تحديث شريط التقدم
                progress = (i + 1) / len(letter_images)
                progress_bar.progress(progress)
                
                # تجهيز الصورة
                processed_image = preprocess_image(letter_image)
                
                # التنبؤ
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                    predictions.append(prediction.item())
                    confidences.append(confidence.item())
            
            # إخفاء شريط التقدم والنص
            progress_bar.empty()
            progress_text.empty()
            
            # رسم المربعات والتنبؤات
            result_image = original_image.copy()
            result_image = draw_boxes(result_image, converted_boxes, predictions, letters_mapping, class_mapping)
            
            # عرض النتائج
            with col2:
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.markdown("<h3>نتائج التعرف</h3>", unsafe_allow_html=True)
                
                # عرض الصورة النهائية
                st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
                st.image(result_image, caption="نتائج التعرف", use_container_width=True)
                st.markdown("<p class='caption'>الحروف المكتشفة مع تصنيفها</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # عرض جدول النتائج
                st.markdown("<div class='custom-container results-table'>", unsafe_allow_html=True)
                results_data = []
                for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                    letter_name = next((k for k, v in class_mapping.items() if v == pred), "غير معروف")
                    arabic_letter = letters_mapping.get(letter_name, "؟")
                    results_data.append({
                        "الترتيب": f"#{i + 1}",
                        "الحرف الثمودي": f"<span class='letter'>{letter_name}</span>",
                        "الحرف العربي": f"<span class='letter'>{arabic_letter}</span>",
                        "نسبة الثقة": f"<span class='confidence'>{conf * 100:.1f}%</span>"
                    })
                
                # تحويل البيانات إلى DataFrame وعرضه
                import pandas as pd
                df = pd.DataFrame(results_data)
                st.markdown("<h4 style='color: #2d5a77; margin: 15px 0;'>تفاصيل النتائج</h4>", unsafe_allow_html=True)
                st.write(df.to_html(escape=False, index=False, classes='dataframe'), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
            logging.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
