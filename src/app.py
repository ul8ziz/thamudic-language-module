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
    # إعداد الصفحة
    st.set_page_config(
        page_title="نظام التعرف على النقوش الثمودية",
        page_icon="🔍",
        layout="wide"
    )
    
    # إضافة CSS مخصص
    st.markdown("""
        <style>
        .stApp {
            direction: rtl;
            text-align: right;
        }
        .css-1d391kg {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            width: 50%;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # العنوان
    st.title("مودل التعرف على النقوش الثمودية")
    st.markdown("---")
    
    # تحميل النموذج
    model, class_mapping, letters_mapping = load_model()
    
    if model is None or class_mapping is None or letters_mapping is None:
        return
        
    # تحميل الصورة
    st.subheader("تحميل صورة")
    uploaded_file = st.file_uploader("اختر صورة للنقش الثمودي", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # عرض الصورة الأصلية
            original_image = Image.open(uploaded_file).convert('RGB')
            
            # اكتشاف مواقع الحروف
            boxes = detect_letters(original_image)
            
            predictions = []
            converted_boxes = []  # قائمة جديدة للصناديق المحولة
            for x, y, w, h in boxes:
                # تحويل تنسيق الصندوق
                converted_boxes.append((x, y, x+w, y+h))
                
                # اقتصاص الحرف
                letter_image = original_image.crop((x, y, x+w, y+h))
                
                # معالجة الصورة
                image_tensor = preprocess_image(letter_image)
                
                # التنبؤ
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item() * 100
                    
                predictions.append(predicted_idx)  # تخزين فقط مؤشر التنبؤ
            
            if predictions:
                st.markdown("**نتيجة التعرف**")
                # رسم المربعات والتنبؤات
                result_image = original_image.copy()
                result_image = draw_boxes(result_image, converted_boxes, predictions, letters_mapping, class_mapping)
                st.image(result_image, use_container_width=True)
            
            # إضافة خط فاصل
            st.markdown("---")
            
            # عرض النتائج في جدول
            st.subheader("نتائج التعرف")
            
            # إنشاء بيانات الجدول
            table_data = []
            for i, ((x, y, w, h), pred) in enumerate(zip(boxes, predictions)):
                thamudic_letter = None
                for letter, idx in class_mapping.items():
                    if idx == pred:
                        thamudic_letter = letter
                        break
                arabic_letter = letters_mapping.get(thamudic_letter, "غير معروف")
                table_data.append({
                    "رقم الحرف": f"letter_{i+1}",
                    "الحرف العربي": arabic_letter,
                    "الحرف الثمودي": thamudic_letter,
                    "نسبة الثقة": f"{confidence:.1f}%"
                })
            
            # عرض الجدول
            if table_data:
                st.table(table_data)
            else:
                st.warning("لم يتم التعرف على أي حروف بشكل مؤكد")
                
        except Exception as e:
            st.error(f"خطأ في معالجة الصورة: {str(e)}")
            
    # معلومات إضافية
    with st.expander("معلومات عن النظام"):
        st.write("""
        - هذا النظام يستخدم شبكة عصبية عميقة للتعرف على الحروف الثمودية.
        - النظام مدرب على مجموعة من صور الحروف الثمودية.
        - دقة النموذج تعتمد على جودة الصورة المدخلة ووضوحها.
        - يمكن للنظام التعرف على عدة حروف في نفس الصورة.
        """)

if __name__ == "__main__":
    main()
