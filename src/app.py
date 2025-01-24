import streamlit as st
import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from thamudic_model import ThamudicRecognitionModel
import logging
import cv2

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
    
    # تحويلات الصورة
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # تطبيق التحويلات
    transformed = transform(image=image_np)
    image_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
    
    # إضافة بُعد الدفعة
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def detect_letters(image):
    """
    اكتشاف مواقع الحروف في الصورة
    """
    # تحويل الصورة إلى تدرج رمادي
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # تطبيق عتبة ثنائية تكيفية
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
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
        if area > 100 and area < (image.size[0] * image.size[1]) / 4:
            boxes.append((x, y, w, h))
    
    return boxes

def draw_boxes(image, boxes, predictions, letters_mapping):
    """
    رسم مربعات حول الحروف المكتشفة مع كتابة التنبؤات
    """
    draw = ImageDraw.Draw(image)
    
    for i, ((x, y, w, h), (thamudic_letter, confidence)) in enumerate(zip(boxes, predictions)):
        # رسم المربع
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='lime',
            width=2
        )
        
        # الحصول على الحرف العربي المقابل
        arabic_letter = letters_mapping.get(thamudic_letter, "غير معروف")
        
        # كتابة التنبؤ
        text = f"letter_{i+1}: {arabic_letter} - {thamudic_letter}"
        draw.text((x, y-20), text, fill='lime')
    
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
            width: 100%;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # العنوان
    st.title("نظام التعرف على النقوش الثمودية")
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
            for x, y, w, h in boxes:
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
                    
                # البحث عن اسم الحرف
                predicted_letter = None
                for letter, idx in class_mapping.items():
                    if idx == predicted_idx:
                        predicted_letter = letter
                        break
                
                # إذا لم يتم العثور على الحرف، استخدم علامة استفهام
                if predicted_letter is None:
                    predicted_letter = "?"
                
                predictions.append((predicted_letter, confidence))
            
            # عرض الصور في صفين
            st.subheader("الصور")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**الصورة الأصلية**")
                st.image(original_image, use_container_width=True)
            
            with col2:
                st.markdown("**نتيجة التعرف**")
                # رسم المربعات والتنبؤات
                result_image = original_image.copy()
                result_image = draw_boxes(result_image, boxes, predictions, letters_mapping)
                st.image(result_image, use_container_width=True)
            
            # إضافة خط فاصل
            st.markdown("---")
            
            # عرض النتائج في جدول
            st.subheader("نتائج التعرف")
            
            # إنشاء بيانات الجدول
            table_data = []
            for i, ((x, y, w, h), (thamudic_letter, confidence)) in enumerate(zip(boxes, predictions)):
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
