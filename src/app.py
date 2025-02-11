"""
تطبيق التعرف على الحروف الثمودية
"""

import os
<<<<<<< Updated upstream
from PIL import Image, ImageEnhance
from core.inference_engine import InferenceEngine

# تهيئة الصفحة
st.set_page_config(
    page_title="محرك التعرف على الكتابات الثمودية",
    page_icon="🔍",
    layout="wide"
)

# CSS للتصميم من اليمين إلى اليسار
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

# تهيئة المسارات
MODEL_PATH = os.path.join('models', 'best_model.pth')
LABEL_MAPPING_PATH = os.path.join('models', 'configs', 'label_mapping.json')

@st.cache_resource
def load_predictor():
    return InferenceEngine(MODEL_PATH, LABEL_MAPPING_PATH)

def preprocess_image(image, contrast=2.0, sharpness=1.5, brightness=1.2, auto_rotate=True):
    """تجهيز الصورة للتحليل مع خيارات متقدمة"""
    # تحويل الصورة إلى RGB إذا كانت RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # تحويل الصورة إلى تدرجات الرمادي
    image = image.convert('L')
    
    # تصحيح دوران الصورة إذا كان مطلوباً
    if auto_rotate:
        try:
            from PIL import ExifTags
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(image._getexif().items())
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # الصورة لا تحتوي على معلومات EXIF
                pass
        except:
            pass
    
    # تحسين السطوع
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # تحسين التباين
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # تحسين الحدة
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    return image

def main():
    st.title("محرك التعرف على الكتابات الثمودية 🔍")
    st.markdown("---")

    try:
        predictor = load_predictor()
        st.success("تم تحميل النموذج بنجاح!")

        # إعدادات معالجة الصورة
        with st.expander("⚙️ إعدادات معالجة الصورة"):
            col1, col2, col3 = st.columns(3)
            with col1:
                contrast = st.slider("التباين", 0.5, 3.0, 2.0, 0.1)
            with col2:
                sharpness = st.slider("الحدة", 0.5, 3.0, 1.5, 0.1)
            with col3:
                brightness = st.slider("السطوع", 0.5, 2.0, 1.2, 0.1)
            auto_rotate = st.checkbox("تصحيح دوران الصورة تلقائياً", value=True)

        # تحميل الصورة
        uploaded_file = st.file_uploader(
            "اختر صورة للتحليل",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file:
            # عرض الصورة والتنبؤات
            col2, col1 = st.columns(2)
            
            with col1:
                st.subheader("نتائج التحليل")
                with st.spinner('جاري تحليل الصورة...'):
                    try:
                        # قراءة وعرض الصورة الأصلية
                        original_image = Image.open(uploaded_file)
                        
                        # معالجة الصورة
                        processed_image = preprocess_image(
                            original_image,
                            contrast=contrast,
                            sharpness=sharpness,
                            brightness=brightness,
                            auto_rotate=auto_rotate
                        )
                        
                        # حفظ الصورة المعالجة
                        temp_path = "temp_image.jpg"
                        processed_image.save(temp_path)
                        
                        # التنبؤ
                        results = predictor.predict(temp_path)
                        
                        if results:
                            st.write(f"تم العثور على {len(results)} حرف")
                            
                            # عرض النتائج الفردية
                            for i, result in enumerate(results):
                                letter = result['letter']
                                thamudic_symbol = predictor.thamudic_symbols.get(letter, '?')
                                confidence = result['confidence']
                                
                                # تلوين النتيجة حسب نسبة الثقة
                                if confidence >= 0.8:
                                    confidence_color = "🟢"  # أخضر للثقة العالية
                                elif confidence >= 0.5:
                                    confidence_color = "🟡"  # أصفر للثقة المتوسطة
                                else:
                                    confidence_color = "🔴"  # أحمر للثقة المنخفضة
                                
                                st.metric(
                                    f"التنبؤ #{i+1} {confidence_color}",
                                    f"{letter} ({thamudic_symbol})",
                                    f"نسبة الثقة: {confidence:.2%}"
                                )
                            
                            # عرض النص الكامل
                            full_text = "".join([result['letter'] for result in results])
                            full_text_thamudic = "".join([predictor.thamudic_symbols.get(result['letter'], '?') for result in results])
                            
                            st.markdown("### النص المتعرف عليه")
                            st.markdown(f"""
                            <div style='direction: rtl; text-align: right; font-size: 24px; 
                                        padding: 20px; background-color: #1e3d59; border-radius: 10px;
                                        color: #ffc13b; font-weight: bold; margin: 10px 0;
                                        border: 2px solid #ffc13b;'>
                                <div style='margin-bottom: 10px;'>النص العربي: {full_text}</div>
                                <div>النص الثمودي: {full_text_thamudic}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("لم يتم العثور على أي نتائج")
                            
                    except Exception as pred_error:
                        st.error(f"خطأ في التنبؤ: {str(pred_error)}")
                        st.write("معلومات إضافية:")
                        st.write(f"نوع الخطأ: {type(pred_error).__name__}")
                        import traceback
                        st.code(traceback.format_exc())

            with col2:
                # عرض الصور
                st.subheader("الصورة الأصلية")
                st.image(original_image, caption="الصورة الأصلية")
                
                st.subheader("الصورة بعد المعالجة")
                st.image(processed_image, caption="الصورة بعد المعالجة")

            # تنظيف الملف المؤقت
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
        st.write("معلومات إضافية:")
        st.write(f"نوع الخطأ: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
=======
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from PIL import Image
import streamlit as st
from thamudic_model import ThamudicRecognitionModel

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

class ThamudicRecognitionApp:
    """تطبيق التعرف على الحروف الثمودية"""
    
    def __init__(self, model_path: str, mapping_file: str, image_size: tuple = (224, 224)):
        """
        تهيئة التطبيق
        
        المعاملات:
            model_path: مسار ملف النموذج
            mapping_file: مسار ملف تعيين الفئات
            image_size: حجم الصورة المطلوب
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # تحميل تعيين الفئات
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            # تحويل البيانات إلى القاموس المطلوب
            self.class_mapping = {item['name']: item['index'] for item in mapping_data['thamudic_letters']}
            
        # إنشاء تعيين عكسي (من الفهرس إلى الفئة)
        self.idx_to_class = {idx: name for name, idx in self.class_mapping.items()}
        
        # تهيئة النموذج
        self.model = ThamudicRecognitionModel(num_classes=len(self.class_mapping))
        self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """تحميل النموذج من ملف"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        معالجة الصورة قبل التعرف
        
        المعاملات:
            image: مصفوفة numpy للصورة
            
        Returns:
            tensor: تنسور PyTorch جاهز للنموذج
        """
        # تحويل إلى تدرج رمادي
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # تطبيق عتبة تكيفية
        binary = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # تنظيف الضوضاء
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # تغيير الحجم
        resized = cv2.resize(cleaned, self.image_size)
        
        # تحويل إلى تنسور
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # إضافة بُعد الدفعة والقناة
        tensor = tensor / 255.0  # تطبيع
        
        return tensor
    
    def predict(self, image: np.ndarray, top_k: int = 3) -> list:
        """
        التعرف على الحرف في الصورة
        
        المعاملات:
            image: مصفوفة numpy للصورة
            top_k: عدد أفضل التنبؤات المطلوبة
            
        Returns:
            list: قائمة من أفضل التنبؤات مع درجات الثقة
        """
        try:
            # معالجة الصورة
            tensor = self.preprocess_image(image)
            tensor = tensor.to(self.device)
            
            # التنبؤ
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # الحصول على أفضل التنبؤات
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_mapping)))
            
            # تحويل النتائج إلى قائمة
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                idx_val = idx.item()
                class_name = self.idx_to_class[idx_val]
                confidence = prob.item() * 100
                predictions.append({
                    'class': class_name,
                    'confidence': confidence
                })
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

def main():
    """النقطة الرئيسية للتطبيق"""
    st.title("التعرف على الحروف الثمودية")
    
    # تحديد المسارات
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'thamudic_model.pth'
    mapping_file = base_dir / 'data' / 'mapping.json'
    
    if not model_path.exists():
        st.error("لم يتم العثور على ملف النموذج. الرجاء تدريب النموذج أولاً.")
        return
        
    try:
        # تهيئة التطبيق
        app = ThamudicRecognitionApp(
            model_path=str(model_path),
            mapping_file=str(mapping_file)
        )
        
        # واجهة تحميل الصور
        st.subheader("تحميل صورة")
        uploaded_file = st.file_uploader(
            "اختر صورة للحرف الثمودي",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # عرض الصورة المحملة
            image = Image.open(uploaded_file)
            st.image(image, caption='الصورة المحملة', use_column_width=True)
            
            # التعرف على الحرف
            predictions = app.predict(np.array(image))
            
            # عرض النتائج
            st.subheader("النتائج:")
            for pred in predictions:
                st.write(f"{pred['class']}: {pred['confidence']:.2f}%")
                
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
        logging.error(f"خطأ في التطبيق: {str(e)}")
>>>>>>> Stashed changes

if __name__ == '__main__':
    main()
