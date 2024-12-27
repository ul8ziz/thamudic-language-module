import streamlit as st
import os
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

if __name__ == '__main__':
    main()
