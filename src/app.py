import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import sys
import os
import logging
from torchvision import transforms

class ThamudicApp:
    def __init__(self):
        """
        Initialize the Thamudic Translation Application
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            # مسار النموذج والخريطة
            self.model_path = os.path.join(self.project_root, 'models/best_model.pth')
            self.mapping_file = os.path.join(self.project_root, 'data/char_mapping.json')
            
            # تحميل خريطة الأحرف
            from utils import load_char_mapping
            self.char_mapping = load_char_mapping(self.mapping_file)
            
            # إنشاء المترجم
            from model import ThamudicTranslator
            self.translator = ThamudicTranslator(
                model_path=self.model_path, 
                char_mapping=self.char_mapping
            )
        except Exception as e:
            st.error(f"خطأ في تهيئة النظام: {str(e)}")
            raise
    
    def preprocess_image(self, uploaded_file):
        """
        معالجة الصورة من الملف المرفوع
        
        Args:
            uploaded_file: ملف الصورة المرفوع من Streamlit
        
        Returns:
            torch.Tensor: الصورة معالجة كتانسور
        """
        # قراءة الصورة
        image = Image.open(uploaded_file).convert('L')
        img_array = np.array(image)
        
        # تحسين التباين باستخدام CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_array = clahe.apply(img_array)
        
        # إزالة الضوضاء باستخدام مرشح ثنائي
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # تطبيق عتبة تكيفية
        binary = cv2.adaptiveThreshold(
            img_array, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # تحسين الحروف باستخدام العمليات المورفولوجية
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # تحويل مرة أخرى إلى صورة PIL
        image = Image.fromarray(binary)
        
        # تطبيق نفس التحويلات المستخدمة في تدريب النموذج
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # تحويل الصورة إلى تانسور
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def show_system_info(self):
        """
        عرض معلومات النظام والملفات
        """
        with st.expander("معلومات النظام 🔍"):
            st.markdown("### معلومات الملفات والنظام")
            
            # عرض معلومات الملفات
            st.write("#### مسارات الملفات:")
            st.code(f"""
            - ملف النموذج: {os.path.abspath(self.model_path)}
            - ملف خريطة الأحرف: {os.path.abspath(self.mapping_file)}
            """)
            
            # عرض معلومات النظام
            st.write("#### إصدارات المكتبات:")
            st.code(f"""
            - Python: {sys.version.split()[0]}
            - Torch: {torch.__version__}
            - OpenCV: {cv2.__version__}
            - NumPy: {np.__version__}
            - Pillow: {Image.__version__}
            """)
            
            # عرض معلومات خريطة الأحرف
            st.write("#### معلومات خريطة الأحرف:")
            st.write(f"- عدد الأحرف المدعومة: {len(self.char_mapping)}")
            
            if st.checkbox("عرض خريطة الأحرف الكاملة"):
                st.json(self.char_mapping)

    def show_project_files(self):
        """
        عرض محتويات ملفات المشروع
        """
        st.markdown("### 📂 محتويات المشروع")
        
        # تعريف الملفات المهمة التي نريد عرضها
        important_extensions = {'.py', '.json', '.txt', '.md', '.yml', '.yaml'}
        
        def is_important_file(filename):
            return any(filename.endswith(ext) for ext in important_extensions)
        
        def read_file_content(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"خطأ في قراءة الملف: {str(e)}"
        
        # مسح الدليل وعرض الملفات
        for root, dirs, files in os.walk(self.project_root):
            # تجاهل المجلدات الخاصة
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__', 'venv', 'env'))]
            
            rel_path = os.path.relpath(root, self.project_root)
            if rel_path == '.':
                st.markdown("#### 📁 المجلد الرئيسي")
            else:
                st.markdown(f"#### 📁 {rel_path}")
            
            # عرض الملفات في هذا المجلد
            for file in files:
                if is_important_file(file):
                    filepath = os.path.join(root, file)
                    st.markdown(f"**📄 {file}**")
                    content = read_file_content(filepath)
                    st.code(content, language='python')

    def run(self):
        """
        تشغيل تطبيق الترجمة الثمودية
        """
        # عنوان التطبيق
        st.title('🔤 مترجم الأحرف الثمودية')
        

        # وصف التطبيق
        st.markdown('''
        ### كيفية الاستخدام
        - قم برفع صورة تحتوي على حروف ثمودية
        - سيقوم النموذج بالتعرف على الحروف وعرض معلوماتها
        ''')
        
        # رفع الصورة
        uploaded_file = st.file_uploader(
            "اختر صورة الحروف الثمودية", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            try:
                # عرض الصورة الأصلية
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption='الصورة الأصلية', use_container_width=True)
                
                # معالجة الصورة
                processed_image = self.preprocess_image(uploaded_file)
                logging.info("تمت معالجة الصورة بنجاح")
                
                # عرض الصورة المعالجة
                with col2:
                    # تحويل المعالجة إلى صورة قابلة للعرض
                    processed_display = processed_image.squeeze()
                    logging.info(f"شكل الصورة المعالجة: {processed_display.shape}")
                    
                    if processed_display.dim() == 2:  # إذا كانت الصورة تحتوي على بعدين فقط
                        processed_display = processed_display.unsqueeze(0)  # إضافة بُعد القناة
                        logging.info("تمت إضافة بُعد القناة للصورة")
                    
                    processed_display = processed_display.permute(1, 2, 0)  # إعادة ترتيب الأبعاد
                    logging.info(f"شكل الصورة بعد إعادة الترتيب: {processed_display.shape}")
                    
                    # إلغاء التطبيع
                    processed_display = processed_display * torch.tensor([0.5]) + torch.tensor([0.5])
                    logging.info(f"مدى قيم الصورة: [{processed_display.min():.2f}, {processed_display.max():.2f}]")
                    
                    # تحويل إلى NumPy
                    processed_display = processed_display.numpy()
                    # تأكد من أن القيم بين 0 و 1
                    processed_display = np.clip(processed_display, 0, 1)
                    st.image(processed_display, caption='الصورة المعالجة', use_container_width=True)
                
                # التعرف على الحروف
                logging.info("بدء عملية التعرف على الحروف...")
                predictions = self.translator.predict(processed_image)
                logging.info(f"عدد التنبؤات: {len(predictions)}")
                
                # عرض نتيجة التعرف
                st.markdown('## نتيجة التعرف')
                
                # جدول للحروف المعترف بها
                results_data = []
                for char, confidence in predictions:
                    logging.info(f"الحرف: {char}, نسبة الثقة: {confidence:.4f}")
                    results_data.append({
                        'الحرف': char,
                        'نسبة الثقة': f'{confidence * 100:.2f}%'
                    })
                
                # عرض الجدول
                if results_data:
                    st.table(results_data)
                    
                    # رسم مربع أخضر للحروف المعترف بها
                    st.markdown('''
                    <style>
                    .green-box {
                        border: 3px solid green;
                        border-radius: 10px;
                        padding: 10px;
                        text-align: center;
                        background-color: rgba(0, 255, 0, 0.1);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    </style>
                    ''', unsafe_allow_html=True)
                    
                    # عرض الحروف في مربع أخضر
                    chars_text = ' '.join([char for char, _ in predictions if confidence > 0.3])
                    st.markdown(f'''
                    <div class="green-box">
                        <h2 style="color: green;">{chars_text}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.warning('⚠️ لم يتم التعرف على أي حروف بشكل مؤكد')
                
            except Exception as e:
                st.error(f'حدث خطأ أثناء معالجة الصورة: {e}')

def main():
    """
    دالة رئيسية لتشغيل التطبيق
    """
    logging.basicConfig(level=logging.INFO)
    app = ThamudicApp()
    app.run()

if __name__ == "__main__":
    main()