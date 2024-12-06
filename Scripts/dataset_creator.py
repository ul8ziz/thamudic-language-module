import os
import sys
import cv2
import csv
import json
import numpy as np
import logging
import shutil
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import traceback

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('dataset_creation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """معالجة الصورة بشكل متقدم"""
    try:
        # التحويل إلى أبيض وأسود
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # تطبيق التمويه
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # التحويل الثنائي التكيفي
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # تغيير الحجم
        resized = cv2.resize(thresh, target_size, interpolation=cv2.INTER_AREA)
        
        return resized
    
    except Exception as e:
        logging.error(f"خطأ في معالجة الصورة: {e}")
        return None

def extract_letters_from_directory(
    directory_path: str, 
    output_dir: str, 
    letter_mapping: Dict
) -> List[Dict]:
    """استخراج الحروف الثمودية من مجلد صور"""
    try:
        # اختيار مجموعة الحروف الثمودية
        letters_data = letter_mapping['thamudic_letters']
        
        extracted_letters = []
        
        # التأكد من وجود المجلد
        os.makedirs(output_dir, exist_ok=True)
        
        # قائمة الملفات في المجلد
        image_files = sorted([
            f for f in os.listdir(directory_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        
        # تحديد عدد الحروف المطلوبة
        num_letters = min(len(image_files), len(letters_data))
        
        for i in range(num_letters):
            image_path = os.path.join(directory_path, image_files[i])
            
            # قراءة الصورة
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logging.warning(f"فشل في قراءة الصورة: {image_path}")
                continue
            
            # معالجة الصورة
            processed_letter = preprocess_image(image)
            
            if processed_letter is not None:
                # اختيار معلومات الحرف
                letter_info = letters_data[i]
                
                # حفظ الصورة باسم الحرف
                filename = f"{letter_info['name']}.png"
                letter_path = os.path.abspath(os.path.join(output_dir, filename))
                
                cv2.imwrite(letter_path, processed_letter)
                
                # تخزين معلومات الحرف
                extracted_letters.append({
                    'path': letter_path,
                    'name': letter_info['name'],
                    'type': 'thamudic',
                    'symbol': letter_info['symbol'],
                    'original_image': image_files[i]
                })
        
        logging.info(f"تم استخراج {len(extracted_letters)} حرف ثمودي من المجلد: {directory_path}")
        return extracted_letters
    
    except Exception as e:
        logging.error(f"خطأ في استخراج الحروف: {e}")
        return []

def create_letter_dataset(
    letters_mapping_path: str, 
    output_base_dir: str,
    thamudic_letters_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2
):
    """
    إنشاء مجموعة بيانات الحروف الثمودية
    
    :param letters_mapping_path: مسار ملف التعيين
    :param output_base_dir: المجلد الأساسي للمخرجات
    :param thamudic_letters_dir: مجلد صور الحروف الثمودية
    :param test_size: نسبة البيانات للاختبار
    :param val_size: نسبة البيانات للتحقق
    """
    try:
        # إنشاء المجلدات
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_base_dir, split, 'thamudic'), exist_ok=True)
        
        # قراءة ملف التعيين
        with open(letters_mapping_path, 'r', encoding='utf-8') as f:
            letter_mapping = json.load(f)
        
        # استخراج الحروف الثمودية
        thamudic_letters = extract_letters_from_directory(
            directory_path=thamudic_letters_dir, 
            output_dir=os.path.join(output_base_dir, 'train', 'thamudic'), 
            letter_mapping=letter_mapping
        )
        
        # تقسيم البيانات
        train_letters, test_letters = train_test_split(thamudic_letters, test_size=test_size, random_state=42)
        train_letters, val_letters = train_test_split(train_letters, test_size=val_size, random_state=42)
        
        # نسخ الصور
        for split, letters in [
            ('train', train_letters), 
            ('val', val_letters), 
            ('test', test_letters)
        ]:
            for letter in letters:
                dest_dir = os.path.join(output_base_dir, split, 'thamudic')
                dest_filename = os.path.basename(letter['path'])
                dest_path = os.path.join(dest_dir, dest_filename)
                
                # التأكد من وجود المجلد
                os.makedirs(dest_dir, exist_ok=True)
                
                # نسخ الصورة
                try:
                    # التأكد من أن الصورة المصدر موجودة
                    if not os.path.exists(letter['path']):
                        logging.warning(f"الصورة غير موجودة: {letter['path']}")
                        continue
                    
                    # نسخ الصورة
                    shutil.copy2(letter['path'], dest_path)
                    letter['path'] = dest_path
                except Exception as e:
                    logging.error(f"خطأ في نسخ الصورة {letter['path']}: {e}")
        
        # إنشاء ملف البيانات الوصفية
        metadata_path = os.path.join(output_base_dir, 'dataset_metadata.csv')
        with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['path', 'name', 'type', 'symbol', 'original_image'])
            writer.writeheader()
            writer.writerows(thamudic_letters)
        
        logging.info(f"تم إنشاء مجموعة البيانات في {output_base_dir}")
        
    except Exception as e:
        logging.error(f"خطأ في إنشاء مجموعة البيانات: {e}")
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    # مسارات محددة بشكل دقيق
    letters_mapping_path = r'd:/Work/rizg/Thamudic_language_recognition/projact/thamudic_env/data/letters/letter_mapping.json'
    output_base_dir = r'd:/Work/rizg/Thamudic_language_recognition/projact/thamudic_env/data/train_dataset'
    
    # مسار الصور الثمودية
    thamudic_letters_dir = r'd:/Work/rizg/Thamudic_language_recognition/projact/thamudic_env/data/letters/thamudic_letters'
    
    create_letter_dataset(
        letters_mapping_path=letters_mapping_path,
        output_base_dir=output_base_dir,
        thamudic_letters_dir=thamudic_letters_dir,
        test_size=0.2,
        val_size=0.2
    )
