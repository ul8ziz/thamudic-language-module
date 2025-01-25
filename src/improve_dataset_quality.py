import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from PIL import Image
import shutil
from tqdm import tqdm

def create_improved_transforms():
    """إنشاء التحويلات المحسنة للصور"""
    return A.Compose([
        # تحسين الجودة
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=1.0),
        
        # تنظيف الضوضاء
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        
        # تحسين التباين
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
        ], p=0.5),
        
        # تحويلات هندسية معتدلة
        A.OneOf([
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], p=0.5),
    ])

def clean_binary_image(image):
    """تنظيف وتحسين الصورة الثنائية"""
    # تحويل إلى تدرج رمادي إذا كانت ملونة
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def improve_letter_image(image_path, output_path, transforms):
    """تحسين صورة حرف واحد"""
    # قراءة الصورة
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"خطأ في قراءة الصورة: {image_path}")
        return False
        
    # تنظيف الصورة
    binary = clean_binary_image(image)
    
    # تحويل إلى RGB للتحويلات
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # تطبيق التحويلات
    transformed = transforms(image=rgb)['image']
    
    # حفظ الصورة
    cv2.imwrite(str(output_path), transformed)
    return True

def process_dataset(input_dir, output_dir, num_variations=5):
    """معالجة مجموعة البيانات بالكامل"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # إنشاء مجلد الإخراج
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # إنشاء التحويلات
    transforms = create_improved_transforms()
    
    # معالجة كل مجلد حرف
    for letter_dir in tqdm(list(input_dir.glob('letter_*'))):
        if not letter_dir.is_dir():
            continue
            
        # إنشاء مجلد الإخراج للحرف
        letter_output_dir = output_dir / letter_dir.name
        letter_output_dir.mkdir(parents=True, exist_ok=True)
        
        # معالجة كل صورة في المجلد
        for img_path in letter_dir.glob('*.png'):
            if 'aug_' in img_path.name:
                continue  # تجاهل الصور المحسنة سابقاً
                
            # حفظ النسخة الأصلية بعد تنظيفها
            base_name = img_path.stem
            clean_output_path = letter_output_dir / f"{base_name}.png"
            improve_letter_image(img_path, clean_output_path, transforms)
            
            # إنشاء نسخ محسنة إضافية
            for i in range(num_variations):
                aug_output_path = letter_output_dir / f"aug_{i}_{base_name}.png"
                improve_letter_image(img_path, aug_output_path, transforms)

def main():
    """النقطة الرئيسية للسكربت"""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'data' / 'letters' / 'processed_letters'
    output_dir = base_dir / 'data' / 'letters' / 'improved_letters'
    
    print("بدء تحسين جودة الصور...")
    process_dataset(input_dir, output_dir)
    print("اكتمل تحسين جودة الصور!")

if __name__ == "__main__":
    main()
