import os
import json
import shutil
import cv2
import numpy as np
from typing import List, Dict

def validate_image(image_path: str) -> bool:
    """
    تحقق من صحة الصورة وجودتها
    
    Args:
        image_path (str): مسار الصورة
    
    Returns:
        bool: True إذا كانت الصورة صالحة، False خلاف ذلك
    """
    try:
        # قراءة الصورة
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"خطأ في قراءة الصورة: {image_path}")
            return False
        
        # التحقق من أبعاد الصورة
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            print(f"الصورة صغيرة جدًا: {image_path}")
            return False
        
        # التحقق من التباين
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < 10:
            print(f"الصورة باهتة جدًا: {image_path}")
            return False
        
        return True
    except Exception as e:
        print(f"خطأ في التحقق من الصورة {image_path}: {e}")
        return False

def clean_and_organize_images(target_base_dir: str) -> Dict:
    """
    تنظيف وترتيب الصور للحروف الثمودية
    
    Args:
        target_base_dir (str): المسار الأساسي لمجلد الحروف
    
    Returns:
        dict: إحصائيات عن الصور المنظمة
    """
    stats = {
        "total_images": 0,
        "valid_images": 0,
        "removed_images": 0
    }
    
    # استخدام المسار المطلق لملف التعيين
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'letter_mapping.json')
    
    # قراءة ملف التعيين
    with open(mapping_path, "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # التكرار على الحروف الثمودية
    for letter_data in letter_mapping['thamudic_letters']:
        index = letter_data['index']
        letter_dir = os.path.join(target_base_dir, f"letter_{index+1}")
        
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
        
        # جمع الصور
        image_files = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        stats["total_images"] += len(image_files)
        
        # تنظيف وإعادة تسمية الصور
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(letter_dir, img_file)
            
            if validate_image(img_path):
                # إعادة التسمية بتنسيق موحد
                new_name = f"letter_{index}_{i}.png"
                new_path = os.path.join(letter_dir, new_name)
                
                # تحويل الصورة إلى تدرجات الرمادي وحفظها
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(new_path, image)
                
                # حذف الصورة القديمة
                if new_path != img_path:
                    os.remove(img_path)
                
                stats["valid_images"] += 1
            else:
                # إزالة الصور غير الصالحة
                os.remove(img_path)
                stats["removed_images"] += 1
    
    return stats

def update_letter_mapping(target_base_dir: str) -> None:
    """
    تحديث ملف التعيين بمسارات الصور المحدثة
    
    Args:
        target_base_dir (str): المسار الأساسي لمجلد الحروف
    """
    # استخدام المسار المطلق لملف التعيين
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'letter_mapping.json')
    
    # قراءة ملف التعيين
    with open(mapping_path, "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # تحديث مسارات الصور للحروف الثمودية
    for letter_data in letter_mapping['thamudic_letters']:
        index = letter_data['index']
        letter_dir = os.path.join(target_base_dir, f"letter_{index+1}")
        
        # جمع مسارات الصور
        image_paths = []
        if os.path.exists(letter_dir):
            for img_file in os.listdir(letter_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_paths.append(os.path.join(letter_dir, img_file))
        
        # تحديث مسارات الصور في التعيين
        letter_data['images'] = image_paths
    
    # حفظ التعيين المحدث
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(letter_mapping, f, ensure_ascii=False, indent=4)
    
    print("تم تحديث مسارات الصور في ملف التعيين.")

def main():
    # مسار مجلد الحروف الثمودية
    target_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'thamudic_letters')
    
    # تنظيف وترتيب الصور
    stats = clean_and_organize_images(target_base_dir)
    
    print("إحصائيات معالجة الصور:")
    print(f"إجمالي الصور: {stats['total_images']}")
    print(f"الصور الصالحة: {stats['valid_images']}")
    print(f"الصور المحذوفة: {stats['removed_images']}")
    
    # تحديث ملف التعيين
    update_letter_mapping(target_base_dir)

if __name__ == "__main__":
    main()
