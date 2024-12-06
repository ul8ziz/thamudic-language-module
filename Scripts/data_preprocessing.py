import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A
from typing import Tuple, List, Any

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    معالجة الصورة للتدريب مع تحسينات إضافية
    
    Args:
        image_path (str): مسار الصورة
        target_size (Tuple[int, int]): حجم الصورة المستهدف
    
    Returns:
        np.ndarray: الصورة المعالجة
    """
    try:
        # قراءة الصورة باللون الرمادي
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"خطأ في قراءة الصورة: {image_path}")
            return None
        
        # تغيير الحجم مع الحفاظ على نسبة العرض للارتفاع
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            new_w = target_size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect_ratio)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # إضافة حواف سوداء للحفاظ على الحجم
        canvas = np.zeros(target_size, dtype=np.uint8)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        
        # تحسين التباين باستخدام CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(canvas)
        
        # تنعيم الصورة
        blurred = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        # استخراج الحواف باستخدام Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # دمج الصورة الأصلية مع الحواف
        combined = cv2.addWeighted(canvas, 0.7, edges, 0.3, 0)
        
        # نرمجة القيم
        normalized_image = combined / 255.0
        
        return normalized_image
    except Exception as e:
        print(f"خطأ في معالجة الصورة {image_path}: {e}")
        return None

def augment_image(image: np.ndarray, num_augmentations: int = 3) -> List[np.ndarray]:
    """
    توليد نسخ معززة من الصورة الإدخال
    
    Args:
        image (np.ndarray): الصورة المدخلة
        num_augmentations (int): عدد النسخ المعززة المطلوبة
    
    Returns:
        List[np.ndarray]: قائمة الصور المعززة
    """
    augmentations = [
        A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
        ])
        for _ in range(num_augmentations)
    ]
    
    augmented_images = []
    for aug in augmentations:
        # التأكد من أن الصورة من نوع uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        augmented = aug(image=image_uint8)['image']
        augmented_normalized = augmented / 255.0
        augmented_images.append(augmented_normalized)
    
    return augmented_images

def load_data(
    data_dir: str = None, 
    augment: bool = True
) -> Tuple[List[np.ndarray], List[int], LabelEncoder]:
    """
    تحميل الصور والتصنيفات مع زيادة البيانات
    
    Args:
        data_dir (str): مسار مجلد الصور
        augment (bool): هل تريد زيادة البيانات
    
    Returns:
        Tuple[List[np.ndarray], List[int], LabelEncoder]: الصور، التصنيفات، مشفر التصنيفات
    """
    # إذا لم يتم تحديد المسار، استخدم المسار الافتراضي
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'thamudic_letters')
    
    # استخدام المسار المطلق لملف التعيين
    mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'letters', 'letter_mapping.json')
    
    # قراءة ملف التعيين
    with open(mapping_path, "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # تهيئة القوائم
    images = []
    labels = []
    label_encoder = LabelEncoder()
    
    # التكرار على الحروف الثمودية
    for letter_data in letter_mapping['thamudic_letters']:
        letter_symbol = letter_data['symbol']
        letter_index = letter_data['index']
        letter_dir = os.path.join(data_dir, f"letter_{letter_index+1}")
        
        # جمع الصور للحرف الحالي
        letter_images = [
            os.path.join(letter_dir, f) 
            for f in os.listdir(letter_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        for img_path in letter_images:
            # معالجة الصورة
            processed_image = preprocess_image(img_path)
            
            if processed_image is not None:
                images.append(processed_image)
                labels.append(letter_index)
                
                # زيادة البيانات
                if augment:
                    augmented_images = augment_image(processed_image)
                    images.extend(augmented_images)
                    labels.extend([letter_index] * len(augmented_images))
    
    # تشفير التصنيفات
    labels_encoded = label_encoder.fit_transform(labels)
    
    return images, labels_encoded, label_encoder

def split_data(
    images: List[np.ndarray], 
    labels: List[int], 
    test_size: float = 0.2, 
    val_size: float = 0.2
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    تقسيم البيانات إلى مجموعات تدريب واختبار والتحقق
    
    Args:
        images (List[np.ndarray]): الصور
        labels (List[int]): التصنيفات
        test_size (float): نسبة بيانات الاختبار
        val_size (float): نسبة بيانات التحقق
    
    Returns:
        Tuple[Any, Any, Any, Any, Any, Any]: 
        x_train, x_test, x_val, y_train, y_test, y_val
    """
    # تقسيم البيانات إلى مجموعة تدريب واختبار
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    # تقسيم مجموعة التدريب إلى تدريب والتحقق
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_temp, y_train_temp, test_size=val_size, stratify=y_train_temp, random_state=42
    )
    
    return x_train, x_test, x_val, y_train, y_test, y_val

def main():
    # تحميل البيانات
    images, labels, label_encoder = load_data()
    
    # تقسيم البيانات
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    print("معلومات البيانات:")
    print(f"إجمالي عدد الصور: {len(images)}")
    print(f"عدد الصور التدريبية: {len(x_train)}")
    print(f"عدد صور الاختبار: {len(x_test)}")
    print(f"عدد صور التحقق: {len(x_val)}")

if __name__ == "__main__":
    main()
