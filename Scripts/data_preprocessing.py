import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A

def preprocess_image(image_path):
    """
    معالجة الصورة للتدريب
    - تحويل إلى تدرجات الرمادي
    - تغيير الحجم
    - تحسين التباين
    - تنعيم الصورة
    - استخراج الحواف
    """
    try:
        # قراءة الصورة
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            return None
        
        # تغيير الحجم
        resized_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        
        # تحسين التباين باستخدام CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_image = clahe.apply(resized_image)
        
        # تنعيم الصورة
        blurred = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        # استخراج الحواف باستخدام Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        edge_enhanced = cv2.convertScaleAbs(laplacian)
        
        # نرمجة القيم
        normalized_image = edge_enhanced / 255.0
        
        return normalized_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def augment_image(image):
    """
    Generate augmented versions of an input image
    
    Args:
        image (numpy.ndarray): Input image to augment
    
    Returns:
        list: List of augmented images
    """
    augmentations = [
        A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ])
        for _ in range(3)  # Generate 3 augmented versions of each image
    ]
    
    augmented_images = []
    for aug in augmentations:
        augmented = aug(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

def load_data(data_dir="../data/letters/thamudic_letters"):
    """
    Load images and labels from the specified directory with data augmentation
    
    Args:
        data_dir (str): Path to directory containing letter images
    
    Returns:
        tuple: (images, labels, label_encoder)
    """
    # قراءة ملف التعيين
    with open("../data/letters/letter_mapping.json", "r", encoding="utf-8") as f:
        letter_mapping = json.load(f)
    
    # تهيئة القوائم
    images = []
    labels = []
    label_encoder = LabelEncoder()
    
    # التكرار على الحروف الثمودية
    for letter_data in letter_mapping['thamudic_letters']:
        letter_symbol = letter_data['symbol']
        letter_name = letter_data['name']
        
        # مسار مجلد الحرف
        letter_dir = os.path.join(data_dir, letter_name)
        
        # التأكد من وجود المجلد
        if not os.path.exists(letter_dir):
            print(f"Warning: Directory not found for letter {letter_name}")
            continue
        
        # جمع جميع الصور للحرف
        letter_images = []
        for img_file in os.listdir(letter_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(letter_dir, img_file)
                processed_img = preprocess_image(img_path)
                
                if processed_img is not None:
                    # إضافة الصورة الأصلية
                    letter_images.append(processed_img)
                    
                    # توليد صور معززة
                    augmented_images = augment_image(processed_img)
                    letter_images.extend(augmented_images)
        
        # إضافة الصور إلى القوائم الرئيسية
        images.extend(letter_images)
        labels.extend([letter_symbol] * len(letter_images))
    
    # تحويل التسميات إلى أرقام
    labels = label_encoder.fit_transform(labels)
    
    # تحويل القوائم إلى مصفوفات numpy
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, label_encoder

def split_data(images, labels, test_size=0.2, val_size=0.2):
    # تقسيم البيانات إلى مجموعات التدريب والاختبار والتحقق
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=val_size, random_state=42
    )
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

if __name__ == "__main__":
    # تحميل البيانات
    images, labels, label_encoder = load_data("../data/letters/thamudic_letters")
    
    # تقسيم البيانات
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}")
    print(f"Test images shape: {test_images.shape}")
