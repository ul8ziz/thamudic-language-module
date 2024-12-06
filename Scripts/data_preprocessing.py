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
    images = []
    labels = []
    original_labels = []
    processed_count = 0
    
    # Load images from directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(data_dir, filename)
            original_image = preprocess_image(image_path)
            
            if original_image is not None:
                # Add original image
                images.append(original_image)
                original_label = filename.split('_')[1].split('.')[0]
                labels.append(original_label)
                original_labels.append(original_label)
                processed_count += 1
                
                # Generate and add augmented images
                augmented_images = augment_image(original_image)
                images.extend(augmented_images)
                labels.extend([original_label] * len(augmented_images))
                original_labels.extend([original_label] * len(augmented_images))
    
    # Convert labels to numpy array and encode
    labels = np.array(labels, dtype=str)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Save label mapping information
    mapping_info = {
        "original_labels": original_labels,
        "mapped_labels": labels.tolist(),
        "encoded_labels": labels.tolist(),
        "label_mapping": {orig: mapped for orig, mapped in zip(original_labels, labels)},
        "label_encoder_classes": label_encoder.classes_.tolist()
    }
    
    # Save label mapping information
    os.makedirs("../data", exist_ok=True)
    with open("../data/label_mapping_info.json", "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=4)
    
    print(f"Processed {processed_count} images.")
    
    return np.array(images), labels, label_encoder

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
