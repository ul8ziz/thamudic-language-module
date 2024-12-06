import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # تحسين الصورة
            image = cv2.resize(image, (128, 128))  # تغيير الحجم
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = cv2.equalizeHist(image)
            # حفظ الصورة
            cv2.imwrite(os.path.join(output_dir, filename), image)

def load_data(data_dir):
    images = []
    labels = []
    
    for label, category in enumerate(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                image_path = os.path.join(category_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, (128, 128))
                    images.append(image / 255.0)  # تطبيع البيانات
                    labels.append(label)
    
    return np.array(images).reshape(-1, 128, 128, 1), np.array(labels)

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
    # معالجة الصور الخام
    preprocess_images("data/raw", "data/processed")
    
    # تحميل البيانات
    images, labels = load_data("data/processed")
    
    # تقسيم البيانات
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}")
    print(f"Test images shape: {test_images.shape}")
