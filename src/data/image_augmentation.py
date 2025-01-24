import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
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
import os
import json
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import albumentations as A
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import tensorflow_addons as tfa
from collections import Counter

class ThamudicImagePreprocessor:
    def __init__(self):
        """تهيئة معالج الصور للنصوص الثمودية"""
        self.target_size = (64, 64)
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def enhance_contrast(self, image):
        """تحسين التباين في الصورة"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)
    
    def denoise(self, image):
        """إزالة الضوضاء من الصورة"""
        # تحويل الصورة إلى مصفوفة numpy
        img_array = np.array(image)
        
        # تطبيق إزالة الضوضاء
        denoised = cv2.fastNlMeansDenoisingColored(
            img_array,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return Image.fromarray(denoised)
    
    def sharpen(self, image):
        """تحسين حدة الصورة"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)
    
    def preprocess_for_model(self, image):
        """تجهيز الصورة للنموذج"""
        # تطبيق التحويلات
        tensor = self.transform(image)
        
        # تحويل التنسور إلى numpy array
        return tensor.numpy()
    
    def binarize(self, image):
        """تحويل الصورة إلى ثنائية (أسود وأبيض)"""
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # تطبيق عتبة Otsu
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return Image.fromarray(binary)
    
    def remove_background(self, image):
        """إزالة خلفية الصورة"""
        # تحويل الصورة إلى RGBA
        rgba = image.convert('RGBA')
        data = np.array(rgba)
        
        # إنشاء قناع للخلفية
        r, g, b, a = data.T
        light_areas = (r > 200) & (g > 200) & (b > 200)
        data[..., :][light_areas.T] = (255, 255, 255, 0)
        
        return Image.fromarray(data)
    
    def deskew(self, image):
        """تصحيح ميل النص"""
        # تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # حساب زاوية الميل
        coords = np.column_stack(np.where(gray < 127))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
            
        # تدوير الصورة
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(image),
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return Image.fromarray(rotated)

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
        image_files = [
            f for f in os.listdir(directory_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        # تحديد عدد الحروف المطلوبة
        num_letters = len(letters_data)
        
        for i in range(num_letters):
            letter_info = letters_data[i]
            letter_name = letter_info['name']
            
            # محاولة العثور على الملف باستخدام الأنماط المختلفة للأسماء
            possible_filenames = [
                f"{letter_name}.png",  # Arabic name
                f"letter_{i}.png",     # English with index
                f"thamudic_{letter_name}.png",  # With thamudic prefix
                f"thamudic_{letter_name}_{i}.png"  # With thamudic prefix and index
            ]
            
            found_file = None
            for filename in possible_filenames:
                if filename in image_files:
                    found_file = filename
                    break
            
            if found_file is None:
                logging.warning(f"الصورة غير موجودة: {os.path.join(directory_path, letter_name)}.png")
                logging.warning(f"تم البحث عن الأسماء التالية: {', '.join(possible_filenames)}")
                continue
            
            image_path = os.path.join(directory_path, found_file)
            
            # قراءة الصورة
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logging.warning(f"فشل في قراءة الصورة: {image_path}")
                continue
            
            # معالجة الصورة
            processed_letter = preprocess_image(image)
            
            if processed_letter is not None:
                # حفظ الصورة باسم الحرف
                output_filename = f"{letter_name}.png"
                letter_path = os.path.abspath(os.path.join(output_dir, output_filename))
                
                cv2.imwrite(letter_path, processed_letter)
                
                extracted_letters.append({
                    'original_path': image_path,
                    'processed_path': letter_path,
                    'letter_info': letter_info
                })
                
                logging.info(f"تمت معالجة وحفظ الحرف: {letter_name}")
            
        return extracted_letters
    
    except Exception as e:
        logging.error(f"خطأ في استخراج الحروف: {e}")
        traceback.print_exc()
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
                dest_filename = os.path.basename(letter['processed_path'])
                dest_path = os.path.join(dest_dir, dest_filename)
                
                # التأكد من وجود المجلد
                os.makedirs(dest_dir, exist_ok=True)
                
                # نسخ الصورة
                try:
                    # التأكد من أن الصورة المصدر موجودة
                    if not os.path.exists(letter['processed_path']):
                        logging.warning(f"الصورة غير موجودة: {letter['processed_path']}")
                        continue
                    
                    # نسخ الصورة
                    shutil.copy2(letter['processed_path'], dest_path)
                    letter['processed_path'] = dest_path
                except Exception as e:
                    logging.error(f"خطأ في نسخ الصورة {letter['processed_path']}: {e}")
        
        # إنشاء ملف البيانات الوصفية
        metadata_path = os.path.join(output_base_dir, 'dataset_metadata.csv')
        with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['original_path', 'processed_path', 'letter_info'])
            writer.writeheader()
            writer.writerows([{'original_path': letter['original_path'], 'processed_path': letter['processed_path'], 'letter_info': json.dumps(letter['letter_info'])} for letter in thamudic_letters])
        
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def augment_image(img):
    """
    توليد نسخة معدلة من الصورة مع تحسينات متقدمة
    """
    try:
        # Random flip
        if tf.random.uniform([], 0, 1) > 0.5:
            img = tf.image.flip_left_right(img)
        
        # Random rotation
        angle = tf.random.uniform([], -0.2, 0.2)
        img = tfa.image.rotate(img, angle)
        
        # Random brightness and contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Random saturation and hue
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.1)
        
        # Ensure values are in [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img
        
    except Exception as e:
        logging.error(f"Error during augmentation: {str(e)}")
        return None

def preprocess_image(image_path):
    """
    معالجة الصورة وتحويلها إلى التنسيق المناسب
    """
    try:
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Convert to float32
        img = tf.cast(img, tf.float32)
        
        # Normalize to [0,1]
        img = img / 255.0
        
        # Get image dimensions
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        
        # Calculate scaling factor
        target_size = tf.constant([128, 128], dtype=tf.int32)
        scale = tf.minimum(
            tf.cast(target_size[0], tf.float32) / tf.cast(h, tf.float32),
            tf.cast(target_size[1], tf.float32) / tf.cast(w, tf.float32)
        )
        
        # Calculate new dimensions
        new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
        
        # Resize image
        img = tf.image.resize(img, [new_h, new_w])
        
        # Calculate padding
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        
        # Pad image
        paddings = [[0, pad_h], [0, pad_w], [0, 0]]
        img = tf.pad(img, paddings, mode='CONSTANT', constant_values=1.0)
        
        # Ensure output shape
        img = tf.ensure_shape(img, [128, 128, 3])
        
        return img
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def load_data(data_dir, label_mapping_file):
    """
    تحميل وتحسين البيانات مع معالجة متقدمة للصور
    """
    try:
        # Load label mapping
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            letter_mapping = mapping_data['thamudic_letters']
        
        images = []
        labels = []
        label_counts = {}
        
        # Process each letter directory
        for letter_idx in range(len(letter_mapping)):
            letter_dir = f"letter_{letter_idx + 1}"  # Directory names start from 1
            letter_path = os.path.join(data_dir, letter_dir)
            
            if not os.path.exists(letter_path):
                logging.warning(f"Directory not found: {letter_path}")
                continue
                
            # Get all image files
            img_files = [f for f in os.listdir(letter_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not img_files:
                logging.warning(f"No images found in {letter_path}")
                continue
            
            # Process each image
            for img_file in img_files:
                img_path = os.path.join(letter_path, img_file)
                img = preprocess_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(letter_idx)  # Use 0-based index for labels
                    label_counts[letter_idx] = label_counts.get(letter_idx, 0) + 1
                    
                    # Generate augmented images for underrepresented classes
                    if label_counts[letter_idx] < 10:  # Minimum samples per class
                        for _ in range(3):  # Generate 3 augmented versions
                            aug_img = augment_image(img)
                            if aug_img is not None:
                                images.append(aug_img)
                                labels.append(letter_idx)
                                label_counts[letter_idx] += 1
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Print dataset statistics
        logging.info("\nDataset Statistics:")
        logging.info(f"Total images: {len(images)}")
        logging.info(f"Number of classes: {len(letter_mapping)}")
        logging.info(f"Images per class: {dict(Counter(labels))}")
        
        if len(images) == 0:
            raise ValueError("No valid images were loaded")
            
        # Convert labels to categorical
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(letter_mapping))
            
        return images, labels, len(letter_mapping)
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def split_data(
    images: List[np.ndarray], 
    labels: List[int], 
    test_size: float = 0.2, 
    val_size: float = 0.2
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Split the data into training, testing, and validation sets
    
    Args:
        images (List[np.ndarray]): Images
        labels (List[int]): Labels
        test_size (float): Test set size as a proportion of the total dataset
        val_size (float): Validation set size as a proportion of the training set
    
    Returns:
        Tuple[Any, Any, Any, Any, Any, Any]: 
        x_train, x_test, x_val, y_train, y_test, y_val
    """
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Calculate class statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)
    n_samples = len(labels)
    min_samples = min(counts)
    
    print(f"Total samples: {n_samples}")
    print(f"Number of classes: {n_classes}")
    print(f"Minimum samples per class: {min_samples}")
    
    # Create indices for each class
    class_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # Initialize empty arrays for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each class proportionally
    for label in unique_labels:
        indices = class_indices[label]
        n_samples_class = len(indices)
        
        # Calculate split sizes
        n_test = max(1, int(n_samples_class * 0.1))  # 10% for test
        n_val = max(1, int((n_samples_class - n_test) * 0.1))  # 10% of remaining for val
        n_train = n_samples_class - n_test - n_val
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Split indices
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Convert indices to arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Create final splits
    x_train = images[train_indices]
    y_train = labels[train_indices]
    x_val = images[val_indices]
    y_val = labels[val_indices]
    x_test = images[test_indices]
    y_test = labels[test_indices]
    
    print(f"Training samples: {len(x_train)} (labels: {len(y_train)})")
    print(f"Validation samples: {len(x_val)} (labels: {len(y_val)})")
    print(f"Testing samples: {len(x_test)} (labels: {len(y_test)})")
    
    # Verify shapes
    print("\nData shapes:")
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return x_train, x_test, x_val, y_train, y_test, y_val

class ThamudicDataset:
    """
    مجموعة بيانات الخط الثمودي مع معالجة متقدمة للصور
    """
    def __init__(self, data_dir: str, label_mapping_file: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load label mapping
        with open(label_mapping_file, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
            
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        # Create label to index mapping
        self.label_to_index = {str(k): i for i, k in enumerate(sorted(self.label_mapping.keys()))}
        
        # Load all images and labels
        for letter_dir in os.listdir(self.data_dir):
            if not letter_dir.startswith('letter_'):
                continue
                
            letter_num = letter_dir.split('_')[1]
            if letter_num not in self.label_to_index:
                continue
                
            letter_path = self.data_dir / letter_dir
            for img_file in letter_path.glob('*.png'):
                self.image_paths.append(img_file)
                self.labels.append(self.label_to_index[letter_num])
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
            
        logging.info(f"Found {len(self.image_paths)} images")
        logging.info(f"Number of classes: {len(set(self.labels))}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = preprocess_image(img_path)
        
        # Apply additional transforms if specified
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label
        
    def get_class_weights(self):
        """
        حساب أوزان الفئات للتعامل مع عدم التوازن في البيانات
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return class_weights

def get_data_loaders(data_dir: str, label_mapping_file: str, batch_size: int = 32, num_workers: int = 4):
    """Create train and validation data loaders"""
    train_dataset = ThamudicDataset(data_dir, label_mapping_file, train=True)
    val_dataset = ThamudicDataset(data_dir, label_mapping_file, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def main():
    # Load the data
    images, labels, num_classes = load_data(data_dir='path_to_data', label_mapping_file='path_to_label_mapping')
    
    # Split the data
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    print("Data information:")
    print(f"Total number of images: {len(images)}")
    print(f"Number of training images: {len(x_train)}")
    print(f"Number of testing images: {len(x_test)}")
    print(f"Number of validation images: {len(x_val)}")

if __name__ == "__main__":
    main()
