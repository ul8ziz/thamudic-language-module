import os
import shutil
from pathlib import Path
import random
import sys
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from scipy import ndimage

# تعيين ترميز المخرجات
sys.stdout.reconfigure(encoding='utf-8')

def enhance_image_quality(image_path, output_path=None):
    """
    تحسين جودة الصورة باستخدام تقنيات متقدمة
    
    Args:
        image_path (str): مسار الصورة المدخلة
        output_path (str): مسار حفظ الصورة المحسنة (اختياري)
    
    Returns:
        PIL.Image: الصورة المحسنة
    """
    try:
        # قراءة الصورة باستخدام OpenCV للمعالجة المتقدمة
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError("فشل في قراءة الصورة")
        
        # 1. تحويل الصورة إلى تدرج رمادي
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # 2. تحسين التباين باستخدام CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 3. إزالة الضوضاء باستخدام مرشح ثنائي
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 4. تطبيق عتبة تكيفية
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # 5. تحسين الحروف باستخدام العمليات المورفولوجية
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 6. تنعيم الحواف
        smoothed = cv2.GaussianBlur(binary, (3,3), 0)
        
        # 7. تحويل إلى صورة PIL
        enhanced_img = Image.fromarray(smoothed)
        
        # حفظ الصورة إذا تم تحديد مسار
        if output_path:
            enhanced_img.save(output_path)
        
        return enhanced_img
        
    except Exception as e:
        print(f"خطأ في معالجة الصورة: {str(e)}")
        return None

def preprocess_image(image_path):
    """
    معالجة الصورة وتحسين جودتها
    
    Args:
        image_path (str): مسار الصورة
    
    Returns:
        PIL.Image: الصورة المعالجة
    """
    try:
        # تحسين جودة الصورة
        enhanced_img = enhance_image_quality(image_path)
        if enhanced_img is None:
            return None
            
        # التحقق من جودة الصورة
        img_array = np.array(enhanced_img)
        
        # حساب نسبة البكسلات غير الفارغة
        non_empty_ratio = np.count_nonzero(img_array) / img_array.size
        
        # التحقق من معايير الجودة
        if non_empty_ratio < 0.01 or non_empty_ratio > 0.99:  # صورة شبه فارغة أو مليئة بالضوضاء
            return None
            
        if img_array.std() < 20:  # تباين منخفض جداً
            return None
            
        return enhanced_img
        
    except Exception as e:
        print(f"خطأ في معالجة الصورة {image_path}: {e}")
        return None

def augment_image(img):
    """
    إنشاء نسخة معدلة من الصورة
    
    Args:
        img (PIL.Image): الصورة الأصلية
    
    Returns:
        PIL.Image: الصورة المعدلة
    """
    try:
        # تحويل الصورة إلى مصفوفة NumPy
        img_array = np.array(img)
        
        # قائمة التحويلات المحتملة
        transformations = [
            # 1. تدوير عشوائي بزاوية صغيرة
            lambda x: ndimage.rotate(x, random.uniform(-10, 10), reshape=False, mode='constant', cval=255),
            
            # 2. إزاحة عشوائية
            lambda x: ndimage.shift(x, (random.uniform(-10, 10), random.uniform(-10, 10)), mode='constant', cval=255),
            
            # 3. تغيير الحجم
            lambda x: cv2.resize(x, None, fx=random.uniform(0.8, 1.2), fy=random.uniform(0.8, 1.2)),
            
            # 4. تشويه منظوري بسيط
            lambda x: cv2.warpPerspective(
                x,
                cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [x.shape[1], 0], [x.shape[1], x.shape[0]], [0, x.shape[0]]]),
                    np.float32([[random.uniform(0, 20), random.uniform(0, 20)],
                               [x.shape[1] - random.uniform(0, 20), random.uniform(0, 20)],
                               [x.shape[1] - random.uniform(0, 20), x.shape[0] - random.uniform(0, 20)],
                               [random.uniform(0, 20), x.shape[0] - random.uniform(0, 20)]])
                ),
                (x.shape[1], x.shape[0])
            ),
        ]
        
        # اختيار تحويل عشوائي وتطبيقه
        transform = random.choice(transformations)
        augmented = transform(img_array)
        
        # التأكد من أن القيم في النطاق الصحيح
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        
        # تحويل مرة أخرى إلى صورة PIL
        return Image.fromarray(augmented)
    
    except Exception as e:
        print(f"خطأ في زيادة البيانات: {str(e)}")
        return img  # إرجاع الصورة الأصلية في حالة الخطأ

def find_coeffs(pa, pb):
    """
    حساب معاملات التحويل المنظوري
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def balance_dataset(src_dir, target_count=None):
    """
    موازنة عدد الصور لكل حرف عن طريق تكرار الصور في الفئات القليلة
    
    Args:
        src_dir (str): مجلد المصدر
        target_count (int): العدد المستهدف للصور لكل حرف. إذا لم يتم تحديده، سيتم استخدام عدد أكبر فئة
    """
    import shutil
    from PIL import Image, ImageEnhance
    import numpy as np
    import random
    
    # حساب عدد الصور لكل حرف
    letter_counts = {}
    letter_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    print("عدد الصور الأصلي لكل حرف:")
    for letter_dir in letter_dirs:
        images = [f for f in os.listdir(os.path.join(src_dir, letter_dir)) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
        letter_counts[letter_dir] = len(images)
        print(f"{letter_dir}: {len(images)} صورة")
    
    # تحديد العدد المستهدف كأكبر عدد صور
    if target_count is None:
        target_count = max(letter_counts.values())
    
    print(f"\nالعدد المستهدف لكل حرف: {target_count}")
    
    # موازنة الصور لكل حرف
    for letter_dir in letter_dirs:
        src_letter_dir = os.path.join(src_dir, letter_dir)
        current_count = letter_counts[letter_dir]
        
        if current_count < target_count:
            # الحصول على قائمة الصور الحالية
            images = [f for f in os.listdir(src_letter_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # عدد الصور المطلوب إضافتها
            needed_copies = target_count - current_count
            print(f"\nإضافة {needed_copies} صورة لحرف {letter_dir}")
            
            # تكرار الصور مع تعديلات بسيطة
            for i in range(needed_copies):
                # اختيار صورة عشوائية للتكرار
                source_img_name = random.choice(images)
                source_img_path = os.path.join(src_letter_dir, source_img_name)
                
                # إنشاء اسم جديد للصورة
                new_img_name = f"aug_{i}_{source_img_name}"
                new_img_path = os.path.join(src_letter_dir, new_img_name)
                
                # فتح وتعديل الصورة
                try:
                    img = Image.open(source_img_path).convert('L')
                    
                    # تطبيق تعديلات عشوائية
                    # 1. تدوير عشوائي بسيط
                    angle = random.uniform(-10, 10)
                    img = img.rotate(angle, expand=False, fillcolor=255)
                    
                    # 2. تغيير التباين قليلاً
                    enhancer = ImageEnhance.Contrast(img)
                    factor = random.uniform(0.8, 1.2)
                    img = enhancer.enhance(factor)
                    
                    # 3. تغيير السطوع قليلاً
                    enhancer = ImageEnhance.Brightness(img)
                    factor = random.uniform(0.8, 1.2)
                    img = enhancer.enhance(factor)
                    
                    # حفظ الصورة الجديدة
                    img.save(new_img_path)
                    
                except Exception as e:
                    print(f"خطأ في معالجة الصورة {source_img_name}: {e}")
                    continue
    
    # طباعة الإحصائيات النهائية
    print("\nعدد الصور النهائي لكل حرف:")
    for letter_dir in letter_dirs:
        images = [f for f in os.listdir(os.path.join(src_dir, letter_dir)) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{letter_dir}: {len(images)} صورة")

def split_data(src_dir, train_dir, val_dir, val_split=0.2):
    """
    تقسيم البيانات إلى مجموعات تدريب وتحقق مع معالجة الصور
    
    Args:
        src_dir (str): مجلد المصدر
        train_dir (str): مجلد التدريب
        val_dir (str): مجلد التحقق
        val_split (float): نسبة بيانات التحقق
    """
    # إنشاء المجلدات
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    
    # موازنة البيانات أولاً
    balance_dataset(src_dir)
    
    letter_dirs = [d for d in os.listdir(src_dir) 
                  if os.path.isdir(os.path.join(src_dir, d)) and d not in ['train', 'val']]
    
    for letter_dir in letter_dirs:
        # إنشاء المجلدات
        train_letter_dir = os.path.join(train_dir, letter_dir)
        val_letter_dir = os.path.join(val_dir, letter_dir)
        Path(train_letter_dir).mkdir(exist_ok=True)
        Path(val_letter_dir).mkdir(exist_ok=True)
        
        # معالجة الصور وتقسيمها
        src_letter_dir = os.path.join(src_dir, letter_dir)
        images = [f for f in os.listdir(src_letter_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # خلط الصور
        random.shuffle(images)
        
        # تقسيم الصور
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # معالجة ونقل صور التدريب
        for img in train_images:
            src_path = os.path.join(src_letter_dir, img)
            processed_img = preprocess_image(src_path)
            if processed_img:
                dst_path = os.path.join(train_letter_dir, img)
                processed_img.save(dst_path)
        
        # معالجة ونقل صور التحقق
        for img in val_images:
            src_path = os.path.join(src_letter_dir, img)
            processed_img = preprocess_image(src_path)
            if processed_img:
                dst_path = os.path.join(val_letter_dir, img)
                processed_img.save(dst_path)
        
        print(f'Processed {letter_dir}: {len(train_images)} for training, {len(val_images)} for validation')

if __name__ == '__main__':
    # تحديد المسارات
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'data', 'letters', 'thamudic_letters')
    train_dir = os.path.join(base_dir, 'data', 'processed', 'train')
    val_dir = os.path.join(base_dir, 'data', 'processed', 'val')
    
    print(f"المسار الأساسي: {base_dir}")
    print(f"مجلد المصدر: {src_dir}")
    print(f"مجلد التدريب: {train_dir}")
    print(f"مجلد التحقق: {val_dir}")
    
    # إنشاء المجلدات إذا لم تكن موجودة
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    try:
        # موازنة البيانات
        print("\nجاري موازنة البيانات...")
        balance_dataset(src_dir)
        
        # تقسيم البيانات
        print("\nجاري تقسيم البيانات...")
        split_data(src_dir, train_dir, val_dir, val_split=0.2)
        
        print("\nتم تحضير البيانات بنجاح!")
        
    except Exception as e:
        print(f"\nحدث خطأ: {e}")
