import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

def organize_dataset(data_dir, train_ratio=0.8):
    """
    تنظيم مجموعة البيانات إلى مجلدات train و val
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    # إنشاء المجلدات
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # معالجة كل مجلد حرف
    for letter_dir in data_dir.glob('letter_*'):
        if not letter_dir.is_dir() or letter_dir.name in ['train', 'val']:
            continue
            
        # الحصول على قائمة الصور
        images = list(letter_dir.glob('*.png'))
        
        # تقسيم الصور إلى train و val
        train_images, val_images = train_test_split(
            images,
            train_size=train_ratio,
            random_state=42,
            shuffle=True
        )
        
        # إنشاء مجلدات الحروف
        letter_train_dir = train_dir / letter_dir.name
        letter_val_dir = val_dir / letter_dir.name
        letter_train_dir.mkdir(parents=True, exist_ok=True)
        letter_val_dir.mkdir(parents=True, exist_ok=True)
        
        # نسخ الصور
        for img in train_images:
            shutil.copy2(img, letter_train_dir / img.name)
        for img in val_images:
            shutil.copy2(img, letter_val_dir / img.name)
            
        print(f"تم تنظيم {letter_dir.name}: {len(train_images)} للتدريب، {len(val_images)} للتحقق")

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data' / 'letters' / 'improved_letters'
    organize_dataset(data_dir)
    print("تم تنظيم البيانات بنجاح!")
