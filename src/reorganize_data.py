import os
import shutil
from pathlib import Path

def reorganize_data():
    """
    إعادة تنظيم مجلدات البيانات
    """
    base_dir = Path('data/letters/processed_letters')
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    
    # إنشاء مجلد مؤقت لحفظ البيانات الحالية
    temp_dir = base_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # نقل المجلدات الحالية إلى المجلد المؤقت
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('letter_'):
            shutil.move(str(item), str(temp_dir / item.name))
    
    # إعادة إنشاء مجلدات التدريب والتحقق
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # نقل المجلدات من المجلد المؤقت إلى مجلدات التدريب والتحقق
    for item in temp_dir.iterdir():
        if item.is_dir():
            # نقل المحتوى إلى مجلد التدريب
            train_letter_dir = train_dir / item.name
            train_letter_dir.mkdir(exist_ok=True)
            
            # نقل المحتوى إلى مجلد التحقق
            val_letter_dir = val_dir / item.name
            val_letter_dir.mkdir(exist_ok=True)
            
            # نقل الملفات
            files = list(item.glob('*.*'))
            split_idx = int(len(files) * 0.8)  # 80% للتدريب
            
            # نقل ملفات التدريب
            for file in files[:split_idx]:
                shutil.copy2(str(file), str(train_letter_dir / file.name))
            
            # نقل ملفات التحقق
            for file in files[split_idx:]:
                shutil.copy2(str(file), str(val_letter_dir / file.name))
            
            print(f'Processed {item.name}: {split_idx} for training, {len(files)-split_idx} for validation')
    
    # حذف المجلد المؤقت
    shutil.rmtree(str(temp_dir))
    print('Data reorganization completed successfully!')

if __name__ == '__main__':
    reorganize_data()
