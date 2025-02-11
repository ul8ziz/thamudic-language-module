"""
إعداد هيكل المشروع وتهيئة المسارات
"""

import os
from pathlib import Path
import json
import shutil

def create_directory(path: Path):
    """إنشاء مجلد إذا لم يكن موجوداً"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"تم إنشاء المجلد: {path}")

def setup_project_structure():
    """إعداد هيكل المشروع الكامل"""
    # المسار الرئيسي للمشروع
    project_root = Path(__file__).parent
    
    # إنشاء المجلدات الرئيسية
    directories = {
        'src': project_root / 'src',
        'data': project_root / 'data',
        'models': project_root / 'models',
        'runs': project_root / 'runs',
        'tests': project_root / 'tests',
    }
    
    # المجلدات الفرعية
    subdirectories = {
        'data/raw': directories['data'] / 'raw',
        'data/letters': directories['data'] / 'letters',
        'models/checkpoints': directories['models'] / 'checkpoints',
        'models/configs': directories['models'] / 'configs',
        'runs/logs': directories['runs'] / 'logs',
        'runs/tensorboard': directories['runs'] / 'tensorboard',
    }
    
    # إنشاء كل المجلدات
    for dir_path in {**directories, **subdirectories}.values():
        create_directory(dir_path)
    
    # إنشاء ملف تعيين الحروف إذا لم يكن موجوداً
    mapping_file = directories['data'] / 'mapping.json'
    if not mapping_file.exists():
        letter_mapping = {
            "thamudic_letters": [
                {"name": letter, "index": i} 
                for i, letter in enumerate("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
            ],
            "version": "1.0",
            "description": "تعيين الحروف العربية للتصنيف",
            "direction": "rtl"
        }
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(letter_mapping, f, ensure_ascii=False, indent=4)
        print(f"تم إنشاء ملف التعيين: {mapping_file}")
    
    # نقل الملفات الموجودة إلى المواقع الصحيحة
    file_moves = [
        (project_root / 'src' / 'app.py', directories['src'] / 'app.py'),
        (project_root / 'src' / 'models.py', directories['src'] / 'models.py'),
        (project_root / 'src' / 'train.py', directories['src'] / 'train.py'),
        (project_root / 'src' / 'data_processing.py', directories['src'] / 'data_processing.py'),
    ]
    
    for src, dst in file_moves:
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"تم نقل الملف: {src} -> {dst}")
    
    print("\nتم إعداد هيكل المشروع بنجاح!")
    print("\nهيكل المشروع:")
    print("="*50)
    for root, dirs, files in os.walk(project_root):
        level = root.replace(str(project_root), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == '__main__':
    setup_project_structure()
