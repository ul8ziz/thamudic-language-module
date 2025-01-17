import json
import yaml
import logging
from pathlib import Path

def load_char_mapping(mapping_file):
    """
    تحميل خريطة الأحرف من ملف JSON
    
    Args:
        mapping_file (str): مسار ملف خريطة الأحرف
    
    Returns:
        dict: قاموس يربط رقم الفهرس بالحرف
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            char_mapping = json.load(f)
        
        # عكس القاموس ليكون المفتاح هو الرقم والقيمة هي الحرف
        index_to_char = {int(v): k for k, v in char_mapping.items()}
        
        return index_to_char
    
    except Exception as e:
        logging.error(f"Error loading character mapping: {e}")
        return {}

def load_config(config_path):
    """
    تحميل ملف الإعدادات
    
    Args:
        config_path (str): مسار ملف الإعدادات
        
    Returns:
        dict: الإعدادات المحملة
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """
    إعداد التسجيل
    
    Returns:
        logging.Logger: المسجل المعد
    """
    logger = logging.getLogger('thamudic')
    logger.setLevel(logging.INFO)
    
    # إنشاء مجلد السجلات إذا لم يكن موجوداً
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # إعداد معالج الملف
    fh = logging.FileHandler(log_dir / 'training.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # إعداد معالج وحدة التحكم
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # تنسيق السجلات
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # إضافة المعالجات
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
