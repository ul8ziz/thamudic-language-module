# نظام التعرف على النصوص الثمودية
# Thamudic Text Recognition System

## نظرة عامة | Overview

نظام متقدم للتعرف على النصوص الثمودية وترجمتها إلى العربية باستخدام تقنيات التعلم العميق والرؤية الحاسوبية. يستخدم النظام PyTorch مع واجهة مستخدم تفاعلية مبنية على Streamlit.

An advanced system for recognizing Thamudic texts and translating them to Arabic using deep learning and computer vision techniques. The system uses PyTorch with an interactive Streamlit interface.

## المميزات | Features

### 🔍 التعرف على النصوص | Text Recognition
- نموذج عميق مبني على PyTorch مع تحسينات متقدمة
- دعم للتعرف على 28 حرفاً ثمودياً
- معالجة متقدمة للصور باستخدام Albumentations
- تحليل الثقة في التنبؤات مع إحصاءات مفصلة
- واجهة مستخدم تفاعلية باستخدام Streamlit

### 🚀 الأداء | Performance
- تحسين النموذج باستخدام CUDA للاستدلال السريع
- دعم معالجة الدفعات للكفاءة العالية
- استخدام الدقة المختلطة لتحسين الأداء
- تحسين استخدام GPU مع PyTorch AMP

### 📊 التحليل والتقارير | Analysis & Reporting
- تقارير تدريب مفصلة مع TensorBoard
- تحليل شامل للتنبؤات وثقة النموذج
- تتبع الأداء عبر الزمن
- تصدير النتائج بتنسيقات متعددة

## المتطلبات | Requirements

```bash
# Deep Learning & Neural Networks
torch>=2.1.0
torchvision>=0.16.0
tensorboard>=2.12.0

# Image Processing
opencv-python>=4.10.0
albumentations>=1.4.24
Pillow>=10.2.0

# Data Science & ML
numpy>=1.24.3
pandas>=2.1.4
scikit-learn>=1.3.0

# Web Interface
streamlit>=1.29.0
streamlit-drawable-canvas>=0.9.3
```

## التثبيت | Installation

```bash
git clone https://github.com/yourusername/thamudic-language-module.git
cd thamudic-language-module
pip install -r requirements.txt
```

## الاستخدام | Usage

### تشغيل واجهة المستخدم | Running the UI

```bash
streamlit run src/app.py
```

### تدريب النموذج | Training the Model

```python
from src.train import train_model
from src.data_loader import ThamudicDataset

# تجهيز البيانات
data_dir = "data/letters/improved_letters"
train_dataset = ThamudicDataset(data_dir, split='train')
val_dataset = ThamudicDataset(data_dir, split='val')

# تدريب النموذج
train_model(
    data_dir=data_dir,
    mapping_file="data/mapping.json",
    model_save_path="models/best_model.pth",
    batch_size=32,
    epochs=50,
    learning_rate=0.001
)
```

### استخدام النموذج للتنبؤ | Using the Model for Inference

```python
import torch
from src.thamudic_model import ThamudicRecognitionModel
from PIL import Image
import cv2

# تحميل النموذج
model = ThamudicRecognitionModel()
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# التنبؤ بصورة واحدة
image = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# إضافة معالجة الصورة هنا...
prediction = model(image)
```

## الهيكل | Structure

```
thamudic-language-module/
├── data/
│   ├── letters/
│   │   ├── improved_letters/    # الصور المحسنة
│   │   ├── processed_letters/   # الصور المعالجة
│   │   └── raw/                 # الصور الأصلية
│   ├── mapping.json            # تعيين الحروف
│   └── thamudic_to_arabic.json # ترجمة الحروف
├── models/                     # النماذج المدربة
├── src/
│   ├── app.py                  # تطبيق Streamlit
│   ├── train.py               # تدريب النموذج
│   ├── data_loader.py         # معالجة البيانات
│   ├── thamudic_model.py      # تعريف النموذج
│   ├── improve_dataset_quality.py  # تحسين جودة الصور
│   └── visualization.py       # أدوات التصور
└── README.md
```

## المساهمة | Contributing

نرحب بمساهماتكم! يرجى اتباع هذه الخطوات:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## الترخيص | License

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
