# نظام التعرف على النصوص الثمودية
# Thamudic Text Recognition System

## نظرة عامة | Overview

نظام متقدم للتعرف على النصوص الثمودية وترجمتها إلى العربية باستخدام تقنيات التعلم العميق والرؤية الحاسوبية.

An advanced system for recognizing Thamudic texts and translating them to Arabic using deep learning and computer vision techniques.

## المميزات | Features

### 🔍 التعرف على النصوص | Text Recognition
- نموذج عميق مبني على EfficientNetB0 مع تحسينات متقدمة
- دعم للتعرف على 32 حرفاً ثمودياً
- معالجة متقدمة للصور مع تقنيات تحسين متعددة
- تحليل الثقة في التنبؤات مع إحصاءات مفصلة

### 🚀 الأداء | Performance
- تحسين النموذج باستخدام TensorRT للاستدلال السريع
- دعم معالجة الدفعات للكفاءة العالية
- استخدام الدقة المختلطة لتحسين الأداء
- تحسين استخدام GPU

### 📊 التحليل والتقارير | Analysis & Reporting
- تقارير تدريب مفصلة مع رسوم بيانية متقدمة
- تحليل شامل للتنبؤات وثقة النموذج
- تتبع الأداء عبر الزمن
- تصدير النتائج بتنسيقات متعددة

## المتطلبات | Requirements

```bash
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
albumentations>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## التثبيت | Installation

```bash
git clone https://github.com/yourusername/thamudic-language-module.git
cd thamudic-language-module
pip install -r requirements.txt
```

## الاستخدام | Usage

### تدريب النموذج | Training the Model

```python
from src.core.model_trainer import train_model
from src.data.data_pipeline import load_data

# Load and preprocess data
train_images, train_labels, val_images, val_labels = load_data()

# Train model with advanced configuration
model, history = train_model(
    train_images, 
    train_labels,
    val_images,
    val_labels,
    model_dir="models",
    num_classes=32
)
```

### استخدام النموذج للتنبؤ | Using the Model for Inference

```python
from src.core.inference_engine import ThamudicInferenceEngine

# Initialize inference engine with advanced configuration
engine = ThamudicInferenceEngine(
    model_path="models/best_model.h5",
    label_encoder_path="models/label_encoder.pkl",
    config={
        'confidence_threshold': 0.7,
        'top_k': 3,
        'use_gpu': True,
        'batch_size': 16
    }
)

# Single image prediction
result = engine.predict_single("path/to/image.jpg")
print(f"Top prediction: {result['top_predictions'][0]['label']}")
print(f"Confidence: {result['top_predictions'][0]['confidence']:.2f}")

# Batch prediction
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = engine.predict_batch(image_paths)

# Analyze predictions
analysis = engine.analyze_predictions(results)
print(f"Mean confidence: {analysis['confidence_stats']['mean']:.2f}")
```

## الهيكل | Structure

```
thamudic-language-module/
├── data/
│   ├── letters/              # صور الحروف الثمودية
│   └── letter_mapping.json   # تعيين الحروف
├── models/                   # النماذج المدربة
├── src/
│   ├── core/
│   │   ├── model_trainer.py  # تدريب النموذج
│   │   └── inference_engine.py # محرك الاستدلال
│   └── data/
│       └── data_pipeline.py  # معالجة البيانات
└── README.md
```

## المساهمة | Contributing

نرحب بمساهماتكم! يرجى اتباع هذه الخطوات:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## الترخيص | License

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## الاتصال | Contact

- البريد الإلكتروني | Email: your.email@example.com
- تويتر | Twitter: [@yourusername](https://twitter.com/yourusername)
