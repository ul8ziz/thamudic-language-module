# مشروع التعرف على اللغة الثمودية

## الوصف
مشروع لتصنيف وتحليل النصوص الثمودية باستخدام التعلم العميق والشبكات العصبية التلافيفية.

## المتطلبات
- Python 3.8+
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- seaborn

## تثبيت التبعيات
```bash
pip install -r requirements.txt
```

## هيكل المشروع
- `Scripts/data_preprocessing.py`: معالجة وتحضير البيانات
- `Scripts/train_model.py`: تدريب نموذج التعرف على اللغة
- `Scripts/evaluate_model.py`: تقييم أداء النموذج
- `Scripts/dataset_creator.py`: إنشاء مجموعة بيانات الحروف

## إنشاء مجموعة بيانات الحروف

### خطوات إعداد مجموعة البيانات
1. وضع صورة الحروف في مجلد `data/raw`
2. تشغيل `Scripts/dataset_creator.py`
3. ستتم معالجة الصورة وتقسيمها تلقائيًا

### هيكل مجلد البيانات
```
data/
├── raw/
│   └── letters_image.png
└── letters/
    ├── thamudic/
    │   ├── letter_0.png
    │   ├── letter_1.png
    │   └── ...
    └── arabic/
        ├── letter_0.png
        ├── letter_1.png
        └── ...
```

### ملاحظات مهمة
- تأكد من جودة الصورة ووضوح الحروف
- قد تحتاج لتعديل كود `dataset_creator.py` حسب شكل صورتك

## خطوات التشغيل
1. وضع الصور الخام في مجلد `data/raw`
2. تشغيل `data_preprocessing.py` لمعالجة الصور
3. تشغيل `train_model.py` لتدريب النموذج
4. تشغيل `evaluate_model.py` لتقييم النموذج

## ملاحظات
- تأكد من تنظيم الصور في مجلدات حسب الفئات
- يمكن ضبط معلمات النموذج في ملفات التدريب والتقييم

## المساهمة
للمساهمة في المشروع، يرجى إنشاء pull request أو فتح issue.
