# Thamudic Language Module | وحدة اللغة الثمودية

## Overview | نظرة عامة
The Thamudic Language Module is an advanced deep learning application designed for the accurate recognition and analysis of ancient Thamudic inscriptions. It leverages cutting-edge computer vision and neural network methodologies to deliver high-performance character recognition with a user-friendly web interface.

وحدة اللغة الثمودية هي تطبيق متقدم للتعلم العميق مصمم للتعرف الدقيق وتحليل النقوش الثمودية القديمة. تستفيد من منهجيات الرؤية الحاسوبية والشبكات العصبية المتطورة لتقديم أداء عالي في التعرف على الأحرف مع واجهة مستخدم سهلة الاستخدام.

## Key Features | الميزات الرئيسية
- **Accurate Character Recognition | التعرف الدقيق على الأحرف**: Employs state-of-the-art deep learning models for precise identification of Thamudic characters | يستخدم أحدث نماذج التعلم العميق للتعرف الدقيق على الأحرف الثمودية
- **Interactive Web Interface | واجهة ويب تفاعلية**: Built with Streamlit for easy interaction and visualization | مبنية باستخدام Streamlit لسهولة التفاعل والتصور
- **Advanced Image Preprocessing | معالجة متقدمة للصور**: Sophisticated techniques to enhance image quality | تقنيات متطورة لتحسين جودة الصورة
- **Real-Time Processing | معالجة في الوقت الحقيقي**: Process images in real-time with immediate feedback | معالجة الصور في الوقت الحقيقي مع تغذية راجعة فورية
- **Comprehensive Analysis | تحليل شامل**: Detailed insights and visual representations of recognition results | رؤى مفصلة وتمثيلات مرئية لنتائج التعرف
- **Multi-Platform Support | دعم متعدد المنصات**: Works across different operating systems | يعمل عبر أنظمة تشغيل مختلفة

## Key Libraries | المكتبات الرئيسية

The Thamudic Language Module leverages several powerful libraries to deliver high-performance character recognition and analysis:

تستخدم وحدة اللغة الثمودية العديد من المكتبات القوية لتقديم أداء عالي في التعرف على الأحرف وتحليلها:

### Deep Learning & Neural Networks | التعلم العميق والشبكات العصبية
- **PyTorch (`torch`)**: Primary deep learning framework for model development and training | إطار التعلم العميق الأساسي لتطوير النماذج والتدريب
- **TensorFlow**: Additional neural network capabilities and model optimization | قدرات إضافية للشبكات العصبية وتحسين النماذج
- **torchvision**: Computer vision extensions for PyTorch | امتدادات الرؤية الحاسوبية لـ PyTorch
- **TensorBoard**: Visualization and monitoring of training metrics | تصور ومراقبة مقاييس التدريب
- **Weights & Biases (`wandb`)**: Experiment tracking and model management | تتبع التجارب وإدارة النماذج

### Image Processing | معالجة الصور
- **OpenCV (`cv2`)**: Advanced image processing and computer vision operations | عمليات معالجة الصور المتقدمة والرؤية الحاسوبية
- **Pillow (`PIL`)**: Image manipulation and enhancement | معالجة وتحسين الصور
- **Albumentations**: Advanced image augmentation library for machine learning | مكتبة متقدمة لتعزيز الصور للتعلم الآلي
- **scikit-image**: Additional image processing utilities | أدوات إضافية لمعالجة الصور

### Data Science & Machine Learning | علوم البيانات والتعلم الآلي
- **NumPy**: Numerical computing and array operations | الحوسبة العددية وعمليات المصفوفات
- **Pandas**: Data manipulation and analysis | معالجة وتحليل البيانات
- **scikit-learn**: Machine learning algorithms and utilities | خوارزميات وأدوات التعلم الآلي
- **SciPy**: Scientific and technical computing | الحوسبة العلمية والتقنية

### Web Application & Visualization | تطبيق الويب والتصور
- **Streamlit**: Interactive web application framework | إطار تطبيق ويب تفاعلي
- **Plotly**: Advanced data visualization | تصور متقدم للبيانات
- **python-dotenv**: Environment variable management | إدارة متغيرات البيئة

### Utility Libraries | مكتبات المساعدة
- **tqdm**: Progress bar for long-running operations | شريط التقدم للعمليات طويلة المدى
- **PyYAML**: YAML configuration file parsing | تحليل ملفات تكوين YAML
- **pathlib**: Cross-platform path manipulation | معالجة المسارات عبر المنصات المختلفة
- **requests**: HTTP library for API interactions | مكتبة HTTP للتفاعلات مع واجهات برمجة التطبيقات
- **python-json-logger**: JSON-formatted logging | تسجيل منسق بتنسيق JSON

### Development & Testing | التطوير والاختبار
- **pytest**: Unit testing framework | إطار اختبار الوحدات
- **Black**: Code formatting | تنسيق الكود
- **isort**: Import sorting | ترتيب الاستيرادات
- **flake8**: Code linting | فحص الكود
- **mypy**: Static type checking | التحقق من النوع الثابت

Each library plays a crucial role in the project's functionality, from data processing and model training to web interface development and code quality assurance.

تلعب كل مكتبة دورًا حاسمًا في وظائف المشروع، من معالجة البيانات وتدريب النماذج إلى تطوير واجهة الويب وضمان جودة الكود.

## System Requirements | متطلبات النظام

### Hardware Requirements | متطلبات الأجهزة
- **CPU**: Multi-core processor (Intel/AMD) | معالج متعدد النواة (Intel/AMD)
- **RAM**: Minimum 8GB (16GB recommended) | الحد الأدنى 8 جيجابايت (يوصى بـ 16 جيجابايت)
- **GPU**: NVIDIA GPU with CUDA support (optional, but recommended) | بطاقة رسومات NVIDIA مع دعم CUDA (اختياري، ولكن موصى به)
- **Storage**: 2GB minimum free space | مساحة حرة لا تقل عن 2 جيجابايت

### Software Requirements | متطلبات البرمجيات
- **Python**: Version 3.10 or higher | الإصدار 3.10 أو أعلى
- **CUDA Toolkit**: Version 11.2 or higher (for GPU acceleration) | الإصدار 11.2 أو أعلى (لتسريع وحدة معالجة الرسومات)
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS | نظام التشغيل: Windows 10/11، Linux (Ubuntu 20.04+)، أو macOS

## Installation | التثبيت

1. Clone the repository | استنساخ المستودع:
```bash
git clone https://github.com/ul8ziz/thamudic-language-module.git
cd thamudic-language-module
```

2. Create and activate a virtual environment (recommended) | إنشاء وتفعيل بيئة افتراضية (موصى به):
```bash
python -m venv venv
# On Windows | على ويندوز
.\venv\Scripts\activate
# On Linux/macOS | على لينكس/ماك
source venv/bin/activate
```

3. Install dependencies | تثبيت التبعيات:
```bash
pip install -r requirements.txt
```

## Usage | الاستخدام

1. Start the web application | تشغيل تطبيق الويب:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501) | افتح متصفح الويب الخاص بك وانتقل إلى عنوان URL المقدم (عادةً http://localhost:8501)

3. Upload an image containing Thamudic inscriptions | قم بتحميل صورة تحتوي على نقوش ثمودية

4. View the recognition results and analysis | عرض نتائج التعرف والتحليل

## Project Structure | هيكل المشروع
```
thamudic-language-module/
├── data/                      # Data files | ملفات البيانات
│   ├── letter_mapping.json    # Character mapping definitions | تعريفات تخطيط الأحرف
│   ├── letters/               # Character image dataset | مجموعة بيانات صور الأحرف
│   └── raw/                   # Raw data files | ملفات البيانات الخام
│
├── models/                    # Model files | ملفات النموذج
│   ├── checkpoints/           # Saved model checkpoints | نقاط حفظ النموذج
│   ├── logs/                  # Training logs | سجلات التدريب
│   └── train_logs.txt         # Detailed training logs | سجلات التدريب المفصلة
│
├── src/                       # Source code | الكود المصدري
│   ├── app.py                 # Main Streamlit application | تطبيق Streamlit الرئيسي
│   ├── assets/                # Application assets (fonts, images) | أصول التطبيق (خطوط، صور)
│   ├── config.py              # Configuration settings | إعدادات التكوين
│   ├── data_processing.py     # Data processing utilities | أدوات معالجة البيانات
│   ├── image_processing.py    # Image processing utilities | أدوات معالجة الصور
│   ├── models.py              # Neural network model definitions | تعريفات نموذج الشبكة العصبية
│   ├── reorganize_data.py     # Data organization utilities | أدوات تنظيم البيانات
│   └── train.py               # Model training script | نص تدريب النموذج
│
├── runs/                      # TensorBoard run data | بيانات تشغيل TensorBoard
├── tests/                     # Test files | ملفات الاختبار
├── requirements.txt           # Project dependencies | تبعيات المشروع
├── run_app.bat                # Windows batch file to run the app | ملف دفعي لتشغيل التطبيق على ويندوز
├── PROJECT_SUMMARY.md         # Technical project summary | ملخص تقني للمشروع
└── README.md                  # Project documentation | توثيق المشروع
```

## Contributing | المساهمة
Contributions are welcome! Please feel free to submit a Pull Request.

المساهمات مرحب بها! لا تتردد في تقديم طلب سحب.

## License | الترخيص
This project is licensed under the MIT License - see the LICENSE file for details.

هذا المشروع مرخص بموجب ترخيص MIT - راجع ملف LICENSE للحصول على التفاصيل.

## Contact | الاتصال
For questions and support, please open an issue in the GitHub repository.

للأسئلة والدعم، يرجى فتح مشكلة في مستودع GitHub.
