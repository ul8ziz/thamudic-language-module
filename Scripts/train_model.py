import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import load_data, split_data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_model(num_classes):
    """
    إنشاء نموذج شبكة عصبية تلافيفية للتصنيف
    """
    model = models.Sequential([
        # الطبقة الأولى التلافيفية
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # الطبقة الثانية التلافيفية
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # الطبقة الثالثة التلافيفية
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # طبقة التسوية
        layers.Flatten(),
        
        # الطبقات الكثافة
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # طبقة الإخراج
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # تجميع النموذج
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(train_images, train_labels, val_images, val_labels, model_dir="../models", num_classes=None):
    """
    تدريب النموذج مع التحسينات
    """
    # إنشاء مجلد النماذج
    os.makedirs(model_dir, exist_ok=True)
    
    # تحديد عدد الفئات
    if num_classes is None:
        num_classes = len(np.unique(train_labels))
    
    # تشكيل البيانات للنموذج
    train_images = train_images.reshape(-1, 128, 128, 1)
    val_images = val_images.reshape(-1, 128, 128, 1)
    
    # إنشاء النموذج
    model = create_model(num_classes)
    
    # إنشاء مولد البيانات مع التحسين
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])
    
    # حساب أوزان الفئات
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # إنشاء callbacks
    callbacks = [
        # حفظ أفضل نموذج
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        # إيقاف مبكر
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        # تقليل معدل التعلم
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.00001
        ),
        # تسجيل التدريب
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, 'training_log.csv')
        )
    ]
    
    # تدريب النموذج
    history = model.fit(
        data_augmentation(train_images, training=True),
        train_labels,
        epochs=200,
        batch_size=8,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # حفظ النموذج النهائي
    model.save(os.path.join(model_dir, "final_model.keras"))
    
    # حفظ سجل التدريب
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=4)
    
    return history, model

def evaluate_model(model, test_images, test_labels, label_encoder):
    """
    تقييم أداء النموذج
    """
    # إعادة تشكيل الصور للتنبؤ
    test_images = test_images.reshape(-1, 128, 128, 1)
    
    # التنبؤ على مجموعة الاختبار
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # حساب الدقة
    test_accuracy = accuracy_score(test_labels, predicted_classes)
    
    # حساب التقرير التفصيلي
    classification_rep = classification_report(test_labels, predicted_classes, 
                                               target_names=label_encoder.classes_)
    
    # طباعة نتائج التقييم
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    
    # محاولة طباعة التقرير بشكل آمن
    try:
        print(classification_rep)
    except UnicodeEncodeError:
        print("Unable to print classification report due to encoding issues.")
    
    # رسم مصفوفة الارتباك
    cm = confusion_matrix(test_labels, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig('../data/confusion_matrix.png')
    plt.close()
    
    return test_accuracy, classification_rep

def plot_training_history(history):
    """
    رسم مخطط تاريخ التدريب
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # الدقة
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../models/training_plots.png')
    plt.close()

if __name__ == "__main__":
    # تحميل البيانات
    images, labels, label_encoder = load_data("../data/letters")
    
    # تقسيم البيانات
    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)
    
    # تدريب النموذج
    history, model = train_model(train_images, train_labels, val_images, val_labels, num_classes=len(label_encoder.classes_))
    
    # تقييم النموذج
    test_loss, test_accuracy = evaluate_model(model, test_images, test_labels, label_encoder)
    
    # رسم مخطط التدريب
    plot_training_history(history)
