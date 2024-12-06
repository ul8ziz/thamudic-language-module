import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from data_preprocessing import load_data, split_data

def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(train_images, train_labels, val_images, val_labels):
    # تحديد عدد الفئات
    num_classes = len(np.unique(train_labels))
    
    # إنشاء النموذج
    model = create_model(num_classes)
    
    # إيقاف مبكر للتدريب
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    # تدريب النموذج
    history = model.fit(
        train_images, train_labels, 
        epochs=50,  # زيادة عدد الدورات
        batch_size=32,
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping]
    )
    
    # حفظ النموذج
    model.save("models/thamudic_model.h5")
    return history

if __name__ == "__main__":
    # تحميل البيانات
    images, labels = load_data("data/processed")
    
    # تقسيم البيانات
    train_images, train_labels, val_images, val_labels, _, _ = split_data(images, labels)
    
    # تدريب النموذج
    train_model(train_images, train_labels, val_images, val_labels)
