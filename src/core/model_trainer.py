import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def create_model(num_classes):
    """
    إنشاء نموذج CNN للتعرف على الحروف الثمودية
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(128, 128, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model(model, save_path):
    """
    حفظ النموذج مع التكوين
    """
    try:
        # Save model architecture
        model_json = model.to_json()
        with open(save_path.replace('.h5', '_architecture.json'), 'w') as f:
            f.write(model_json)
            
        # Save weights
        model.save_weights(save_path.replace('.h5', '_weights.h5'))
        
        # Save full model
        tf.keras.models.save_model(
            model,
            save_path,
            overwrite=True,
            include_optimizer=True,
            save_format='h5',
            save_traces=False
        )
        
        logging.info(f"Model saved successfully to {save_path}")
        
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise

def train_model(train_images, train_labels, val_images, val_labels, model_dir, num_classes):
    """
    تدريب النموذج مع حفظ النتائج
    """
    try:
        # Create model
        model = create_model(num_classes)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(model_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        # Train model
        logging.info("Starting model training...")
        logging.info(f"Training model with {len(train_images)} images and {num_classes} classes")
        
        history = model.fit(
            train_images,
            train_labels,
            batch_size=32,
            epochs=100,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        save_model(model, os.path.join(model_dir, 'final_model.h5'))
        
        return model, history
        
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise

def evaluate_model(model, test_images, test_labels, label_mapping):
    """
    تقييم النموذج مع مقاييس متقدمة وتحليل
    """
    # Convert test labels to categorical
    test_labels_cat = tf.keras.utils.to_categorical(test_labels, len(label_mapping))
    
    # Get predictions
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels_cat, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'classification_report': report
    }
    
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

def plot_training_history(history):
    """
    رسم تاريخ التدريب مع تحليلات متقدمة
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    
    # Plot learning rate
    if 'lr' in history.history:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('models/learning_rate.png')
        plt.close()

if __name__ == "__main__":
    # تحميل البيانات
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'letters', 'thamudic_letters')
    mapping_path = os.path.join(base_dir, 'data', 'letters', 'letter_mapping.json')
    model_dir = os.path.join(base_dir, 'models')
    
    print(f"Loading data from: {data_dir}")
    print(f"Using mapping file: {mapping_path}")
    print(f"Model will be saved to: {model_dir}")
    
    # تحميل البيانات وتقسيمها
    from src.data.data_pipeline import load_data, split_data
    images, labels, label_encoder = load_data(data_dir, mapping_path)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    # تدريب النموذج
    train_model(x_train, y_train, x_val, y_val, model_dir=model_dir, num_classes=len(label_encoder.classes_))

    # إنشاء نموذج اختبار
    def create_test_model():
        # إنشاء نموذج اختبار
        model = create_model(num_classes=32)
        
        # التأكد من وجود مجلد النماذج
        os.makedirs('models', exist_ok=True)
        
        # حفظ النموذج
        save_model(model, 'models/best_model.h5')
        
        # إنشاء ملف تعيين التسميات
        os.makedirs('models/configs', exist_ok=True)
        import json
        label_mapping = {str(i): chr(0x0627 + i) for i in range(32)}  # حروف عربية
        with open('models/configs/label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    create_test_model()
