import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from src.data.data_pipeline import load_data, split_data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

def create_model(num_classes):
    """
    إنشاء نموذج شبكة عصبية تلافيفية للتصنيف مع تحسينات
    """
    # Use a more powerful base model
    base_model = EfficientNetB0(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model = models.Sequential([
        # Convert single channel to 3 channels
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(tf.cast(x, tf.float32))),
        
        # Base model
        base_model,
        
        # Custom top layers with stronger regularization
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax',
                    kernel_initializer='he_normal')
    ])
    
    # Use a custom learning rate schedule
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    
    # Compile with improved optimizer settings
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(train_images, train_labels, val_images, val_labels, model_dir="../models", num_classes=None):
    """
    تدريب النموذج مع تحسينات متقدمة
    """
    if num_classes is None:
        num_classes = len(np.unique(train_labels))
        
    print(f"\nTraining model with {num_classes} classes...")
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Data augmentation layer with more aggressive transformations
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])

    # Create and compile the model
    model = create_model(num_classes)
    
    # Calculate class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Convert images to float32 and normalize
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    
    # Add channel dimension if needed
    if len(train_images.shape) == 3:
        train_images = np.expand_dims(train_images, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
    
    # Create tf.data.Dataset objects with larger buffer size
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(16)
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # Training callbacks with better configuration
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model with cosine annealing
    initial_epoch = 0
    epochs_per_cycle = 50
    total_epochs = 200
    
    print("\nStarting training with cosine annealing...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save the final model and training history
    model.save(os.path.join(model_dir, 'final_model.h5'))
    plot_training_history(history)
    
    return model, history

def evaluate_model(model, test_images, test_labels, label_encoder):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: Test labels
        label_encoder: Label encoder used for training
        
    Returns:
        tuple: Test loss and accuracy
    """
    # Reshape test images
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
    
    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes,
                              target_names=[str(i) for i in range(len(label_encoder.classes_))]))
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Create the models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(model_dir, 'training_plots.png'))
    plt.close()

class ThamudicModel(nn.Module):
    def __init__(self, num_classes=32):
        super(ThamudicModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_test_model():
    # إنشاء نموذج اختبار
    model = ThamudicModel()
    
    # التأكد من وجود مجلد النماذج
    os.makedirs('models', exist_ok=True)
    
    # حفظ النموذج
    torch.save(model.state_dict(), 'models/best_model.pth')
    
    # إنشاء ملف تعيين التسميات
    os.makedirs('models/configs', exist_ok=True)
    import json
    label_mapping = {str(i): chr(0x0627 + i) for i in range(32)}  # حروف عربية
    with open('models/configs/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

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
    images, labels, label_encoder = load_data(data_dir, mapping_path)
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(images, labels)
    
    # تدريب النموذج
    train_model(x_train, y_train, x_val, y_val, model_dir=model_dir, num_classes=len(label_encoder.classes_))

    # إنشاء نموذج اختبار
    create_test_model()
