import os
import tensorflow as tf
from src.data.data_pipeline import load_data, split_data
from src.core.model_trainer import train_model
import json

def main():
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("No GPU devices found. Using CPU.")

    # Configure paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    letter_mapping_file = os.path.join(base_dir, 'models', 'configs', 'letter_mapping.json')
    label_mapping_file = os.path.join(base_dir, 'models', 'configs', 'label_mapping.json')

    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    images, labels, label_names = load_data(
        data_dir=data_dir,
        letter_mapping_file=letter_mapping_file,
        label_mapping_file=label_mapping_file
    )

    print("\nSplitting data...")
    train_images, val_images, test_images, train_labels, val_labels, test_labels = split_data(
        images, labels, test_size=0.15, val_size=0.15
    )

    print("\nTraining model...")
    model, history = train_model(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        model_dir=models_dir,
        num_classes=len(set(labels))
    )

    # Save training history
    history_file = os.path.join(models_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        json.dump(history_dict, f, indent=4)

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
