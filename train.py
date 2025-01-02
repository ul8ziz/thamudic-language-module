import os
import logging
import numpy as np
import tensorflow as tf
from src.core.model_trainer import create_model, save_model
from src.core.data_loader import load_and_preprocess_data
from src.utils.visualization import plot_training_history

def main():
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data', 'letters', 'thamudic_letters')
        model_dir = os.path.join(base_dir, 'models')
        mapping_file = os.path.join(base_dir, 'data', 'letters', 'letter_mapping.json')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load and preprocess data
        (train_images, train_labels), (val_images, val_labels), num_classes = load_and_preprocess_data(
            data_dir,
            mapping_file
        )
        
        # Create model
        model = create_model(num_classes)
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model_weights.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
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
        
        # Train the model
        logging.info("\nStarting model training...")
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
        
        # Save best model
        save_model(model, os.path.join(model_dir, 'best_model.h5'))
        
        # Plot training history
        plot_training_history(history)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()