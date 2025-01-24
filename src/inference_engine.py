import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
from tensorflow.keras.mixed_precision import Policy
import datetime
import albumentations as A
import logging
import os
import argparse
from pathlib import Path

class ThamudicInferenceEngine:
    """
    محرك استدلال متقدم للتعرف على النصوص الثمودية
    Advanced inference engine for Thamudic text recognition
    """
    
    def __init__(self, model_path, mapping_path, config=None):
        """
        Initialize the inference engine with advanced configuration
        
        Args:
            model_path: Path to the trained model
            mapping_path: Path to the label mapping
            config: Optional configuration dictionary
        """
        self.config = config or {
            'confidence_threshold': 0.7,
            'top_k': 3,
            'preprocessing': {
                'target_size': (128, 128),
                'normalize': True,
                'augment': False
            }
        }
        
        # Load model and mapping
        self.model = self._load_model(model_path)
        self.label_mapping = self._load_mapping(mapping_path)
        
        # Initialize preprocessing pipeline
        self.preprocessor = ImagePreprocessor(self.config['preprocessing'])
    
    def _load_model(self, model_path):
        """Load the TensorFlow model"""
        try:
            # Try loading SavedModel format
            model = tf.keras.models.load_model(model_path)
        except:
            # Try loading from architecture and weights
            try:
                with open(f"{model_path}_architecture.json", 'r') as f:
                    model_json = f.read()
                    model = tf.keras.models.model_from_json(model_json)
                    model.load_weights(f"{model_path}_weights.h5")
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")
        
        return model
    
    def _load_mapping(self, mapping_path):
        """Load the label mapping"""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            return mapping_data['thamudic_letters']
    
    def predict(self, image):
        """
        Predict Thamudic text in the image
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            List of predictions with confidence scores
        """
        # Preprocess image
        processed_image = self.preprocessor.process(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get top k predictions
        top_k = min(self.config['top_k'], len(self.label_mapping))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            if confidence >= self.config['confidence_threshold']:
                letter_info = self.label_mapping[str(idx)]
                results.append({
                    'symbol': letter_info['symbol'],
                    'name': letter_info['name'],
                    'confidence': confidence
                })
        
        return results

class ImagePreprocessor:
    """Advanced image preprocessing pipeline"""
    
    def __init__(self, config):
        self.config = config
        
        # Create augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.3),
        ])
    
    def process(self, image):
        """
        Process image with advanced techniques
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Processed image tensor
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize
        image = cv2.resize(image, self.config['target_size'])
        
        # Apply augmentation if enabled
        if self.config['augment']:
            image = self.augmentation(image=image)['image']
        
        # Normalize if enabled
        if self.config['normalize']:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image

class InferenceEngine:
    def __init__(self, model_dir):
        """
        تهيئة محرك الاستدلال
        """
        try:
            self.model_dir = model_dir
            self.model = None
            self.label_mapping = None
            self.load_model()
            
        except Exception as e:
            logging.error(f"Error initializing inference engine: {str(e)}")
            raise
            
    def load_model(self):
        """
        تحميل النموذج وخريطة التصنيفات
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'best_model.h5')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, 'final_model.h5')
                
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found in {self.model_dir}")
                
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"Loaded model from {model_path}")
            
            # Load label mapping
            mapping_path = os.path.join(self.model_dir, 'label_mapping.json')
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Label mapping not found at {mapping_path}")
                
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)['thamudic_letters']
                
            logging.info("Model and label mapping loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
    def preprocess_image(self, image_path):
        """
        معالجة الصورة قبل التصنيف
        """
        try:
            # Read and resize image
            img = tf.keras.preprocessing.image.load_img(
                image_path,
                target_size=(128, 128)
            )
            
            # Convert to array and add batch dimension
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Normalize
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise
            
    def predict(self, image_path, top_k=3):
        """
        التنبؤ بالحرف الثمودي من الصورة
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Get predictions
            predictions = self.model.predict(img_array)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                letter = self.label_mapping[str(idx)]
                confidence = float(predictions[0][idx])
                results.append({
                    'letter': letter,
                    'confidence': confidence,
                    'index': int(idx)
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise
            
    def predict_batch(self, image_paths, batch_size=32):
        """
        التنبؤ بمجموعة من الصور دفعة واحدة
        """
        try:
            # Preprocess images
            images = []
            for path in image_paths:
                img_array = self.preprocess_image(path)
                images.append(img_array[0])
                
            images = np.array(images)
            
            # Get predictions
            predictions = self.model.predict(images, batch_size=batch_size)
            
            # Process results
            results = []
            for i, pred in enumerate(predictions):
                top_indices = np.argsort(pred)[-3:][::-1]
                image_results = []
                
                for idx in top_indices:
                    letter = self.label_mapping[str(idx)]
                    confidence = float(pred[idx])
                    image_results.append({
                        'letter': letter,
                        'confidence': confidence,
                        'index': int(idx)
                    })
                    
                results.append({
                    'image_path': image_paths[i],
                    'predictions': image_results
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Error during batch prediction: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Predict Thamudic inscriptions')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model and label mapping')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    if not args.input:
        raise ValueError('Input path must be provided and cannot be None or empty.')
    
    predictor = InferenceEngine(args.model_dir)
    
    if Path(args.input).is_file():
        if not args.input:
            raise ValueError('Input path must be provided and cannot be None or empty.')
        # Single image
        predictions = predictor.predict(args.input)
        print("Predictions:", predictions)
    else:
        # Directory of images
        results = []
        for img_path in Path(args.input).glob('**/*.jpg'):
            try:
                predictions = predictor.predict(str(img_path))
                results.append({
                    'image_path': str(img_path),
                    'predictions': predictions
                })
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # Save results
        with open(Path(args.output_dir) / 'predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(results)} images. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()
