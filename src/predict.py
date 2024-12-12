import torch
from model import ThamudicRecognitionModel
from image_preprocessing import ThamudicImagePreprocessor
from PIL import Image
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict

class ThamudicPredictor:
    def __init__(self, model_path: str, label_mapping_path: str):
        # Load label mapping
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        # Create reverse mapping
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ThamudicRecognitionModel(num_classes=len(self.label_mapping))
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ThamudicImagePreprocessor()
    
    def predict_single_character(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict a single Thamudic character from an image
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_idx = int(torch.argmax(output, dim=1).item())
            confidence = probabilities[0][pred_idx].item()
        
        predicted_char = self.idx_to_label[pred_idx]
        return predicted_char, confidence
    
    def predict_inscription(self, image_path: str, visualization_path: str = None) -> List[Tuple[str, float]]:
        """
        Predict all characters in a Thamudic inscription
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Enhance inscription
        enhanced_image = self.preprocessor.enhance_inscription(image)
        
        # Segment characters
        character_images = self.preprocessor.segment_inscription(enhanced_image)
        
        predictions = []
        visualization_image = image.copy()
        
        for i, char_img in enumerate(character_images):
            char, confidence = self.predict_single_character(char_img)
            predictions.append((char, confidence))
            
            # Add visualization
            if visualization_path:
                x = i * (image.shape[1] // len(character_images))
                y = image.shape[0] - 30
                cv2.putText(
                    visualization_image,
                    f"{char} ({confidence:.2f})",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        if visualization_path:
            cv2.imwrite(visualization_path, visualization_image)
        
        return predictions
    
    def predict_batch(self, image_dir: str, output_dir: str):
        """
        Process multiple inscriptions in a directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for img_path in Path(image_dir).glob('**/*.jpg'):
            try:
                predictions = self.predict_inscription(
                    str(img_path),
                    str(output_path / f"{img_path.stem}_annotated.jpg")
                )
                
                results[img_path.name] = {
                    'text': ''.join(char for char, _ in predictions),
                    'confidences': [conf for _, conf in predictions]
                }
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # Save results
        with open(output_path / 'predictions.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict Thamudic inscriptions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--label_mapping', type=str, required=True,
                        help='Path to label mapping file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    if not args.input:
        raise ValueError('Input path must be provided and cannot be None or empty.')
    
    predictor = ThamudicPredictor(args.model_path, args.label_mapping)
    
    if Path(args.input).is_file():
        if not args.input:
            raise ValueError('Input path must be provided and cannot be None or empty.')
        # Single image
        predictions = predictor.predict_inscription(
            args.input,
            str(Path(args.output_dir) / 'output_annotated.jpg')
        )
        print("Predictions:", ''.join(char for char, _ in predictions))
        print("Confidences:", [f"{conf:.2f}" for _, conf in predictions])
    else:
        # Directory of images
        results = predictor.predict_batch(args.input, args.output_dir)
        print(f"Processed {len(results)} images. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()
