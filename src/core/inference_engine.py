import torch
from PIL import Image
import numpy as np
from src.data.image_augmentation import ThamudicImagePreprocessor
from src.core.model_trainer import ThamudicModel
import json
import argparse
from pathlib import Path

class InferenceEngine:
    def __init__(self, model_path, label_mapping_path):
        """تهيئة محرك التنبؤ"""
        # تهيئة النموذج
        self.model = ThamudicModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        self.preprocessor = ThamudicImagePreprocessor()
        
        # تحميل تعيينات التسميات
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        # تعيين الرموز الثمودية
        self.thamudic_symbols = {
            'ا': 'ᚠ', 'ب': 'ᚢ', 'ت': 'ᚦ', 'ث': 'ᚨ',
            'ج': 'ᚱ', 'ح': 'ᚳ', 'خ': 'ᚷ', 'د': 'ᚹ',
            'ذ': 'ᚻ', 'ر': 'ᚾ', 'ز': 'ᛁ', 'س': 'ᛂ',
            'ش': 'ᛃ', 'ص': 'ᛄ', 'ض': 'ᛅ', 'ط': 'ᛆ',
            'ظ': 'ᛇ', 'ع': 'ᛈ', 'غ': 'ᛉ', 'ف': 'ᛊ',
            'ق': 'ᛋ', 'ك': 'ᛌ', 'ل': 'ᛍ', 'م': 'ᛎ',
            'ن': 'ᛏ', 'ه': 'ᛐ', 'و': 'ᛑ', 'ي': 'ᛒ',
            'ء': 'ᛓ', 'ئ': 'ᛔ', 'ؤ': 'ᛕ', 'ة': 'ᛖ',
            'ى': 'ᛗ'
        }

    def predict(self, image_path):
        """التنبؤ بالحروف في الصورة"""
        try:
            # تحميل وتحسين الصورة
            image = Image.open(image_path)
            enhanced_image = self.preprocessor.enhance_image(image)
            
            # تحويل الصورة إلى تنسور
            image_tensor = self.preprocessor.preprocess_image(enhanced_image)
            
            # إضافة بُعد الدفعة
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # التنبؤ
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # الحصول على أعلى التنبؤات
                confidences, predictions = torch.max(probabilities, 1)
                
                results = []
                for pred, conf in zip(predictions, confidences):
                    label_idx = pred.item()
                    if str(label_idx) in self.label_mapping:
                        letter = self.label_mapping[str(label_idx)]
                        results.append({
                            'letter': letter,
                            'confidence': conf.item()
                        })
                
                return results
                
        except Exception as e:
            print(f"خطأ في التنبؤ: {str(e)}")
            return []

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
    
    predictor = InferenceEngine(args.model_path, args.label_mapping)
    
    if Path(args.input).is_file():
        if not args.input:
            raise ValueError('Input path must be provided and cannot be None or empty.')
        # Single image
        predictions = predictor.predict(args.input)
        print("Predictions:", ''.join(char['letter'] for char in predictions))
        print("Confidences:", [f"{conf:.2f}" for char in predictions for conf in [char['confidence']]])
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
