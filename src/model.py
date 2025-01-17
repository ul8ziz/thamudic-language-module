import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import os
import logging
import json

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=1),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ThamudicFeatureExtractor(nn.Module):
    """
    مستخرج السمات المخصص للحروف الثمودية
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # طبقات استخراج السمات المحلية
        self.local_features = nn.Sequential(
            # طبقة 1: استخراج التفاصيل الدقيقة
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=8),
            
            # طبقة 2: تجميع السمات المحلية
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=16),
        )
        
        # طبقات استخراج السمات العالمية
        self.global_features = nn.Sequential(
            # طبقة 3: استخراج السمات العالية المستوى
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
        )
        
        # آلية الانتباه للتركيز على الأجزاء المهمة
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # استخراج السمات المحلية
        local_feats = self.local_features(x)
        
        # استخراج السمات العالمية
        global_feats = self.global_features(local_feats)
        
        # تطبيق الانتباه
        attention_weights = self.attention(global_feats)
        attended_feats = global_feats * attention_weights
        
        return attended_feats, attention_weights

class ThamudicRecognitionModel(nn.Module):
    def __init__(self, num_classes=58):
        super().__init__()
        
        # مستخرج السمات المخصص
        self.feature_extractor = ThamudicFeatureExtractor(in_channels=1)
        
        # طبقات التصنيف
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # مؤشر الثقة
        self.confidence = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # تهيئة الأوزان
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # استخراج السمات
        features, attention = self.feature_extractor(x)
        
        # التصنيف
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        flattened = pooled.view(pooled.size(0), -1)
        
        logits = self.classifier(flattened)
        confidence = self.confidence(flattened)
        
        return logits, confidence, attention

class ModelWithAttention(nn.Module):
    """Thamudic Character Recognition Model with Attention Mechanism.
    Designed for small datasets with transfer learning.
    """
    def __init__(self, num_classes=29, input_channels=1):
        super().__init__()
        
        # استخدام ResNet18 مع الأوزان المدربة مسبقًا
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # تعديل الطبقة الأولى للتعامل مع الصور الرمادية
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # تجميد الطبقات الأساسية مع السماح بالتحديث التدريجي
        for param in list(self.backbone.parameters())[:-4]:  # تجميد كل الطبقات ما عدا آخر بلوكين
            param.requires_grad = False
        
        # إزالة الطبقة النهائية
        backbone_layers = list(self.backbone.children())[:-2]
        self.backbone = nn.Sequential(*backbone_layers)
        
        # طبقة الانتباه المتقدمة
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # طبقات التصنيف مع تقنيات منع الإفراط
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # زيادة معدل الإسقاط
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # استخراج السمات
        features = self.backbone(x)
        
        # حساب خريطة الانتباه
        attention_weights = self.attention(features)
        
        # تطبيق الانتباه
        attended_features = features * attention_weights
        
        # التصنيف
        output = self.classifier(attended_features)
        
        return output, attention_weights

class ThamudicTranslator:
    def __init__(self, model_path: str, char_mapping: Dict[int, str]):
        """
        Initialize the Thamudic Translator with a pre-trained model.
        
        Args:
            model_path (str): Path to the pre-trained model weights
            char_mapping (Dict[int, str]): Mapping of class indices to characters
        """
        # تحديد عدد الفئات من مجموعة التعيينات
        self.char_mapping = {}
        self.reverse_mapping = {}
        
        # التحقق من نوع char_mapping
        if isinstance(char_mapping, dict):
            if 'class_mapping' in char_mapping:
                # إذا كان يحتوي على class_mapping، نستخدم التعيين المباشر
                self.char_mapping = {v: k for k, v in char_mapping['class_mapping'].items()}
                self.reverse_mapping = char_mapping['class_mapping']
            elif 'thamudic_to_arabic' in char_mapping:
                # إذا كان يحتوي على thamudic_to_arabic، نستخدم التعيين العكسي
                thamudic_to_arabic = char_mapping['thamudic_to_arabic']
                arabic_to_class = char_mapping['class_mapping']
                self.char_mapping = {arabic_to_class[arabic]: thamudic for thamudic, arabic in thamudic_to_arabic.items()}
                self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}
            else:
                # إذا كان تعيين مباشر
                self.char_mapping = char_mapping
                self.reverse_mapping = {v: k for k, v in char_mapping.items()}
        else:
            raise ValueError("char_mapping must be a dictionary")
        
        num_classes = len(self.char_mapping)
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Character mapping: {self.char_mapping}")
        logging.info(f"Reverse mapping: {self.reverse_mapping}")
        
        # إنشاء النموذج
        self.model = create_model(num_classes=num_classes)
        
        # تحميل الأوزان
        if os.path.exists(model_path):
            try:
                # تحميل الأوزان على CPU
                state_dict = torch.load(model_path, map_location='cpu')
                
                # إذا كان state_dict يحتوي على 'model_state_dict'
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # تحميل الأوزان مع تجاهل المفاتيح غير المتطابقة
                model_dict = self.model.state_dict()
                # طباعة المفاتيح المتوفرة
                logging.info(f"Available keys in state_dict: {state_dict.keys()}")
                logging.info(f"Required keys in model: {model_dict.keys()}")
                
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(state_dict)
                
                # تحميل الأوزان
                self.model.load_state_dict(model_dict, strict=False)
                logging.info("Model weights loaded successfully from %s", model_path)
            except Exception as e:
                logging.error("خطأ في تحميل الأوزان: %s", str(e), exc_info=True)
                raise
        else:
            logging.error("ملف الأوزان غير موجود: %s", model_path)
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        # تعيين وضع التقييم
        self.model.eval()
        
        # إعداد التحويلات
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_char_mapping(self):
        """
        تحميل تعيين الحروف من الملف
        """
        try:
            with open(os.path.join(self.base_dir, 'data', 'char_mapping.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # إنشاء تعيين مباشر من الفئات إلى الحروف الثمودية
                self.char_mapping = {}
                for idx, (thamudic, arabic) in enumerate(data['thamudic_to_arabic'].items()):
                    self.char_mapping[idx] = thamudic
                logging.info(f"تم تحميل {len(self.char_mapping)} حرف")
                logging.info(f"التعيين: {self.char_mapping}")
        except Exception as e:
            logging.error(f"خطأ في تحميل تعيين الحروف: {e}")
            self.char_mapping = {}
    
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        التنبؤ بالأحرف الثمودية في الصورة المدخلة
        
        Args:
            image_tensor (torch.Tensor): تانسور الصورة المدخلة
            top_k (int, optional): عدد التنبؤات الأعلى المراد إرجاعها. الافتراضي 5.
        
        Returns:
            List[Tuple[str, float]]: قائمة بالأحرف المتنبأ بها مع درجة الثقة
        """
        logging.info(f"Input tensor shape: {image_tensor.shape}")
        
        # التأكد من أن التانسور بالأبعاد الصحيحة
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # تطبيق تقنيات تحسين الإدخال
        image_tensor = torch.nn.functional.normalize(image_tensor, p=2, dim=1)
        
        # التنبؤ
        with torch.no_grad():
            try:
                # تشغيل النموذج مع تقنية التجميع
                outputs = []
                transforms_list = [
                    lambda x: x,  # الصورة الأصلية
                    lambda x: torch.flip(x, [2]),  # قلب أفقي
                    lambda x: torch.flip(x, [3]),  # قلب رأسي
                    lambda x: torch.rot90(x, 1, [2, 3]),  # تدوير 90 درجة
                    lambda x: torch.rot90(x, -1, [2, 3])  # تدوير -90 درجة
                ]
                
                for transform in transforms_list:
                    transformed_input = transform(image_tensor)
                    output = self.model(transformed_input)
                    outputs.append(output)
                
                # تجميع النتائج
                outputs = torch.stack(outputs).mean(0)
                
                # تطبيق softmax للحصول على احتمالات
                probabilities = F.softmax(outputs, dim=1)
                
                # الحصول على أعلى top_k احتمالات
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.char_mapping)))
                
                # تحويل النتائج
                predictions = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    idx = idx.item()
                    if idx in self.char_mapping:
                        char = self.char_mapping[idx]
                        confidence = float(prob.item())
                        logging.info(f"Index {idx} -> Character '{char}' with confidence {confidence:.4f}")
                        
                        # استخدام عتبة ثقة منخفضة جداً
                        if confidence > 0.0001:  # 0.01%
                            predictions.append((char, confidence))
                
                # إذا لم يتم العثور على تطابقات، إرجاع أعلى احتمال
                if not predictions and len(top_indices[0]) > 0:
                    idx = top_indices[0][0].item()
                    if idx in self.char_mapping:
                        char = self.char_mapping[idx]
                        confidence = float(top_probs[0][0].item())
                        logging.info(f"Returning top prediction: Character '{char}' with confidence {confidence:.4f}")
                        predictions.append((char, confidence))
                
                return sorted(predictions, key=lambda x: x[1], reverse=True)
                
            except Exception as e:
                logging.error(f"خطأ في التنبؤ: {str(e)}", exc_info=True)
                return []
    
    def translate(self, image_path: str) -> str:
        """
        Translate the Thamudic text in the given image to Arabic.
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            str: Translated Arabic text
        """
        # التنبؤ بالأحرف الثمودية
        thamudic_text = self.predict(image_path)
        logging.info(f"Predicted Thamudic text: {thamudic_text}")
        
        # تحويل النص الثمودي إلى العربية
        arabic_text = []
        for char, confidence in thamudic_text:
            if char in self.reverse_mapping:
                arabic_char = self.reverse_mapping[char]
                arabic_text.append(arabic_char)
                logging.info(f"Translated '{char}' to '{arabic_char}'")
            else:
                logging.warning(f"Character '{char}' not found in reverse mapping")
        
        translated_text = ''.join(arabic_text)
        logging.info(f"Translated Arabic text: {translated_text}")
        return translated_text

class ImagePreprocessor:
    """معالج الصور المتقدم للحروف الثمودية"""
    
    def __init__(self, target_size=(128, 128)):  # تقليل حجم الصورة للحفاظ على التفاصيل
        self.target_size = target_size
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """إنشاء سلسلة التحويلات المتقدمة"""
        return transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),  # تحويل إلى رمادي
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # تطبيع محايد
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # تحسين حدة الحروف
        ])
    
    def preprocess(self, image):
        """
        معالجة الصورة وتحسين جودتها
        
        Args:
            image (PIL.Image): الصورة المدخلة
            
        Returns:
            torch.Tensor: الصورة المعالجة
        """
        # تحسين جودة الصورة
        enhanced_image = self._enhance_image(image)
        
        # تطبيق التحويلات
        tensor = self.transform(enhanced_image)
        
        # إضافة قناة إضافية للتفاصيل الدقيقة
        edge_tensor = self._extract_edges(tensor)
        
        return tensor
    
    def _enhance_image(self, image):
        """
        تحسين جودة الصورة
        
        Args:
            image (PIL.Image): الصورة المدخلة
            
        Returns:
            PIL.Image: الصورة المحسنة
        """
        from PIL import ImageEnhance
        
        # تحسين التباين
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # تحسين الحدة
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # زيادة حدة الحروف
        
        return image
    
    def _extract_edges(self, tensor):
        """
        استخراج حواف الحروف
        
        Args:
            tensor (torch.Tensor): تنسور الصورة
            
        Returns:
            torch.Tensor: تنسور الحواف
        """
        import torch.nn.functional as F
        
        # تطبيق فلتر سوبل للكشف عن الحواف
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # تطبيق الفلاتر
        edges_x = F.conv2d(tensor.unsqueeze(0), sobel_x, padding=1)
        edges_y = F.conv2d(tensor.unsqueeze(0), sobel_y, padding=1)
        
        # حساب شدة الحواف
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        
        return edges.squeeze(0)

class AttentionBlock(nn.Module):
    """
    كتلة الانتباه للتركيز على المناطق المهمة في الصورة
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # طبقة التخفيض
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # المتوسط والحد الأقصى للتجميع
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        
        # حساب أوزان الانتباه
        avg_attention = self.mlp(avg_pool)
        max_attention = self.mlp(max_pool)
        
        attention = self.sigmoid(avg_attention + max_attention)
        attention = attention.view(x.size(0), x.size(1), 1, 1)
        
        return x * attention

class ThamudicModel(nn.Module):
    """
    نموذج التعرف على الحروف الثمودية
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        # استخدام ResNet18 كنموذج أساسي
        self.backbone = resnet18(pretrained=pretrained)
        
        # تعديل الطبقة الأولى للتعامل مع الصور الرمادية
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # إضافة كتل الانتباه
        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(256)
        self.attention4 = AttentionBlock(512)
        
        # تعديل الطبقة النهائية
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
        # التهيئة
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        تهيئة الأوزان
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        التمرير الأمامي
        
        Args:
            x (torch.Tensor): دفعة من الصور
            
        Returns:
            tuple: (التصنيفات، أوزان الانتباه)
        """
        logging.info(f"Input shape: {x.shape}")
        
        try:
            # تمرير الصورة عبر النموذج الأساسي
            x = self.backbone.conv1(x)
            logging.info(f"After conv1: {x.shape}")
            
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            logging.info(f"After initial layers: {x.shape}")
            
            # تطبيق طبقات ResNet مع الانتباه
            x = self.backbone.layer1(x)
            x = self.attention1(x)
            logging.info(f"After layer1: {x.shape}")
            
            x = self.backbone.layer2(x)
            x = self.attention2(x)
            logging.info(f"After layer2: {x.shape}")
            
            x = self.backbone.layer3(x)
            x = self.attention3(x)
            logging.info(f"After layer3: {x.shape}")
            
            x = self.backbone.layer4(x)
            x = self.attention4(x)
            logging.info(f"After layer4: {x.shape}")
            
            # التجميع العالمي وتسطيح البيانات
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            logging.info(f"After flatten: {x.shape}")
            
            # التصنيف النهائي
            x = self.backbone.fc(x)
            logging.info(f"Final output: {x.shape}")
            
            return x
            
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise

def create_model(num_classes, pretrained=True):
    """
    إنشاء نموذج جديد
    
    Args:
        num_classes (int): عدد الفئات
        pretrained (bool): استخدام أوزان مدربة مسبقاً
        
    Returns:
        ThamudicModel: النموذج المنشأ
    """
    logging.info(f"Creating model with {num_classes} classes")
    
    try:
        # إنشاء النموذج الأساسي
        model = ThamudicModel(num_classes=num_classes, pretrained=pretrained)
        logging.info("Model created successfully")
        
        # طباعة ملخص النموذج
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # تأكد من أن النموذج في وضع التقييم
        model.eval()
        
        # تحقق من الأبعاد
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 224, 224)  # دفعة واحدة، قناة واحدة
            try:
                # تشغيل النموذج مع تقنية التجميع
                output = model(dummy_input)
                logging.info(f"Test forward pass successful. Output shape: {output.shape}")
            except Exception as e:
                logging.error(f"Error in test forward pass: {str(e)}")
                raise
        
        return model
    except Exception as e:
        logging.error(f"Error creating model: {str(e)}", exc_info=True)
        raise

def save_model_weights(model, path='models/thamudic_model.pth'):
    """
    حفظ أوزان النموذج مع معلومات إضافية
    
    Args:
        model (nn.Module): النموذج المراد حفظ أوزانه
        path (str): مسار حفظ الأوزان
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': model.backbone.fc[-1].out_features,
        'input_channels': model.backbone.conv1.in_channels
    }, path)

def load_model_weights(model_type='resnet', path='models/thamudic_model.pth'):
    """
    تحميل أوزان النموذج مع التأكد من التوافق
    
    Args:
        model_type (str): نوع النموذج
        path (str): مسار ملف الأوزان
    
    Returns:
        nn.Module: النموذج مع الأوزان المحملة
    """
    # تحميل المعلومات
    checkpoint = torch.load(path)
    
    # إنشاء نموذج متوافق
    model = create_model(
        model_type=model_type, 
        num_classes=checkpoint['num_classes'],
        pretrained=checkpoint['input_channels']
    )
    
    # تحميل الأوزان
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def print_model_summary(model, input_size=(3, 224, 224), device='cpu'):
    """
    Print a summary of the model architecture.
    
    Args:
        model (nn.Module): Neural network model
        input_size (tuple): Size of input tensor
        device (str or torch.device): Device to move the model to
    """
    model = model.to(device)
    
    # Prepare input tensor
    input_tensor = torch.randn(16, *input_size).to(device)
    
    # Model summary details
    summary_str = f"Model Architecture Summary:\n"
    summary_str += f"   Model Name: {model.__class__.__name__}\n"
    summary_str += f"   Input Size: {input_size}\n"
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary_str += f"   Total Parameters: {total_params:,}\n"
    summary_str += f"   Trainable Parameters: {trainable_params:,}\n"
    
    # Estimate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 ** 2
    summary_str += f"   Model Size: {model_size_mb:.2f} MB\n"
    
    # Compute FLOPs and parameters
    try:
        from torchprofile import profile_macs
        macs = profile_macs(model, input_tensor)
        summary_str += f"   Multiply-Accumulate Operations (MACs): {macs:,}\n"
    except ImportError:
        summary_str += "   Note: Install 'torchprofile' to get MACs estimation\n"
    
    # Print summary
    print(summary_str)
    
    # Optional: Visualize model architecture
    try:
        from torchsummary import summary
        print("\nDetailed Layer Summary:")
        summary(model, input_size)
    except ImportError:
        print("   Note: Install 'torchsummary' for detailed layer summary")
