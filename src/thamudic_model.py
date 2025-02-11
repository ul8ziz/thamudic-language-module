"""
نموذج التعرف على الحروف الثمودية مع هيكل CNN مخصص
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import Dict, Tuple

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ResidualBlock(nn.Module):
    """كتلة متبقية للتعلم العميق"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # اتصال مختصر
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class SEBlock(nn.Module):
    """كتلة Squeeze-and-Excitation للانتباه"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ThamudicRecognitionModel(nn.Module):
    """نموذج التعرف على الحروف الثمودية"""
    def __init__(self, num_classes: int, dropout_rate: float = 0.4):
        super(ThamudicRecognitionModel, self).__init__()
        
        # الطبقات التلافيفية
        self.features = nn.Sequential(
            # الكتلة 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # الكتلة 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # الكتلة 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # الكتلة المتبقية
            ResidualBlock(128, 128)
        )
        
        # حساب حجم المخرجات من الطبقات التلافيفية
        # الصورة الأصلية 224x224 -> 112x112 -> 56x56 -> 28x28
        # عدد القنوات النهائي هو 128
        conv_output_size = 128 * 28 * 28
        
        # الطبقات المتصلة
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # تهيئة الأوزان
        self._initialize_weights()
    
    def _initialize_weights(self):
        """تهيئة أوزان النموذج"""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """التقدم الأمامي للنموذج"""
        # التأكد من أن الإدخال أحادي القناة
        if x.dim() == 3:
            x = x.unsqueeze(1)  # إضافة بُعد القناة
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # إضافة بُعد الدفعة والقناة
        
        x = self.features(x)
        x = self.classifier(x)
        return x

class ThamudicRecognitionTrainer:
    """مدرب نموذج التعرف على الحروف الثمودية"""
    
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # استخدام معدل تعلم مخصص
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # استخدام جدولة التعلم التجيبية
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,  # إعادة التشغيل كل 10 حقب
            eta_min=learning_rate * 0.01  # الحد الأدنى هو 1% من معدل التعلم الأصلي
        )

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """تدريب حقبة واحدة"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            # إضافة تقليم التدرج
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # حساب الدقة
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        # تحديث معدل التعلم
        self.scheduler.step()
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """تقييم النموذج"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # حساب الدقة
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def save_checkpoint(self, path: str):
        """حفظ نقطة تفتيش للنموذج"""
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        logging.info(f"تم حفظ نقطة التفتيش في {path}")
    
    def load_checkpoint(self, path: str):
        """تحميل نقطة تفتيش للنموذج"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"تم تحميل نقطة التفتيش من {path}")
