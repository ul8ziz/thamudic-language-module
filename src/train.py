#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from datetime import datetime
from data import ThamudicDataset
from model import create_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Custom loss function for label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    """
    دالة خسارة تنعيم التسميات
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def check_cuda_availability():
    """
    Check and print detailed information about CUDA and GPU availability.
    
    Returns:
        torch.device: Recommended device for training
    """
    print("Checking GPU Availability...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        # Get GPU details
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available! Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"   GPU {i}: {gpu_name}")
            print(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
        
        # Select the first GPU
        device = torch.device('cuda:0')
        
        # Check CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        return device
    else:
        print("CUDA is NOT available. Falling back to CPU.")
        return torch.device('cpu')

def load_config(config_path='d:/ul8ziz/GitHub/thamudic-language-module/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def get_train_transforms():
    """
    Generate comprehensive data augmentation transforms for training.
    
    Returns:
        torchvision.transforms.Compose: Augmentation transforms
    """
    return transforms.Compose([
        transforms.RandomRotation(20),  # تدوير بزاوية ±20 درجة
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2),  # انزياح أفقي ورأسي بنسبة 20%
            scale=(0.8, 1.2),  # تغيير الحجم بين 80% و120%
            shear=10  # قص بزاوية ±10 درجة
        ),
        transforms.ColorJitter(
            brightness=0.2,  # تغيير السطوع
            contrast=0.2,    # تغيير التباين
            saturation=0.2,  # تغيير التشبع
            hue=0.1          # تغيير التدرج اللوني
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # قلب أفقي بنسبة 50%
        transforms.RandomVerticalFlip(p=0.2),    # قلب رأسي بنسبة 20%
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # قيم معيارية من ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms():
    """
    Generate validation transforms with minimal augmentation.
    
    Returns:
        torchvision.transforms.Compose: Validation transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def custom_collate(batch):
    """
    Custom collate function to handle variable-sized images
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Resize images to a fixed size
    images = [transforms.Resize((224, 224))(img) for img in images]
    
    # Stack images and labels
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    
    return images, labels

def create_data_loaders(config, char_mapping, device):
    """
    إنشاء محملات البيانات للتدريب والتحقق
    
    Args:
        config (dict): إعدادات التدريب
        char_mapping (dict): خريطة الأحرف
        device (torch.device): جهاز التدريب
        
    Returns:
        tuple: محملات بيانات التدريب والتحقق
    """
    # إنشاء مجموعة بيانات التدريب
    train_dataset = ThamudicDataset(
        data_dir=config['data']['train_path'],
        char_mapping=char_mapping,
        augmentation_prob=config['data']['augmentation']['probability']
    )
    
    # إنشاء مجموعة بيانات التحقق
    val_dataset = ThamudicDataset(
        data_dir=config['data']['val_path'],
        char_mapping=char_mapping,
        augmentation_prob=0.0
    )
    
    # إنشاء محملات البيانات
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=device.type == 'cuda'
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """
    تدريب النموذج
    
    Args:
        model: النموذج المراد تدريبه
        train_loader: محمل بيانات التدريب
        val_loader: محمل بيانات التحقق
        num_epochs: عدد الدورات
        device: الجهاز
    
    Returns:
        النموذج المدرب
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # وضع التدريب
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.*train_correct/train_total:.2f}%')
        
        # وضع التقييم
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # جدولة معدل التعلم
        scheduler.step(val_acc)
        
        # التوقف المبكر
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # حفظ أفضل نموذج
            save_model_weights(model, 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    return model

def validate_model(model, val_loader, criterion, device):
    """
    التحقق من أداء النموذج
    
    Args:
        model (nn.Module): النموذج للتحقق
        val_loader (DataLoader): محمل بيانات التحقق
        criterion (nn.Module): دالة الخسارة
        device (torch.device): الجهاز
    
    Returns:
        tuple: متوسط الخسارة ودقة التحقق
    """
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct_predictions / total_predictions
    
    return val_loss, val_accuracy

def plot_learning_curves(training_results):
    """
    رسم منحنيات التعلم
    
    Args:
        training_results (dict): نتائج التدريب
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # رسم الخسائر
    plt.subplot(1, 2, 1)
    plt.plot(training_results['train_losses'], label='خسارة التدريب')
    plt.plot(training_results['val_losses'], label='خسارة التحقق')
    plt.title('منحنى الخسائر')
    plt.xlabel('الحقبة')
    plt.ylabel('الخسارة')
    plt.legend()
    
    # رسم الدقة
    plt.subplot(1, 2, 2)
    plt.plot(training_results['train_accuracies'], label='دقة التدريب')
    plt.plot(training_results['val_accuracies'], label='دقة التحقق')
    plt.title('منحنى الدقة')
    plt.xlabel('الحقبة')
    plt.ylabel('الدقة')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def save_model_weights(model, path):
    """
    حفظ أوزان النموذج
    
    Args:
        model (nn.Module): النموذج
        path (str): مسار حفظ الأوزان
    """
    torch.save(model.state_dict(), path)

class ThamudicDataAugmentation:
    """
    فئة متخصصة في زيادة البيانات للأحرف الثمودية
    """
    @staticmethod
    def rotate_image(image, angle_range=(-15, 15)):
        """
        تدوير الصورة بزاوية عشوائية
        
        Args:
            image (PIL.Image or np.ndarray): الصورة المراد تدويرها
            angle_range (tuple): نطاق الزوايا للتدوير
        
        Returns:
            PIL.Image: الصورة المدورة
        """
        # التأكد من أن الصورة بتنسيق PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # اختيار زاوية عشوائية
        angle = random.uniform(angle_range[0], angle_range[1])
        
        # تدوير الصورة
        rotated_image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        
        return rotated_image
    
    @staticmethod
    def add_noise(image, noise_type='gaussian', intensity=0.02):
        """
        إضافة ضوضاء للصورة
        
        Args:
            image (PIL.Image or np.ndarray): الصورة المراد إضافة الضوضاء لها
            noise_type (str): نوع الضوضاء (gaussian, salt_and_pepper)
            intensity (float): شدة الضوضاء
        
        Returns:
            PIL.Image: الصورة مع الضوضاء
        """
        # التأكد من أن الصورة بتنسيق NumPy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # إضافة ضوضاء جاوسية
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # إضافة ضوضاء ملح وفلفل
        elif noise_type == 'salt_and_pepper':
            salt_prob = intensity / 2
            pepper_prob = intensity / 2
            
            # إنشاء قناع عشوائي
            noise_mask = np.random.rand(*image.shape[:2])
            
            # إضافة النقاط البيضاء
            salt_mask = noise_mask < salt_prob
            image[salt_mask] = 255
            
            # إضافة النقاط السوداء
            pepper_mask = noise_mask > (1 - pepper_prob)
            image[pepper_mask] = 0
            
            noisy_image = image
        
        return Image.fromarray(noisy_image)
    
    @staticmethod
    def perspective_transform(image, max_shift=0.1):
        """
        تحويل منظور الصورة
        
        Args:
            image (PIL.Image or np.ndarray): الصورة المراد تحويلها
            max_shift (float): الحد الأقصى للتحويل كنسبة مئوية
        
        Returns:
            PIL.Image: الصورة بعد التحويل
        """
        # التأكد من أن الصورة بتنسيق NumPy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        height, width = image.shape[:2]
        
        # نقاط المصدر والهدف
        src_points = np.float32([
            [0, 0], 
            [width-1, 0], 
            [0, height-1], 
            [width-1, height-1]
        ])
        
        # حساب نقاط التحويل مع إضافة اضطراب عشوائي
        dst_points = np.float32([
            [width * random.uniform(0, max_shift), height * random.uniform(0, max_shift)],
            [width * (1 - random.uniform(0, max_shift)), height * random.uniform(0, max_shift)],
            [width * random.uniform(0, max_shift), height * (1 - random.uniform(0, max_shift))],
            [width * (1 - random.uniform(0, max_shift)), height * (1 - random.uniform(0, max_shift))]
        ])
        
        # حساب مصفوفة التحويل
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # تطبيق التحويل
        transformed_image = cv2.warpPerspective(
            image, 
            matrix, 
            (width, height), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
        
        return Image.fromarray(transformed_image)
    
    @staticmethod
    def augment_image(image, augmentations=None):
        """
        تطبيق تقنيات زيادة البيانات على الصورة
        
        Args:
            image (PIL.Image or np.ndarray): الصورة المراد زيادتها
            augmentations (list): قائمة بتقنيات الزيادة
        
        Returns:
            PIL.Image: الصورة بعد زيادة البيانات
        """
        # الإعدادات الافتراضية للزيادة
        if augmentations is None:
            augmentations = [
                ('rotate', {'angle_range': (-15, 15)}),
                ('noise', {'noise_type': 'gaussian', 'intensity': 0.02}),
                ('perspective', {'max_shift': 0.1})
            ]
        
        # نسخ الصورة الأصلية
        augmented_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image.copy())
        
        # تطبيق التقنيات
        for aug_type, params in augmentations:
            if aug_type == 'rotate':
                augmented_image = ThamudicDataAugmentation.rotate_image(augmented_image, **params)
            elif aug_type == 'noise':
                augmented_image = ThamudicDataAugmentation.add_noise(augmented_image, **params)
            elif aug_type == 'perspective':
                augmented_image = ThamudicDataAugmentation.perspective_transform(augmented_image, **params)
        
        return augmented_image
    
    @staticmethod
    def generate_augmented_dataset(original_dataset, num_augmentations=3):
        """
        توليد مجموعة بيانات معززة
        
        Args:
            original_dataset (list): مجموعة البيانات الأصلية
            num_augmentations (int): عدد الزيادات لكل صورة
        
        Returns:
            list: مجموعة البيانات المعززة
        """
        augmented_dataset = []
        
        for image, label in original_dataset:
            # إضافة الصورة الأصلية
            augmented_dataset.append((image, label))
            
            # توليد نسخ معززة
            for _ in range(num_augmentations):
                augmented_image = ThamudicDataAugmentation.augment_image(image)
                augmented_dataset.append((augmented_image, label))
        
        return augmented_dataset

def prepare_augmented_data(dataset_path):
    """
    تحضير البيانات المعززة للتدريب
    
    Args:
        dataset_path (str): مسار مجموعة البيانات
    
    Returns:
        list: مجموعة البيانات المعززة
    """
    # تحميل البيانات الأصلية
    original_dataset = load_dataset(dataset_path)
    
    # زيادة البيانات
    augmented_dataset = ThamudicDataAugmentation.generate_augmented_dataset(
        original_dataset, 
        num_augmentations=3
    )
    
    return augmented_dataset

def train(
    model_type='resnet', 
    dataset_path='data/thamudic_letters', 
    epochs=100, 
    learning_rate=3e-4, 
    batch_size=32, 
    use_augmentation=True
):
    """
    تدريب نموذج التعرف على الأحرف الثمودية
    
    Args:
        model_type (str): نوع النموذج للتدريب
        dataset_path (str): مسار مجموعة البيانات
        epochs (int): عدد دورات التدريب
        learning_rate (float): معدل التعلم
        batch_size (int): حجم دفعة التدريب
        use_augmentation (bool): استخدام زيادة البيانات
    
    Returns:
        nn.Module: النموذج المدرب
    """
    # إعداد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # إنشاء النموذج
    num_classes = 29  
    model = create_model(model_type=model_type, num_classes=num_classes)
    model.to(device)
    
    # تحميل وتحضير البيانات
    if use_augmentation:
        # زيادة البيانات
        dataset = prepare_augmented_data(dataset_path)
    else:
        # تحميل البيانات الأصلية
        dataset = load_dataset(dataset_path)
    
    # تقسيم البيانات
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # إنشاء محمل البيانات
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # معايير التدريب
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  
    
    # جدولة معدل التعلم
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  
    )
    
    # جدولة معدل التعلم
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  
        div_factor=25,
        final_div_factor=1000
    )
    
    # تتبع أفضل دقة
    best_accuracy = 0.0
    patience = 10  
    no_improve = 0
    
    # التدريب
    for epoch in range(epochs):
        # وضع التدريب
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # التدريب على الدفعات
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # تصفير المتدرجات
            optimizer.zero_grad()
            
            # الانتشار الأمامي
            outputs, _ = model(inputs)  
            loss = criterion(outputs, labels)
            
            # الانتشار العكسي والتحسين
            loss.backward()
            
            # تقليم المتدرجات لمنع الانفجار
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # تتبع الخسارة والدقة
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # التحقق
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs, _ = model(inputs)  
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # حساب المتوسطات
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # طباعة الإحصائيات
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')
        
        # حفظ أفضل نموذج
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model_weights(model, f'models/best_thamudic_model_{model_type}.pth')
            no_improve = 0
        else:
            no_improve += 1
            
        # التوقف المبكر
        if no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    return model

class ThamudicTrainer:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logging()
        
        # إعداد wandb للتتبع
        wandb.init(project="thamudic-recognition", config=self.config)
        
        self._setup_model()
        self._setup_data()
        self._setup_training()
    
    def _setup_model(self):
        self.model = create_model(
            model_type=self.config['model'].get('type', 'resnet'),
            num_classes=self.config['model'].get('num_classes', 28),
            input_channels=self.config['model'].get('input_channels', 3)
        ).to(self.device)
        
        # دالة الخسارة مع تنعيم التسميات
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.config['training'].get('label_smoothing', 0.1)
        )
        
        # المحسن مع تدرج الوزن
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # جدول معدل التعلم
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['training']['epochs'] // 3,
            T_mult=2
        )
    
    def _setup_data(self):
        train_dataset = ThamudicDataset(
            data_dir=self.config['data']['train_path'],
            char_mapping=self.config['data']['mapping_file'],
            augmentation_prob=self.config['data']['augmentation']['probability']
        )
        
        val_dataset = ThamudicDataset(
            data_dir=self.config['data']['val_path'],
            char_mapping=self.config['data']['mapping_file'],
            augmentation_prob=0
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
    
    def _setup_training(self):
        self.best_val_acc = 0
        self.patience_counter = 0
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            # تقليم التدرجات لمنع الانفجار
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = total_loss / len(self.val_loader)
        
        # حفظ أفضل نموذج
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.save_model()
        
        return val_acc, val_loss
    
    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
            
            # تدريب
            train_loss, train_acc = self.train_epoch()
            
            # تقييم
            val_acc, val_loss = self.validate()
            
            # تحديث جدول معدل التعلم
            self.scheduler.step()
            
            # تسجيل المقاييس
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
            
            # حفظ أفضل نموذج
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config['paths']['best_model'])
                self.logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                self.patience_counter += 1
            
            # الإيقاف المبكر
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info('Early stopping triggered')
                break
            
            self.logger.info(
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%'
            )

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    طباعة شريط تقدم في سطر واحد
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    تدريب النموذج لحقبة واحدة
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    # عنوان الحقبة
    print(f"\n{'='*20} حقبة {epoch} {'='*20}")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # صفر التدرجات
        optimizer.zero_grad()
        
        # الانتشار الأمامي
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # الانتشار العكسي والتحسين
        loss.backward()
        optimizer.step()
        
        # حساب الدقة
        _, predicted = outputs.max(1)
        total += targets.size(0)
        running_corrects += predicted.eq(targets).sum().item()
        running_loss += loss.item()
        
        # طباعة التقدم
        current_loss = loss.item()
        current_acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)
        suffix = f'خسارة: {current_loss:.4f} | دقة: {current_acc:.1f}%'
        print_progress_bar(batch_idx + 1, len(train_loader), 
                         prefix='تدريب:', suffix=suffix)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * running_corrects / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    تقييم النموذج على مجموعة التحقق
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    print("\nبدء التحقق...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # الانتشار الأمامي
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # حساب الدقة
            _, predicted = outputs.max(1)
            total += targets.size(0)
            running_corrects += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            # طباعة التقدم
            current_loss = loss.item()
            current_acc = 100. * running_corrects / total
            suffix = f'خسارة: {current_loss:.4f} | دقة: {current_acc:.1f}%'
            print_progress_bar(batch_idx + 1, len(val_loader), 
                             prefix='تحقق: ', suffix=suffix)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * running_corrects / total
    
    return epoch_loss, epoch_acc

def main():
    """
    الدالة الرئيسية للتدريب
    """
    # إعداد التسجيل
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # التأكد من وجود مجلد النماذج
    os.makedirs('models', exist_ok=True)
    
    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # تحميل البيانات
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'data', 'processed', 'train')
    val_dir = os.path.join(base_dir, 'data', 'processed', 'val')

    print(f"مجلد التدريب: {train_dir}")
    print(f"مجلد التحقق: {val_dir}")
    
    # إنشاء مجموعات البيانات
    train_dataset = ThamudicDataset(train_dir)
    val_dataset = ThamudicDataset(val_dir)
    
    print(f"عدد صور التدريب: {len(train_dataset)}")
    print(f"عدد صور التحقق: {len(val_dataset)}")
    
    # إنشاء data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # إنشاء النموذج
    num_classes = len(train_dataset.class_to_idx)
    print(f"عدد الفئات: {num_classes}")
    
    model = create_model(num_classes=num_classes)
    model = model.to(device)
    
    # حفظ معلومات الفئات
    class_info = {
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes
    }
    
    with open('models/class_info.json', 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=4)
    
    # تدريب النموذج
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        device=device
    )
    
    # حفظ النموذج النهائي
    save_model_weights(trained_model, f'models/final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

if __name__ == "__main__":
    main()
