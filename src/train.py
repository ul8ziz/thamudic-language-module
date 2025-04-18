"""
Training script for Thamudic character recognition model
سكربت تدريب نموذج التعرف على الحروف الثمودية
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from pathlib import Path
import logging
from tqdm import tqdm
from models import ThamudicRecognitionModel
from dataset_processing import ThamudicDataset
from torch.utils.tensorboard import SummaryWriter
import json

# Setup logging | إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_dir, writer):
    """
    Train the model with validation and early stopping
    تدريب النموذج مع التحقق والإيقاف المبكر
    """
    
    # Initialize best validation loss and patience counter
    # تهيئة أفضل خسارة للتحقق وعداد الصبر
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        # مرحلة التدريب
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        progress_bar = tqdm(train_loader, desc=f'Training | التدريب')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            # حساب الدقة
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            train_loss += loss.item()
            
            # Update progress bar
            # تحديث شريط التقدم
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * train_correct / train_total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        # Validation phase
        # مرحلة التحقق
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("\nValidating... | التحقق...")
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation | التحقق'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch statistics
        # حساب إحصائيات الحلقة
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Log to TensorBoard
        # تسجيل إلى TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Learning rate scheduling
        # جدولة معدل التعلم
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        # التحقق من الإيقاف المبكر
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            # حفظ أفضل نموذج
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
                
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

def main():
    # Set random seed for reproducibility
    # تعيين البذرة العشوائية لإمكانية التكرار
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Check if GPU is available and set the device
    # التحقق من توفر GPU وتعيين الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Setup directories | إعداد المجلدات
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'letters'
    save_dir = base_dir / 'models' / 'checkpoints'
    log_dir = base_dir / 'runs'
    
    # Create directories if they don't exist
    # إنشاء المجلدات إذا لم تكن موجودة
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Data augmentation transformations for training
    # تحويلات تحسين البيانات للتدريب
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(5),  # Limited rotation to preserve character shape | تدوير محدود للحفاظ على شكل الحرف
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Limited translation | إزاحة محدودة
            scale=(0.9, 1.1),      # Limited scaling | تغيير حجم محدود
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transformations | تحويلات التحقق
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with order validation
    # تحميل مجموعة البيانات مع التحقق من صحة الترتيب
    train_dataset = ThamudicDataset(data_dir, transform=transform_train)
    val_dataset = ThamudicDataset(data_dir, transform=transform_val)
    
    # Check character order
    # التحقق من ترتيب الأحرف
    with open(base_dir / 'data' / 'letter_mapping.json', 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        expected_order = [item['letter'] for item in mapping_data['thamudic_letters']]
    
    # Check if actual order matches expected order
    # التحقق من تطابق الترتيب الفعلي مع الترتيب المتوقع
    actual_order = [cls.split('_')[1] for cls in train_dataset.classes]
    if actual_order != expected_order:
        raise ValueError(
            "Character order in dataset does not match expected order. "
            f"Expected order: {expected_order}, "
            f"Actual order: {actual_order}"
        )
    
    # Create DataLoader with increased number of workers
    # إنشاء DataLoader مع زيادة عدد العمال
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
    
    # Initialize model
    # تهيئة النموذج
    model = ThamudicRecognitionModel(num_classes=len(train_dataset.classes)).to(device)
    
    # Loss function with class weights
    # دالة الخسارة مع أوزان الفئات
    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(device))
    
    # Optimizer with modified parameters
    # محسن التعلم مع تعديل المعاملات
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with modified parameters
    # جدولة معدل التعلم مع تعديل المعاملات
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # Initialize TensorBoard writer
    # تهيئة كاتب TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Train the model
    # تدريب النموذج
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        save_dir=save_dir,
        writer=writer
    )
    
    writer.close()
    logging.info('Training completed successfully | انتهى التدريب بنجاح')
    
if __name__ == '__main__':
    main()
