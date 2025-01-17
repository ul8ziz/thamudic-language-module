import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data import ThamudicDataset
from model import create_model
import logging
import argparse

def load_model_and_data(args):
    """
    تحميل النموذج والبيانات
    
    Args:
        args: معاملات البرنامج
        
    Returns:
        tuple: (النموذج، محمل البيانات، معلومات الفئات)
    """
    # تحميل معلومات الفئات
    try:
        dataset = ThamudicDataset(args.data_dir)
        class_info = {
            'num_classes': len(dataset.class_to_idx),
            'class_to_idx': dataset.class_to_idx
        }
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات: {e}")
        return None, None, None
    
    # تحميل النموذج
    model = create_model(num_classes=class_info['num_classes'])
    try:
        state_dict = torch.load(args.model, map_location=args.device)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
    except Exception as e:
        logging.error(f"خطأ في تحميل النموذج: {e}")
        return None, None, None
    
    return model, dataloader, class_info

def evaluate_model(model, dataloader, device):
    """
    تقييم النموذج
    
    Args:
        model: النموذج المدرب
        dataloader: محمل البيانات
        device: الجهاز المستخدم
        
    Returns:
        tuple: (التنبؤات، التصنيفات الحقيقية)
    """
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    رسم مصفوفة الارتباك
    
    Args:
        y_true: التصنيفات الحقيقية
        y_pred: التنبؤات
        class_names: أسماء الفئات
        output_dir: مجلد الإخراج
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('مصفوفة الارتباك')
    plt.xlabel('التنبؤات')
    plt.ylabel('القيم الحقيقية')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_dir):
    """
    حفظ تقرير التصنيف
    
    Args:
        y_true: التصنيفات الحقيقية
        y_pred: التنبؤات
        class_names: أسماء الفئات
        output_dir: مجلد الإخراج
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3
    )
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """
    الدالة الرئيسية
    """
    parser = argparse.ArgumentParser(description='تقييم نموذج التعرف على الحروف الثمودية')
    parser.add_argument('--model', default='models/best_model.pth', help='مسار النموذج المدرب')
    parser.add_argument('--data_dir', required=True, help='مجلد بيانات الاختبار')
    parser.add_argument('--output_dir', default='D:/ul8ziz/GitHub/thamudic-language-module/evaluation', help='مجلد حفظ نتائج التقييم')
    parser.add_argument('--batch_size', type=int, default=32, help='حجم الدفعة')
    args = parser.parse_args()
    
    # إعداد التسجيل
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # تحديد الجهاز
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {args.device}')
    
    # إنشاء مجلد الإخراج
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تحميل النموذج والبيانات
    model, dataloader, class_info = load_model_and_data(args)
    if None in (model, dataloader, class_info):
        return
    
    # تقييم النموذج
    predictions, true_labels = evaluate_model(model, dataloader, args.device)
    
    # تحويل الفهارس إلى أسماء الفئات
    idx_to_class = {v: k for k, v in class_info['class_to_idx'].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # رسم مصفوفة الارتباك
    plot_confusion_matrix(true_labels, predictions, class_names, args.output_dir)
    
    # حفظ تقرير التصنيف
    save_classification_report(true_labels, predictions, class_names, args.output_dir)
    
    print(f"\nتم حفظ نتائج التقييم في المجلد: {args.output_dir}")

if __name__ == "__main__":
    main()
