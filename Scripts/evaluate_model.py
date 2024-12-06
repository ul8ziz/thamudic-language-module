import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, split_data

def evaluate_model(test_images, test_labels):
    # تحميل النموذج
    model = tf.keras.models.load_model("models/thamudic_model.h5")
    
    # تقييم النموذج
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # التنبؤ بالتصنيفات
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # طباعة تقرير التصنيف
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes))
    
    # رسم مصفوفة الارتباك
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(test_labels, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # تحميل البيانات
    images, labels = load_data("data/processed")
    
    # تقسيم البيانات
    _, _, _, _, test_images, test_labels = split_data(images, labels)
    
    # تقييم النموذج
    evaluate_model(test_images, test_labels)
