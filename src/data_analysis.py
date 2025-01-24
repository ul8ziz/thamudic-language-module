import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Tuple
from data_loader import ThamudicDataset, load_and_preprocess_data
import albumentations as A
import json

class DataAnalyzer:
    """Advanced data analysis tools for Thamudic dataset"""
    
    def __init__(self, dataset: ThamudicDataset, output_dir: str = 'analysis_results'):
        """
        Initialize the analyzer
        
        Args:
            dataset: ThamudicDataset instance
            output_dir: Directory to save analysis results
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution and balance"""
        class_counts = pd.Series(self.dataset.labels).value_counts()
        
        # Calculate distribution metrics
        stats = {
            'total_samples': len(self.dataset),
            'num_classes': len(class_counts),
            'min_class_size': int(class_counts.min()),
            'max_class_size': int(class_counts.max()),
            'mean_class_size': float(class_counts.mean()),
            'std_class_size': float(class_counts.std()),
            'imbalance_ratio': float(class_counts.max() / class_counts.min())
        }
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close()
        
        return stats
    
    def analyze_image_quality(self, sample_size: int = 100) -> Dict:
        """Analyze image quality metrics"""
        metrics = {
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'noise': []
        }
        
        indices = np.random.choice(len(self.dataset), min(sample_size, len(self.dataset)), replace=False)
        
        for idx in indices:
            image, _ = self.dataset[idx]
            image = image.numpy().squeeze()
            
            # Calculate metrics
            metrics['brightness'].append(np.mean(image))
            metrics['contrast'].append(np.std(image))
            
            # Calculate sharpness using Laplacian
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            metrics['sharpness'].append(np.var(laplacian))
            
            # Estimate noise using median absolute deviation
            noise = np.median(np.abs(image - np.median(image)))
            metrics['noise'].append(noise)
        
        # Calculate statistics
        stats = {}
        for metric, values in metrics.items():
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # Plot distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for (metric, values), ax in zip(metrics.items(), axes.ravel()):
            sns.histplot(values, ax=ax)
            ax.set_title(f'{metric.capitalize()} Distribution')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_quality_metrics.png')
        plt.close()
        
        return stats
    
    def analyze_feature_space(self, model: torch.nn.Module, device: torch.device) -> None:
        """Analyze feature space using dimensionality reduction"""
        loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        features = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for images, batch_labels in loader:
                images = images.to(device)
                # Assuming model has a method to extract features
                batch_features = model.extract_features(images)
                features.append(batch_features.cpu().numpy())
                labels.extend(batch_labels.numpy())
        
        features = np.concatenate(features)
        labels = np.array(labels)
        
        # Apply dimensionality reduction
        for method, reducer in [
            ('PCA', PCA(n_components=2)),
            ('t-SNE', TSNE(n_components=2))
        ]:
            reduced_features = reducer.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                                c=labels, cmap='tab20')
            plt.colorbar(scatter)
            plt.title(f'{method} Visualization of Feature Space')
            plt.savefig(self.output_dir / f'feature_space_{method.lower()}.png')
            plt.close()
    
    def analyze_augmentation_impact(self, num_samples: int = 5) -> None:
        """Analyze the impact of data augmentation"""
        indices = np.random.choice(len(self.dataset), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        for i, idx in enumerate(indices):
            image, _ = self.dataset[idx]
            image = image.numpy().squeeze()
            
            # Original image
            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title('Original')
            
            # Apply different augmentations
            for j, aug in enumerate([
                A.RandomRotate90(p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
            ], 1):
                augmented = aug(image=image)['image']
                axes[i, j].imshow(augmented, cmap='gray')
                axes[i, j].set_title(aug.__class__.__name__)
            
            for ax in axes[i]:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'augmentation_examples.png')
        plt.close()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'class_distribution': self.analyze_class_distribution(),
            'image_quality': self.analyze_image_quality(),
        }
        
        # Save report as JSON
        with open(self.output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

def main():
    """Run comprehensive data analysis"""
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent
        data_dir = project_dir / 'data' / 'letters' / 'processed_letters'
        mapping_file = project_dir / 'data' / 'mapping.json'
        output_dir = project_dir / 'analysis_results'
        
        # Load dataset
        train_dataset, _ = load_and_preprocess_data(
            str(data_dir),
            str(mapping_file),
            enhance_quality=True
        )
        
        # Create analyzer and generate report
        analyzer = DataAnalyzer(train_dataset, str(output_dir))
        report = analyzer.generate_report()
        
        # Log results
        logging.info("Analysis completed successfully")
        logging.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main()
