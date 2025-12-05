# Pneumonia Binary Classifier

A deep learning-based binary classifier for detecting pneumonia in chest X-ray images using transfer learning with ResNet-18 architecture.

## Overview

This project implements a convolutional neural network (CNN) classifier to distinguish between normal and pneumonia cases in chest X-ray images. Using transfer learning and systematic hyperparameter optimization, the model achieves an F1 score of **0.9831** on held-out validation data.

## Dataset

The implementation uses the **PneumoniaMNIST** subset from the MedMNIST collection:

- **Training Set**: 4,708 images
  - Normal cases: 1,341 (28.5%)
  - Pneumonia cases: 3,367 (71.5%)
- **Validation Set**: 524 images (held-out for final evaluation)
- **Image Format**: 28×28 grayscale chest X-rays
- **Labels**: Binary classification (0 = normal, 1 = pneumonia)

## Key Features

### Model Architecture
- **Base Model**: ResNet-18 with ImageNet pretrained weights
- **Transfer Learning**: Full fine-tuning of all layers for domain adaptation
- **Custom Classifier Head**: Two-layer architecture (512→256→1) with dropout regularization
- **Input Adaptation**: Modified first convolutional layer for single-channel grayscale input

### Data Preprocessing & Augmentation
- Image resizing from 28×28 to 224×224 pixels
- Training augmentations:
  - Random rotation
  - Random horizontal flip
  - Random affine transformations
  - Normalization (mean=0.5, std=0.5)
- Validation preprocessing: Resize and normalize only (no augmentation)

### Class Imbalance Handling
- **Weighted Loss Function**: Positive class weight of 0.398 to penalize minority class misclassification
- **Stratified K-Fold CV**: Maintains consistent class distribution across all folds

## Training Strategy

### Hyperparameter Optimization
Systematic grid search over:
- **Batch sizes**: [32, 64]
- **Learning rates**: [1e-3, 1e-4]

**Optimal configuration** (batch size 32, learning rate 1e-4):
- Mean F1 Score: 0.9815 ± 0.0017 (3-fold CV)
- Superior generalization with minimal variance

### Cross-Validation
- **5-fold stratified cross-validation** on training set
- Mean F1 Score: **0.9835 ± 0.0022**
- Low standard deviation indicates robust, consistent performance

### Training Configuration
- **Loss Function**: BCEWithLogitsLoss with class weighting
- **Optimizer**: Adam with weight decay (1e-4) for L2 regularization
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Epochs**: 20 (final model), 15 (cross-validation), 12 (grid search)
- **Regularization**: 
  - Dropout: 0.5 (first layer), 0.3 (second layer)
  - Data augmentation
  - Weight decay

## Performance

### Final Results
- **Validation F1 Score**: 0.9831
- **Peak Performance**: Achieved at epoch 5
- **Train-Validation Gap**: < 0.02 (minimal overfitting)

### Model Robustness
- Consistent performance across all cross-validation folds
- Early training volatility (epoch 2: F1 = 0.8405) successfully overcome
- Stable convergence by epoch 4 (F1 = 0.9462)

## Technical Implementation

### Dependencies
```python
- torch
- torchvision
- medmnist
- numpy
- matplotlib
- scikit-learn
```

### Architecture Details

**Modified ResNet-18**:
```
Input: 1×224×224 (grayscale chest X-ray)
↓
Modified Conv1: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
↓
ResNet-18 Backbone (all layers trainable)
↓
Custom Classifier:
  - Dropout(0.5)
  - Linear(512 → 256)
  - ReLU
  - Dropout(0.3)
  - Linear(256 → 1)
↓
Output: Single logit for binary classification
```

## Key Design Decisions

1. **Full Fine-Tuning**: All ResNet layers remain trainable to enable deep adaptation to medical imaging domain
2. **Two-Stage Dropout**: Progressive regularization (0.5 → 0.3) balances overfitting prevention with model expressiveness
3. **Bottleneck Architecture**: 256-unit hidden layer forces feature compression while maintaining discriminative capacity
4. **Conservative Learning Rate**: 1e-4 enables careful weight updates that preserve ImageNet knowledge while adapting to X-rays
5. **Smaller Batch Size**: Batch size 32 introduces beneficial gradient noise for improved generalization

## Results Summary

The model successfully distinguishes pneumonia from normal chest X-rays with high accuracy and consistency. The systematic approach to hyperparameter tuning, coupled with effective handling of class imbalance and strong regularization strategies, resulted in a robust classifier suitable for medical image analysis applications.