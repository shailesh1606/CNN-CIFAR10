# CNN-Based Object Classification (CIFAR-10)

## Problem Overview

This project implements a lightweight Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The performance of the CNN is compared with classical machine learning models.

Pipeline followed:
Dataset Loading → Normalization → CNN Design → Training → Validation → Accuracy Comparison

---

## Dataset

CIFAR-10 contains:
- 60,000 color images (32 × 32)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images and 10,000 test images

The dataset was loaded using `torchvision.datasets.CIFAR10`.

---

## Preprocessing

- Images converted to tensors
- Pixel values normalized
- Data augmentation applied to training set:
  - Random Horizontal Flip
  - Random Rotation

---

## Model Architecture

The CNN consists of:
- Two Convolutional layers
- ReLU activations
- MaxPooling layers
- Fully connected layers
- Output layer with 10 classes

Loss Function: CrossEntropyLoss  
Optimizer: Adam  

---

## Classical ML Comparison

For comparison, the images were flattened and trained using:

- Support Vector Machine (SVM)
- Logistic Regression

---

## Results

| Model | Accuracy |
|-------|----------|
| CNN | ~75% |
| SVM | ~45% |
| Logistic Regression | ~28–35% |

The CNN outperforms classical models because it preserves spatial information and learns hierarchical image features.

---

## Tools Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Google Colab

---

## Conclusion

The experiment shows that CNNs perform significantly better than classical machine learning models for image classification tasks due to their ability to extract spatial features.
