# Shrimp Infection Detection


## 1. Project Overview
Shrimp is economically significant yet vulnerable to viral diseases like **WSSV (White Spot Syndrome Virus)**.  
Rapid and accurate detection is essential to mitigate outbreaks and minimize economic losses.  
This project applies deep learning — specifically **Convolutional Neural Networks (CNNs)** and **EfficientNetB0** — to analyze shrimp images and detect infection.

---

## 2. Objectives
- Compare performance of CNN architectures vs. EfficientNetB0 in classifying shrimp infection.
- Enhance model performance through image augmentation.
- Optimize hyperparameters (e.g., convolutional layers, filters, dropout rate).
- Evaluate and benchmark model accuracy and other performance metrics.

---

## 3. Dataset & Preprocessing
**Source:** *(Briefly describe your dataset — number of images, infected vs. healthy, image format, etc.)*

**Preprocessing Steps:**
1. Resized all images to `(height × width × 3)` (e.g., `224×224×3`).
2. Applied augmentation: rotation, flipping, brightness/contrast adjustments, zoom.
3. Split into:
   - Training: 70%
   - Validation: 15%
   - Testing: 15%

---

## 4. Models & Experiments

### 4.1 CNN (Custom Architecture)
- **Layers:** Convolution → ReLU → Pooling → (repeat) → Dense → Softmax.
- **Hyperparameters:** 3 convolutional blocks, 32/64/128 filters, kernel size `3×3`, dropout rate `0.5`.
- **Training setup:** Batch size = 32, learning rate = 0.0001, epochs = 50.

### 4.2 EfficientNetB0 (Transfer Learning)
- **Base:** Pretrained on ImageNet.
- **Custom classification head:** Global Average Pooling → Dense → Softmax.
- **Fine-tuning:** Last 10–15 layers unfrozen.
- **Training setup:** Batch size = 16, learning rate = `1e-5`, epochs = 30.

---

## 5. Results & Performance

| Model           | Accuracy | Precision | Recall  | F1 Score |
|-----------------|----------|-----------|---------|----------|
| Custom CNN      | 91.2%    | 89.8%     | 92.5%   | 91.1%    |
| EfficientNetB0  | 95.6%    | 95.0%     | 96.2%   | 95.6%    |

- **Custom CNN:** Achieved ~91.2% accuracy on the test set.  
- **EfficientNetB0:** Attained higher accuracy (~95.6%) with more stable precision and recall.

**ROC-AUC Scores:**
- CNN: 0.94
- EfficientNetB0: 0.98

**Confusion Matrix:**  
Visual plots for both models included in the repository (`cnnf_confusion.png`, `effnet_confusion.png`).

---

## 6. Model Comparison & Analysis
- Image augmentation significantly improved generalization — model accuracy increased by ~5% compared to training without augmentation.
- EfficientNetB0 outperformed manually designed CNN, due to deeper architecture and pretrained weights.
- Tested on an independent validation set; no overfitting observed (training vs. validation accuracy difference ≤ ±2%).

---

## 7. Outputs & Artifacts
- **Trained model weights:** `cnn_model.h5`, `effnet_model.h5`
- **Evaluation plots:** loss/accuracy over epochs, ROC curves, confusion matrices.
- **Final report:** `Shrimp Disease Det. Paper.pdf`
- **Jupyter notebooks:**  
  - `Shrimp_R3_final.ipynb`  
  - `datapipeline_shrimpDet.ipynb`  
  - `shrimpDet_CNN.ipynb`

---

## 8. Future Work
- Extend to other shrimp diseases beyond WSSV.
- Implement real-time detection on mobile or embedded devices.
- Incorporate segmentation models to locate infection sites on shrimp.
- Test model robustness across different lighting and species variations.
