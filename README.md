🦐 Shrimp Disease Detection using Deep Learning
📌 Overview
This project focuses on detecting White Spot Syndrome Virus (WSSV) infections in shrimp using Convolutional Neural Networks (CNN) and EfficientNet-B0.
The goal is to help shrimp farmers identify infections early and minimize economic losses.

🎯 Objectives
Develop an AI-based detection system for WSSV.

Compare custom CNN with EfficientNet-B0.

Use image augmentation to improve accuracy.

Achieve >95% accuracy in classification.

📊 Methodology
1. Dataset Preparation
1,650 shrimp images (healthy & infected).

Images resized to 200×200 and normalized.

Data augmentation (rotation, flips, scaling, brightness).

2. Model Architectures
Custom CNN: 3 convolutional layers, ReLU activation, max-pooling, dense layers.

EfficientNet-B0: Transfer learning from ImageNet, fine-tuned for shrimp detection.

3. Training
Loss function: Binary Crossentropy

Optimizer: RMSprop (lr=0.001)

Metric: Accuracy

4. Evaluation
Custom CNN accuracy: 95%

EfficientNet-B0 accuracy: 97.9%

Benchmarked against ERCN, VGG16, LSTM, GRU.

📈 Results
Model	Accuracy
EfficientNet-B0 (Opt.)	98.9%
Custom CNN	95.0%
CNN-based Model	98.46%
ERCN	95.2%
VGG16	91.0%
LSTM	92.3%
GRU	90.0%
