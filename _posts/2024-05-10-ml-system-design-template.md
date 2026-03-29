---
layout: post
title: ML System Design Template
date: 2024-05-10
description: End-to-end ML system design framework covering requirements, data preparation, model development, evaluation, deployment, and monitoring -- with an EMG decoder case study.
tags: system-design
categories:
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

# Overview:
1. Clarifying requirements
2. Framing the problem as an ML task
3. Data preparation
4. Model development
5. Evaluation
6. Deployment and serving
7. Monitoring and infrastructure

{% include figure.liquid loading="eager" path="assets/img/blog/ch1-02.webp" class="img-fluid rounded z-depth-1" %}

## Clarifying Requirements
- Business Objective
- Features the system needs to support
- Data: what are the data sources? How large is the dataset? Is the data labeled?
- Constraints: How much computing power available? Is it cloud-based or on-device?
- Scale of the system: How many users do we have?
- Performance: How fast must prediction be? Does accuracy have more priority or latency?

## Framing the problem as an ML task
- Define the ML objective
   - I.e. Application: Ad Click Prediction System + Business Objective: Increase User Clicks + ML Objective: Maximize Click-through Rate
- Specify system's input and output
- Choose right ML Category (Supervised, Unsupervised, RL, Classification)

## Data preparation
- Data Sources: Who collected it? How clean is data?
- Data Storage

{% include figure.liquid loading="eager" path="assets/img/blog/ch1-07.webp" class="img-fluid rounded z-depth-1" %}

- Extract, Transform, and Load (ETL):
   - Extract: Extracts data from different sources
   - Transform: Data is cleansed, mapped, and transformed into specific format
   - Load: Transformed data is loaded into the target destination

{% include figure.liquid loading="eager" path="assets/img/blog/ch1-08.webp" class="img-fluid rounded z-depth-1" %}

- Data Types:
   - Unstructured Data: No schema + difficult to search (NoSQL databases / Data lakes)
- Feature Engineering: Deletion, imputation, feature scaling (normalization, standardization, log scaling)
- Discretization (bucketing): Convert continuous feature into categorical feature
- Encoding Categorical Features: Integer encoding, one-hot encoding, embedding learning

## Model Development
- Model Selection:
   - Establish a simple baseline
   - Experiment with simple models
   - Switch to more complex models if needed
- Model Training:
   - Collect raw data, identify features and labels
   - Select sampling strategy: convenience, snowball, stratified, reservoir, importance sampling
   - Split data: training, evaluation (validation), and test
   - Address class imbalances: resampling, altering loss function
   - Distributed training: Data Parallelism, Model Parallelism

## Evaluation
- Offline: Precision, recall, F1 Score, accuracy, ROC-AUC, confusion matrix
- Online: e.g., click-through rate, revenue lift

## Deployment and Serving
- Cloud vs. On-device Deployment
- Model Compression:
   - Knowledge distillation: train small model (student) to mimic larger model (teacher)
   - Pruning: setting least useful parameters to zero
   - Quantization: use fewer bits to represent parameters

## Test in Production
- Shadow deployment: deploy new model in parallel (only existing model's prediction is served)
- A/B testing: portion of traffic routed to new model (must be random)

## Prediction Pipeline
- Batch prediction
- Online prediction

## Monitoring
- Data distribution constantly changes
- Train on large datasets
- Regularly retrain model

---

## Case Study: EMG Gesture Recognition System

### 1. Problem Definition
- **Objective:** Develop a real-time gesture recognition system using sEMG data for a wearable device.
- **Requirements:** High accuracy, low latency, robust across users, efficient for wearable.

### 2. Data Preparation
- **Features:** Time-domain (RMS, Variance, MAV, SSI, Slope Sign Change, Zero Crossing, Waveform Length), Frequency (Median Frequency, FFT), Time-frequency (Spectrogram)
- **Multivariate Power Frequency Feature:** 40 Hz high-pass filter -> CSD over rolling window -> Riemannian Tangent Space Mapping -> 384-dimensional feature vector per 80ms window

### 3. Model Architectures
**Wrist Pose:**
- Input: MPF
- Architecture: CNN (dropout) + LayerNorm + FC_ReLU + CNN (dropout) + LayerNorm + FC_ReLU
- Optimization: AdamW + cosine annealing + early stopping

**Discrete Gestures:**
- Input: High-pass filtered (40 Hz) sEMG signal
- Architecture: CNN (stride 10) + LayerNorm + 3 LSTM + LayerNorm + Linear (9 classes)
- Loss: Binary Cross-Entropy
- Optimization: AdamW + gradient clipping

**Handwriting:**
- Input: MPF + SpecAugment + rotational augmentation
- Architecture: FC_LeakyReLU + 15-layer Conformer (4 attention heads) + channel average pooling + Linear + Softmax
- Loss: CTC Loss
- Metric: Character Error Rate

### 4. Evaluation
- **Wrist Pose:** Acquisition Time, Dial-in Time
- **Discrete Gestures:** Completion Rate, Confusion Matrix, First Hit Probability
- **Handwriting:** Character Error Rate, Adjusted Words per Minute

### 5. Future Enhancements
- Personalization via federated learning
- Self-supervised learning (masking signal and predicting masked signal)
