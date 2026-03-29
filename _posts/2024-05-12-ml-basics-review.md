---
layout: post
title: ML Basics Review
date: 2024-05-12
description: Supervised and unsupervised learning, bias-variance tradeoff, regularization (L1/L2/Elastic Net), regression and classification metrics, SVM, and logistic regression.
tags: fundamentals
categories:
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

## Supervised Learning
Supervised learning consists of input values (independent variables) and target/output values (dependent variables) with the goal of creating a function for mapping inputs to outputs accurately.

## Unsupervised Learning
Unsupervised learning consists of just input data with no output/target data. The goal is to find relationships, characteristics, and patterns among the input data.

## Reinforcement Learning
An agent interacts with an environment to maximize cumulative reward. The agent learns by taking actions and receiving feedback in the form of rewards or penalties. The goal is to learn the optimal policy.

## Steps in ML Pipeline
1. Data Collection
2. Data Preprocessing — balancing, normalization, cleaning missing values
3. Feature Engineering — transforming raw data into useful features
4. Data Splitting — train/valid/test
5. Model Selection — choose optimal model for the use case
6. Model Training — learn patterns via optimization
7. Model Evaluation — performance and generalization metrics
8. Model Optimization — hyperparameter tuning, regularization, ensembles
9. Model Deployment
10. Model Maintenance

## Generative vs. Discriminative Models

**Generative models** learn the joint probability distribution $$P(X, Y) = P(Y) \cdot P(X \mid Y)$$.

**Discriminative models** learn the conditional probability $$P(Y \mid X)$$ directly.

Key differences:
- Generative models can generate new samples; discriminative models learn decision boundaries.
- Generative models are more expensive to train (estimate both $$P(Y)$$ and $$P(X \mid Y)$$).

## Bias vs. Variance
- **High Bias:** Model oversimplifies, misses patterns → underfitting.
- **High Variance:** Model is overly complex, sensitive to noise → overfitting.

Decreasing bias increases variance, and vice versa.

## Regularization

Adds a penalty term to the loss function to prevent overfitting:

$$
J_{\text{reg}} = J + \alpha \cdot R(w)
$$

### L1 Regularization (Lasso)
Encourages sparsity — can drive parameters to zero (feature selection):

$$
R(w) = \|w\|_1 = |w_1| + |w_2| + \cdots + |w_n|
$$

### L2 Regularization (Ridge)
Promotes smaller parameter values without driving them to zero:

$$
R(w) = \|w\|_2^2 = w_1^2 + w_2^2 + \cdots + w_n^2
$$

### Elastic Net
Combines L1 and L2:

$$
J_{\text{reg}} = J + \alpha \left[\lambda \cdot \|w\|_1 + (1 - \lambda) \cdot \|w\|_2^2 \right]
$$

## Regression Metrics

| Metric | Formula |
|--------|---------|
| **MSE** | $$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$ |
| **RMSE** | $$\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$ |
| **MAE** | $$\frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$$ |
| **R-Squared** | $$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$ |

## Classification Metrics

| Metric | Formula |
|--------|---------|
| **Accuracy** | $$\frac{TP + TN}{TP + TN + FP + FN}$$ |
| **Precision** | $$\frac{TP}{TP + FP}$$ |
| **Recall** | $$\frac{TP}{TP + FN}$$ |
| **F1 Score** | $$2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$ |
| **Specificity** | $$\frac{TN}{TN + FP}$$ |

**AUC-ROC:** Area under the ROC curve (True Positive Rate vs. False Positive Rate at various thresholds).

## SVM

Finds the hyperplane that separates two classes with the largest margin:

$$
w^T x + b = 0
$$

Optimization objective:

$$
\min \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1
$$

### Kernel Trick
Maps features into higher-dimensional space where classes become linearly separable:
- **Linear:** $$K(x, y) = x^T y$$
- **Polynomial:** $$K(x, y) = (a \cdot x^T y + c)^d$$
- **RBF (Gaussian):** $$K(x, y) = \exp(-\gamma \|x - y\|^2)$$

## Logistic Regression

Models class probability using the sigmoid function:

$$
h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

Cost function (binary cross-entropy):

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i)) \right]
$$
