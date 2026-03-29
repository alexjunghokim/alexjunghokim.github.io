---
layout: post
title: Fundamentals of CNN
date: 2024-05-11
description: Discrete convolution, pooling, output feature map sizing, transposed convolution, and batch normalization.
tags: fundamentals
categories:
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

Discrete convolution is a linear transformation that preserves the notion of ordering, it is sparse (only few input units contribute to given output unit) and reuses parameters (same weights applied to multiple locations in the input). A kernel slides across the input feature map. At each location, product between each element of the kernel and input element it overlaps is computed and the results are summed up to obtain the output in the current location. If there are multiple feature maps (i.e. R.G.B.), the kernel will have to be 3-dimensional / each one of the feature maps will be convolved with a distinct kernel - and the resulting feature maps will be summed up elementwise to produce the output feature map.

### Pooling
This operation reduces the size of feature maps by using some function to summarize subregions, such as taking the average of the maximum value. Pooling works by sliding a window across the input and feeding the content of the window to a pooling function.

### Output Feature Map Size
Given the following symbols:

$$
\begin{gathered}
i = \text{input size}\\
k = \text{kernel size}\\
s = \text{stride size}\\
p = \text{padding size}\\
\\
\text{For any i and k, and for s = 1 and p = 0:}\\
o = (i - k) + 1 \\
\\
\text{For any i and k, and for s = 1:}\\
o = (i - k) + 2p + 1 \\
\\
\text{For any i and for k odd (k = 2n+1, }n \in \mathbb{N}) \text{, s = 1 and p = \lfloor k/2 \rfloor = n:} \\
o = i + 2\lfloor k/2 \rfloor - (k - 1) = i \\
\\
\text{For any i, k and s, and for p = 0:} \\
o = \left\lfloor \frac{i-k}{s} \right\rfloor + 1 \\
\\
\text{For any i, k, p and s:} \\
o = \left\lfloor \frac{i + 2p - k}{s} \right\rfloor + 1
\end{gathered}
$$

For pooling arithmetic, since pooling does not involve zero padding:

$$
\begin{gathered}
\text{For any i, k and s:} \\
o = \left\lfloor \frac{i - k}{s} \right\rfloor + 1
\end{gathered}
$$

### Transposed Convolution
The need for transposed convolution arises from the desire to use a transformation going in the opposite direction of a normal convolution. This may be useful as a decoding layer of a convolutional autoencoder or to project feature maps to a higher-dimensional space.

$$
\begin{gathered}
\text{Convolution described by s = 1, p = 0, and k has an associated transposed} \\
\text{convolution described by k' = k, s' = s and p' = k - 1 and its output size is:} \\
o' = i' + (k - 1)
\end{gathered}
$$

### Batch Normalization
The input element is subtracted by the estimated mean and then divided by the standard deviation, followed by a scale coefficient and offset:

$$
\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.
$$

$$
\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.
$$

Batch normalization runs differently in training and inference modes: in training the normalization is done using minibatch statistics, during inference, the whole dataset is used to find the mean and standard deviation for normalization.
