---
layout: post
title: "Estimating Probabilities from Data: MLE, MAP, and Bayesian Inference"
date: 2024-05-13
description: Maximum Likelihood Estimation, Bayesian approach with Beta priors, MAP estimation, and the posterior predictive distribution.
tags: fundamentals
categories:
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

Data comes from the distribution $$P(X, Y)$$. If we have access to $$P(X, Y)$$, we can use the Bayes Optimal Classifier to predict the most likely label: $$\operatorname{argmax}_{y} P(y \mid x)$$.

Most of supervised learning is estimating $$P(X, Y)$$:
- **Discriminative:** $$P(X, Y) = P(Y \mid X) \cdot P(X)$$
- **Generative:** $$P(X, Y) = P(X \mid Y) \cdot P(Y)$$

## Maximum Likelihood Estimation

1. Make an explicit modeling assumption about what type of distribution data is sampled from.
2. Set parameters of this distribution so data you observed is as likely as possible.

For a coin toss, assume outcomes follow a binomial distribution with parameters $$n$$ and $$\theta$$:

$$
P(D \mid \theta) = \binom{n_H + n_T}{n_H} \theta^{n_H} (1 - \theta)^{n_T}
$$

**MLE Principle:** Find $$\hat{\theta}$$ to maximize the likelihood $$P(D; \theta)$$:

$$
\hat{\theta}_{MLE} = \operatorname*{argmax}_{\theta} \, P(D; \theta)
$$

Take the log (monotonic, turns products into sums), differentiate, and set to zero:

$$
\begin{align}
\hat{\theta}_{MLE} &= \operatorname*{argmax}_{\theta} \, n_H \cdot \log(\theta) + n_T \cdot \log(1 - \theta)
\end{align}
$$

Solving:

$$
\frac{n_H}{\theta} = \frac{n_T}{1 - \theta} \implies \theta = \frac{n_H}{n_H + n_T}
$$

**With prior knowledge:** If you suspect $$\theta \approx 0.5$$ but have small sample size, add $$m$$ imaginary throws:

$$
\hat{\theta} = \frac{n_H + m}{n_H + n_T + 2m}
$$

## Bayesian Approach

Model $$\theta$$ as a random variable with prior distribution $$P(\theta)$$:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
$$

- $$P(\theta)$$ — prior over $$\theta$$
- $$P(D \mid \theta)$$ — likelihood
- $$P(\theta \mid D)$$ — posterior

A natural prior is the **Beta distribution** (conjugate prior to the binomial):

$$
P(\theta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}
$$

where $$B(\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)}$$. The posterior becomes:

$$
P(\theta \mid D) \propto \theta^{n_H + \alpha - 1} (1 - \theta)^{n_T + \beta - 1}
$$

## Maximum a Posteriori (MAP) Estimation

Find $$\hat{\theta}$$ that maximizes the posterior $$P(\theta \mid D)$$:

$$
\begin{align}
\hat{\theta}_{MAP} &= \operatorname*{argmax}_{\theta} \, \log P(D \mid \theta) + \log P(\theta) \\
&= \operatorname*{argmax}_{\theta} \, (n_H + \alpha - 1) \cdot \log(\theta) + (n_T + \beta - 1) \cdot \log(1 - \theta) \\
&\implies \hat{\theta}_{MAP} = \frac{n_H + \alpha - 1}{n_H + n_T + \alpha + \beta - 2}
\end{align}
$$

- MAP is identical to MLE with $$\alpha - 1$$ hallucinated heads and $$\beta - 1$$ hallucinated tails.
- As $$n \rightarrow \infty$$, $$\hat{\theta}_{MAP} \rightarrow \hat{\theta}_{MLE}$$.

## "True" Bayesian Approach

Use the posterior predictive distribution directly:

$$
P(Y \mid D, X) = \int_{\theta} P(Y \mid \theta, D, X) \, P(\theta \mid D) \, d\theta
$$

This is generally intractable in closed form. For the coin toss:

$$
\begin{align}
P(\text{heads} \mid D) &= \int_{\theta} \theta \, P(\theta \mid D) \, d\theta \\
&= E[\theta \mid D] \\
&= \frac{n_H + \alpha}{n_H + \alpha + n_T + \beta}
\end{align}
$$

This uses the fact that $$P(\text{heads} \mid D, \theta) = P(\text{heads} \mid \theta)$$ — which holds because we assumed data is drawn from a binomial distribution.
