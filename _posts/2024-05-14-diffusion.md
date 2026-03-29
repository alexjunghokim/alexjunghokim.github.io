---
layout: post
title: Forward and Reverse Diffusion
date: 2024-05-14
description: Forward diffusion as a Markov chain, reparameterization trick, reverse diffusion with neural networks, and the ELBO objective.
tags: fundamentals
categories:
giscus_comments: false
related_posts: false
toc:
  beginning: true
---

## Forward Diffusion

In a Markov Chain process of T steps (where each step depends on the previous step), add Gaussian Noise to the current time step to generate next time step:

$$
\begin{gathered}
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \mu_t = \sqrt{1-\beta_t}\, x_{t-1},\; \Sigma_t = \beta_t I) \\
\text{Sample } \epsilon \sim \mathcal{N}(0, I) \text{ and set } x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon
\end{gathered}
$$

> $$\beta_t$$ at each time step is not constant — a variance scheduler is utilized (typically linear, quadratic, or cosine).

### Reparameterization Trick

$$
\begin{gathered}
\alpha_t = 1 - \beta_t \quad \text{(scaling factor that depends on noise variance)} \\[6pt]
\bar{\alpha}_t = \prod_{s=0}^{t} \alpha_s \quad \text{(cumulative scaling up to timestep } t\text{)} \\[6pt]
\epsilon_{0}, \epsilon_{1}, \ldots, \epsilon_{t-1} \sim \mathcal{N}(0, I) \\[6pt]
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon_0
\end{gathered}
$$

This allows us to express $$x_t$$ directly in terms of the initial data point $$x_0$$ and a single noise term, skipping all intermediate steps.

> Since $$\beta_t$$ is a hyperparameter, we can precompute $$\alpha_t$$ and $$\bar{\alpha}_t$$ for all timesteps.

## Reverse Diffusion

- As $$T \rightarrow \infty$$, $$x_T$$ approaches an isotropic Gaussian. We start by sampling $$x_T \sim \mathcal{N}(0, I)$$.
- Using the reverse distribution $$p(x_{t-1} \mid x_t)$$, iteratively sample backward from $$T$$ to 0.
- $$p(x_{t-1} \mid x_t)$$ is intractable — it requires knowing the distribution of all possible images. We use a neural network to approximate it:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t))
$$

The neural network learns the mean $$\mu_\theta$$ and variance $$\Sigma_\theta$$.

## Objective Function

Similar to VAE, we use the variational lower bound (ELBO) to minimize the negative log-likelihood with respect to the ground truth data sample $$x_0$$.
