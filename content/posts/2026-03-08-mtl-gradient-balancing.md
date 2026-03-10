---
title: "Gradient Balancing in Multi-Task Recommendation: A Systematic Guide"
date: 2026-03-08T10:00:00-07:00
draft: false
description: "A comprehensive survey of gradient balancing methods for multi-task recommendation systems, from GradNorm to DRGrad, with math and pseudocode for each approach."
tags: ['machine-learning', 'multi-task-learning', 'recommendation-systems', 'optimization', 'deep-learning']
---
<p align="center">
<img src="/mtml_balancing/MTML.png" width="600" height="300"><br>
<p>
<!--more-->

Modern recommendation systems rarely optimize a single objective. A typical industrial system simultaneously optimizes click-through rate (CTR), conversion rate (CVR), dwell time, and downstream revenue signals — all through a shared model backbone. When these tasks compete rather than cooperate during training, the result is **negative transfer**: the multi-task model performs worse than individually trained single-task models on one or more objectives. The root cause almost always traces back to poorly balanced gradients.

This post surveys the major gradient balancing methods developed between 2018 and 2025, grouped by contribution type. For each method we cover the core intuition, the mathematical formulation, pseudocode for implementation, and a frank assessment of its trade-offs. The goal is to give practitioners a navigable reference for deciding which technique to apply in their own systems.

---

## 1. Problem Formulation

### The Multi-Task Loss

Let $T$ be the number of tasks and $\mathcal{L}_i$ the loss for task $i$. The standard multi-task objective is:


$$\mathcal{L}_{total} = \sum_{i=1}^{T} w_i \mathcal{L}_i$$


where $w_i \geq 0$ are scalar task weights. The parameter update follows:


$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{total} = \theta - \eta \sum_{i=1}^{T} w_i \mathbf{g}_i$$


where $\mathbf{g}_i = \nabla_\theta \mathcal{L}_i$ is the gradient of task $i$ with respect to shared parameters $\theta$.

### The Three Primary Failure Modes

**Gradient Conflict.** When two task gradients point in opposing directions, their combination partially or fully cancels. Formally, conflict occurs when the cosine similarity is negative:


$$\cos(\mathbf{g}_i, \mathbf{g}_j) = \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_i\| \|\mathbf{g}_j\|} < 0$$


This directly causes negative transfer: progress on task $i$ undoes progress on task $j$.

**Magnitude Imbalance.** Tasks with high-frequency labels (e.g., clicks) naturally produce larger gradient norms than sparse objectives (e.g., purchases). If $\|\mathbf{g}_{CTR}\| \gg \|\mathbf{g}_{CVR}\|$, the dense task dominates the parameter update regardless of the scalar weight $w_i$. This is the **seesaw effect** in recommendation: improving CTR degrades CVR, or vice versa, because the shared backbone is pulled predominantly by one objective.

**Curvature Sensitivity.** In curved loss landscapes, the gradient magnitude at a given point reflects both the learning signal and the local curvature. High positive curvature along one task's direction means even a moderate step causes large loss changes for other tasks. Tasks whose loss surfaces have sharply different curvature will be misweighted by any method that only looks at first-order gradient information.

### Notation Summary

| Symbol | Meaning |
|--------|---------|
| $T$ | Number of tasks |
| $\mathbf{g}_i$ | Gradient of task $i$ w.r.t. shared parameters |
| $w_i$ | Scalar weight for task $i$ |
| $\mathbf{h}$ | Shared representation / bottleneck |
| $\alpha$ | Hyperparameter controlling aggressiveness of balancing |

---

## 2. Methods

### 2.1 GradNorm (ICML 2018)

**Intuition.** Imagine a relay race where some runners are sprinting and others are jogging. GradNorm is the coach who watches pace and tells the sprinters to slow down, or the joggers to speed up, so the team arrives in formation [^1]. Concretely, it maintains a running estimate of each task's training speed and adjusts $w_i$ so that faster-learning tasks do not crowd out slower ones.

**Math.** At training step $t$, define the weighted gradient norm for task $i$:


$$G_W^{(i)}(t) = \|w_i(t) \nabla_W \mathcal{L}_i(t)\|_2$$


where $W$ denotes the parameters of the final shared layer (a proxy for the full network). The **relative inverse training rate** measures how fast task $i$ learns relative to the average:


$$\tilde{\mathcal{L}}_i(t) = \frac{\mathcal{L}_i(t)}{\mathcal{L}_i(0)}$$


The average rate is $\bar{\mathcal{L}}(t) = \frac{1}{T} \sum_i \tilde{\mathcal{L}}_i(t)$.

The target gradient norm for task $i$ is:


$$G_W^{target}(t) = \bar{G}_W(t) \cdot \left[\frac{\tilde{\mathcal{L}}_i(t)}{\bar{\mathcal{L}}(t)}\right]^\alpha$$


where $\bar{G}_W(t) = \frac{1}{T} \sum_i G_W^{(i)}(t)$ is the mean gradient norm. The hyperparameter $\alpha \geq 0$ controls how aggressively slow learners are prioritized:
- $\alpha = 0$: all tasks receive equal gradient norm (full equalization).
- $\alpha \to \infty$: tasks that are falling behind receive disproportionately large boosts.
- Typical value: $\alpha \in [0.5, 2.0]$.

The GradNorm auxiliary loss, minimized with respect to $w_i$ only (treating the network parameters as constants), is:


$$\mathcal{L}_{grad}(t; \{w_i\}) = \sum_{i=1}^{T} \left| G_W^{(i)}(t) - G_W^{target}(t) \right|$$


After each update to $w_i$, the weights are renormalized so $\sum_i w_i = T$ (preserving total loss scale).

**Pseudocode.**

```python
# GradNorm Training Step
# Inputs: model, tasks, alpha hyperparameter
# W = parameters of the shared last layer

L0 = [compute_loss(task_i, model) for task_i in tasks]  # initial losses

for step in training_steps:
    losses = [compute_loss(task_i, model) for task_i in tasks]
    L_total = sum(w[i] * losses[i] for i in range(T))

    # Standard backward for network parameters
    optimizer_net.zero_grad()
    L_total.backward(retain_graph=True)

    # Compute gradient norms at shared layer W
    G = [grad_norm(W, losses[i], w[i]) for i in range(T)]
    G_bar = mean(G)

    # Relative inverse training rates
    L_tilde = [losses[i] / L0[i] for i in range(T)]
    L_tilde_bar = mean(L_tilde)

    # Target norms
    G_target = [G_bar * (L_tilde[i] / L_tilde_bar) ** alpha for i in range(T)]

    # GradNorm loss (stop gradient through G_target)
    L_grad = sum(abs(G[i] - stop_gradient(G_target[i])) for i in range(T))

    # Update task weights
    optimizer_w.zero_grad()
    L_grad.backward()
    optimizer_w.step()

    # Renormalize weights
    w = w / sum(w) * T

    optimizer_net.step()
```

**Pros and Cons.**
- Pros: Simple to implement; prevents any single task from dominating; well-validated on vision benchmarks.
- Cons: Uses gradient norms at a single shared layer as a proxy — not per-parameter direction balancing; sensitive to the choice of $\alpha$ and initialization of $w_i$; does not directly address gradient direction conflicts.

---

### 2.2 PCGrad / Gradient Surgery (NeurIPS 2020)

**Intuition.** Picture two hikers trying to climb different hills. If they are roped together, one pulling north while the other pulls south, they cancel each other out. PCGrad (Projecting Conflicting Gradients) cuts the rope and instead lets each hiker only keep the component of their pull that does not drag the other backward [^2]. Geometrically, if task $i$'s gradient has a negative projection onto task $j$'s gradient, that conflicting component is removed.

**Math.** For two tasks $i$ and $j$, a conflict exists when:


$$\mathbf{g}_i \cdot \mathbf{g}_j < 0$$


In this case, project out the component of $\mathbf{g}_i$ that lies along $\mathbf{g}_j$:


$$\mathbf{g}_i^{PC} = \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$


The subtracted term $\frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$ is exactly the component of $\mathbf{g}_i$ in the direction of $\mathbf{g}_j$. When the dot product is negative, this component points anti-parallel to $\mathbf{g}_j$, so removing it prevents $\mathbf{g}_i$ from harming task $j$.

Why does this work geometrically? Recall that $\mathbf{g}_i \cdot \mathbf{g}_j = \|\mathbf{g}_i\|\|\mathbf{g}_j\|\cos\theta$. When $\theta > 90°$, the projection of $\mathbf{g}_i$ onto $\hat{\mathbf{g}}_j$ is negative. After surgery, the modified $\mathbf{g}_i^{PC}$ satisfies:


$$\mathbf{g}_i^{PC} \cdot \mathbf{g}_j = \mathbf{g}_i \cdot \mathbf{g}_j - \frac{(\mathbf{g}_i \cdot \mathbf{g}_j)(\mathbf{g}_j \cdot \mathbf{g}_j)}{\|\mathbf{g}_j\|^2} = \mathbf{g}_i \cdot \mathbf{g}_j - \mathbf{g}_i \cdot \mathbf{g}_j = 0$$


So the modified gradient is orthogonal to $\mathbf{g}_j$ — it no longer conflicts.

For $T$ tasks, the procedure applies pairwise: for task $i$, iterate over all $j \neq i$ and sequentially project out each conflicting direction. The final update is:


$$\mathbf{g}^{final} = \sum_{i=1}^{T} \mathbf{g}_i^{PC}$$


**Pseudocode.**

```python
# PCGrad (Gradient Surgery)
def pcgrad(gradients):
    # gradients: list of T gradient vectors, one per task
    T = len(gradients)
    projected = [g.clone() for g in gradients]

    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            dot = torch.dot(projected[i], gradients[j])
            if dot < 0:
                # Remove conflicting component
                projected[i] -= (dot / (gradients[j].norm() ** 2)) * gradients[j]

    return sum(projected)

# In training loop:
for batch in dataloader:
    grads = [compute_gradient(loss_i, model) for loss_i in task_losses]
    combined_grad = pcgrad(grads)
    apply_gradient(model, combined_grad)
```

**Complexity.** The pairwise loop is $O(T^2)$ in gradient computations and $O(T^2 \cdot d)$ in memory where $d$ is the parameter dimension. This becomes prohibitive for $T > 10$ tasks.

**Pros and Cons.**
- Pros: Principled geometric solution; no scalar hyperparameters; prevents destructive interference directly.
- Cons: $O(T^2)$ complexity is unscalable; only addresses gradient direction, not magnitude imbalance; order of pairwise projections can matter.

---

### 2.3 MGDA — Multiple Gradient Descent Algorithm (NeurIPS 2018)

**Intuition.** MGDA reframes multi-task learning as a **multi-objective optimization** (MOO) problem [^3]. Rather than combining gradients with fixed weights, it searches for a descent direction that simultaneously reduces all task losses — a Pareto-improving step. Think of it as finding the direction in gradient space that every task can agree on.

**Math.** A point $\theta$ is **Pareto-stationary** if no single update direction can improve all objectives simultaneously. MGDA finds the minimum-norm point in the convex hull of task gradients:


$$\mathbf{g}^* = \sum_{i=1}^{T} \alpha_i \mathbf{g}_i, \quad \alpha_i \geq 0, \quad \sum_{i=1}^{T} \alpha_i = 1$$



$$\{\alpha_i^*\} = \arg\min_{\alpha} \left\| \sum_{i=1}^{T} \alpha_i \mathbf{g}_i \right\|^2$$


If $\mathbf{g}^* = \mathbf{0}$, the current point is already Pareto-stationary. Otherwise, taking a step in the direction $-\mathbf{g}^*$ guarantees a reduction in every task loss (to first order).

The optimization over $\{\alpha_i\}$ is a quadratic program (QP) over the simplex, solvable efficiently with the Frank-Wolfe algorithm or the dual form. The computational cost is $O(T^2 \cdot d)$ per step.

**Pros and Cons.**
- Pros: Principled MOO framework; theoretical convergence guarantees to Pareto-stationary points; no task weighting hyperparameters.
- Cons: Can stall at poor Pareto points (e.g., a saddle in the objectives); expensive QP per step; the Pareto-stationary criterion may not align with practical task priorities.

---

### 2.4 CAGrad (NeurIPS 2021)

**Intuition.** CAGrad (Conflict-Averse Gradient Descent) unifies vanilla gradient descent and MGDA by introducing a tunable constraint. The key insight is: rather than minimizing all tasks equally (MGDA) or ignoring conflicts (plain sum), CAGrad maximizes the *worst-case* task improvement while staying close to the average gradient. This is the multi-task analogue of max-min fairness [^4].

**Math.** Let the average gradient be:


$$\mathbf{g}_0 = \frac{1}{T} \sum_{i=1}^{T} \mathbf{g}_i$$


CAGrad's primal problem at each step is:


$$\mathbf{d}^* = \arg\max_{\mathbf{d}} \min_{i \in [T]} \langle \mathbf{g}_i, \mathbf{d} \rangle \quad \text{s.t.} \quad \|\mathbf{d} - \mathbf{g}_0\| \leq c \|\mathbf{g}_0\|$$


The constraint ball of radius $c \|\mathbf{g}_0\|$ centered at $\mathbf{g}_0$ controls how far from the average gradient we are allowed to deviate:
- $c = 0$: The only feasible point is $\mathbf{g}_0$ itself — recovers vanilla gradient descent.
- $c \to \infty$: The constraint is vacuous and the solution recovers MGDA (min-norm in convex hull).

Rather than solving the primal directly, the paper derives a **dual problem** over task weights $\mathbf{w}$ on the probability simplex $\mathcal{W} = \{\mathbf{w} : \sum_i w_i = 1,\ w_i \geq 0\}$. Let $\mathbf{g}_\mathbf{w} = \sum_{i=1}^T w_i \mathbf{g}_i$ and $\varphi = c^2 \|\mathbf{g}_0\|^2$. The dual objective is:


$$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \mathcal{W}}\ F(\mathbf{w}) := \mathbf{g}_\mathbf{w}^\top \mathbf{g}_0 + \sqrt{\varphi}\, \|\mathbf{g}_\mathbf{w}\|$$


The optimal Lagrange multiplier is $\lambda^* = \|\mathbf{g}_{\mathbf{w}^*}\| / \sqrt{\varphi}$, and the final parameter update direction combines the average gradient with a scaled weighted component:


$$\mathbf{d}^* = \mathbf{g}_0 + \frac{\sqrt{\varphi}}{\|\mathbf{g}_{\mathbf{w}^*}\|}\, \mathbf{g}_{\mathbf{w}^*}$$


**Pseudocode.**

```python
import torch

def cagrad(gradients, c=0.5, num_iters=20, lr_w=0.01):
    """
    CAGrad: Conflict-Averse Gradient Descent (Liu et al., NeurIPS 2021)
    Solves the dual: min_{w in simplex} g_w.T @ g0 + sqrt(phi) * ||g_w||
    Returns the update direction d* = g0 + (sqrt(phi)/||g_w*||) * g_w*
    """
    T = len(gradients)
    # Stack gradients: shape [T, d]
    G = torch.stack(gradients)               # [T, d]
    g0 = G.mean(dim=0)                       # average gradient
    phi = (c * g0.norm()) ** 2               # c^2 * ||g0||^2
    sqrt_phi = phi ** 0.5

    # Initialize uniform weights on the simplex
    w = torch.full((T,), 1.0 / T, requires_grad=True)

    optimizer_w = torch.optim.SGD([w], lr=lr_w)

    for _ in range(num_iters):
        optimizer_w.zero_grad()

        w_simplex = torch.softmax(w, dim=0)  # project onto simplex
        g_w = (w_simplex.unsqueeze(1) * G).sum(dim=0)  # weighted gradient

        # Dual objective: g_w.T @ g0 + sqrt(phi) * ||g_w||
        loss_dual = g_w @ g0 + sqrt_phi * g_w.norm()
        loss_dual.backward()
        optimizer_w.step()

    with torch.no_grad():
        w_star = torch.softmax(w, dim=0)
        g_w_star = (w_star.unsqueeze(1) * G).sum(dim=0)
        # Final update direction
        d_star = g0 + (sqrt_phi / (g_w_star.norm() + 1e-8)) * g_w_star

    return d_star

# In training loop:
grads = [compute_gradient(loss_i, model) for loss_i in task_losses]
update = cagrad(grads, c=0.5)
apply_gradient(model, update)
```

**Pros and Cons.**
- Pros: Unifies vanilla GD and MGDA under one framework; convergence to a stationary point of the average loss is guaranteed (Theorem 3.2 in the paper); $c$ gives intuitive control.
- Cons: Requires tuning $c \in [0, 1)$; solving the dual subproblem per step adds overhead; for very large $T$, the per-step cost grows.

---

### 2.5 MetaBalance (WWW 2022)

**Intuition.** In recommendation systems, there is often a natural hierarchy: a primary task (e.g., CTR) that we care most about, and auxiliary tasks (e.g., dwell time, likes) that provide helpful regularization. MetaBalance takes this hierarchy seriously and rescales auxiliary gradients to match the primary task's gradient norm — layer by layer, iteration by iteration [^5].

**Math.** Let $\mathbf{G}_{tar}$ be the target task gradient and $\mathbf{G}_{aux,i}$ the $i$-th auxiliary gradient, both w.r.t. a shared parameter block $\theta$.

**Step 1 — Exponential moving averages of gradient norms** (Algorithm 2, the complete version):


$$m_{tar}^t = \beta\, m_{tar}^{t-1} + (1-\beta)\,\|\mathbf{G}_{tar}^t\|$$

$$m_{aux,i}^t = \beta\, m_{aux,i}^{t-1} + (1-\beta)\,\|\mathbf{G}_{aux,i}^t\|$$


with $\beta = 0.9$ and $m^0 = 0$. Using moving averages instead of instantaneous norms removes noise from per-batch gradient variance.

**Step 2 — Strategy selection.** Choose which auxiliary gradients to rescale:

- **Strategy A (Reduce)**: rescale only when $m_{aux,i}^t > m_{tar}^t$ — the auxiliary is overpowering the target.
- **Strategy B (Enlarge)**: rescale only when $m_{aux,i}^t < m_{tar}^t$ — the auxiliary is too weak to transfer useful signal.
- **Strategy C (Flexible)**: rescale always (A ∪ B). Empirically best across benchmarks.

**Step 3 — Gradient rescaling with relax factor $r$.** Rather than hard norm-matching (which forces $\|\mathbf{G}_{aux,i}\| = \|\mathbf{G}_{tar}\|$ and over-suppresses useful signal), MetaBalance uses a partial interpolation:


$$\mathbf{G}_{aux,i}^t \leftarrow \mathbf{G}_{aux,i}^t \cdot \underbrace{\left[\left(\frac{m_{tar}^t}{m_{aux,i}^t} - 1\right) r + 1\right]}_{w_{aux,i}^t}$$


The effective dynamic weight $w_{aux,i}^t$ has two intuitive limits:
- $r = 0$: $w_{aux,i}^t = 1$, no rescaling (vanilla multi-task).
- $r = 1$: $w_{aux,i}^t = m_{tar}^t / m_{aux,i}^t$, full norm-matching.

Intermediate $r$ (e.g., $r = 0.7$) balances convergence speed and stability. Since gradient imbalances vary across layers, MetaBalance applies this independently **per parameter block** $\theta$ of the shared network.

**Pseudocode (Algorithm 2 — complete version).**

```python
# MetaBalance: Complete version with moving averages and relax factor
# Applied independently per shared parameter block (e.g., each MLP layer)

def metabalance_update(model, target_loss, aux_losses, optimizer,
                       r=0.7, beta=0.9, m_tar=None, m_aux=None, strategy='C'):
    K = len(aux_losses)

    # Lazily initialize moving average accumulators per layer
    if m_tar is None:
        m_tar  = {name: 0.0 for name, _ in model.shared_params()}
        m_aux  = [{name: 0.0 for name, _ in model.shared_params()} for _ in range(K)]

    # --- Compute target gradients ---
    optimizer.zero_grad()
    target_loss.backward(retain_graph=True)
    G_tar = {name: p.grad.clone() for name, p in model.shared_params()}

    # Update target moving average per layer
    for name in G_tar:
        m_tar[name] = beta * m_tar[name] + (1 - beta) * G_tar[name].norm().item()

    # --- Compute and rescale auxiliary gradients ---
    G_aux_total = {name: torch.zeros_like(G_tar[name]) for name in G_tar}

    for i, aux_loss in enumerate(aux_losses):
        optimizer.zero_grad()
        aux_loss.backward(retain_graph=True)

        for name, p in model.shared_params():
            G_i = p.grad.clone()
            norm_i = G_i.norm().item()

            # Update auxiliary moving average
            m_aux[i][name] = beta * m_aux[i][name] + (1 - beta) * norm_i

            # Apply strategy condition using moving averages
            m_t = m_tar[name]
            m_a = m_aux[i][name] + 1e-8

            apply = (strategy == 'A' and m_a > m_t) or \
                    (strategy == 'B' and m_a < m_t) or \
                    (strategy == 'C')

            if apply:
                # Eq. (4): dynamic weight with relax factor
                w = (m_t / m_a - 1.0) * r + 1.0
                G_i = G_i * w

            G_aux_total[name] += G_i

    # --- Combine and apply ---
    optimizer.zero_grad()
    for name, p in model.shared_params():
        p.grad = G_tar[name] + G_aux_total[name]
    optimizer.step()

    return m_tar, m_aux  # pass back for next iteration
```

**Pros and Cons.**
- Pros: Per-layer per-step control; moving averages stabilize rescaling; single hyperparameter $r$ regardless of number of auxiliary tasks; optimizer-agnostic (works with Adam, Adagrad, etc.); validated on Alibaba datasets (+8.34% NDCG@10 [^5]).
- Cons: Requires an explicit primary/auxiliary task hierarchy; per-layer backward passes add compute overhead; does not address gradient direction conflicts, only magnitude.

---

### 2.6 SLGrad (2023)

**Intuition.** Even within a single auxiliary task, different training samples contribute differently to the primary objective. Some auxiliary samples genuinely help the primary task; others introduce noise or conflict. SLGrad operates at the finest possible granularity — individual samples — assigning weights based on how well each sample's gradient aligns with the primary task's validation gradient [^6].

**Math.** Let $\mathbf{g}_{val}^{primary}$ be the gradient of the primary task on a held-out validation set. For each sample $s$ in the auxiliary task batch, compute the cosine similarity:


$$w_s = \cos(\mathbf{g}_s, \mathbf{g}_{val}^{primary}) = \frac{\mathbf{g}_s \cdot \mathbf{g}_{val}^{primary}}{\|\mathbf{g}_s\| \|\mathbf{g}_{val}^{primary}\|}$$


Samples with $w_s > 0$ push in a direction compatible with the primary objective; samples with $w_s < 0$ conflict. SLGrad reweights the auxiliary loss by $\max(0, w_s)$ or uses $w_s$ directly as a soft weight to suppress conflicting samples.

**Pros and Cons.**
- Pros: Finest possible granularity; directly suppresses noisy auxiliary samples; interpretable per-sample weights.
- Cons: Requires a separate validation set for the primary task; computing per-sample gradients is expensive (requires multiple backward passes or per-sample gradient approximations); not easily scalable to large batch sizes.

---

### 2.7 PLE — Progressive Layered Extraction (RecSys 2020)

**Intuition.** Rather than fixing gradient conflicts after the fact, PLE prevents them architecturally [^7]. It partitions model capacity into shared experts and task-specific experts, organized in hierarchical layers. Task-specific experts receive gradients only from their own task, eliminating cross-task gradient interference at the source.

**Math.** Let $L$ be the number of extraction layers. At layer $l$:
- Shared experts $\{E_s^{(l)}\}$: receive gradients from all tasks via gating.
- Task-specific experts $\{E_k^{(l)}\}$ for task $k$: receive gradients only from task $k$.

The output for task $k$ at layer $l$ is:


$$\mathbf{h}_k^{(l)} = \sum_j g_{k,j}^{(l)} E_j^{(l)}(\mathbf{h}_k^{(l-1)})$$


where $g_{k,j}^{(l)}$ are gating weights that blend shared and task-specific expert outputs. Crucially, the gating ensures that gradients flowing backward through task-specific experts are isolated from other tasks.

**Pros and Cons.**
- Pros: Structural separation eliminates gradient conflict by design; well-validated in large-scale production settings; no optimization-time overhead.
- Cons: Increased parameter count; the static expert structure may not adapt to shifting task relationships; requires careful capacity allocation.

---

### 2.8 STEM-Net (AAAI 2024)

**Intuition.** Many multi-task seesaw effects originate not in the shared tower but in the shared embedding table. If a user who is a heavy clicker but a rare purchaser updates the same embedding with conflicting signals, the embedding itself becomes a source of conflict. STEM-Net separates embeddings at the task level and uses a stop-gradient gate to prevent backward leakage between task-specific embedding paths [^8].

**Math.** For each task $i$, STEM-Net maintains:
- A task-specific embedding table $\mathbf{E}_i$
- A globally shared embedding table $\mathbf{E}_{shared}$

The final representation for task $i$ is:


$$\mathbf{h}_i = f_i\!\left(\mathbf{E}_i(\mathbf{x}),\ \text{stop\_grad}\!\left(\mathbf{E}_{shared}(\mathbf{x})\right)\right)$$


The **All Forward Task-specific Backward (AFTB)** gating mechanism allows all embeddings to contribute during the forward pass (maximizing information flow) while blocking gradient flow from task $i$'s loss to task $j$'s embedding table (preventing conflict). This is implemented via `stop_gradient` operators that zero out gradients selectively.

STEM-Net is particularly effective for **comparable samples** — users with balanced multi-task feedback (e.g., users who both click and purchase). For such users, conflicting embedding gradients are most damaging.

**Pros and Cons.**
- Pros: Resolves conflicts at the embedding source; significant seesaw-effect reduction in Tencent production; AFTB gating is elegant and effective.
- Cons: Multiplicative memory cost (one embedding table per task plus shared); not a plug-and-play addition to existing architectures; the benefit concentrates on "comparable samples".

---

### 2.9 MultiBalance (Meta, 2024)

**Intuition.** GradNorm and PCGrad operate on the full parameter gradient $\mathbf{g}_i \in \mathbb{R}^d$ where $d$ can be in the billions. This is computationally expensive and often noisy. MultiBalance observes that in representation-based recommenders, all task heads attach to a shared bottleneck $\mathbf{h}$. Balancing gradients at the bottleneck — a much smaller vector — achieves most of the benefit at a fraction of the cost [^9].

**Math.** MultiBalance has two components: (1) a copy-and-fork architecture to obtain per-task representation gradients in a single backward pass, and (2) an MGDA-style min-norm balancing with EMA stabilization.

**Step 1 — EMA-rescaled representation gradients.** In the forward pass, the shared representation $\Phi$ is copied $M$ times — one per task head. A single backward pass then yields $M$ separate representation gradients $g_{k+1,m} = \nabla_\Phi f_m$. These are stabilized with an exponential moving average of their norms:


$$u_{k+1,m} = (1 - \gamma)\, u_{k,m} + \gamma\, \|g_{k+1,m}\|$$


The EMA-rescaled gradient (smoothed direction at historical average magnitude) is:


$$v_{k+1,m} = \frac{u_{k+1,m}}{\|g_{k+1,m}\|}\, g_{k+1,m}$$


Without this rescaling, noisy per-batch spikes in gradient norms destabilize the balancing weights.

**Step 2 — MGDA min-norm objective over representation gradients.** Let $V_{k+1}$ be the matrix stacking $\{v_{k+1,m}\}$. MultiBalance solves the surrogate min-norm problem (MGDA at the representation level):


$$\min_{\lambda \in \Delta^M} \|V_{k+1}\,\lambda\|$$


Theorem 1 in the paper shows this cheap $O(M^2 n')$ objective (where $n'$ is representation dim) guarantees approximate Pareto stationarity at the full parameter level, avoiding the infeasible $O(M^2 n)$ cost of parameter-space MGDA (where $n \sim 10^9$).

**Step 3 — λ update via projected gradient descent with prior regularization:**


$$\lambda_{k+1} = \Pi_{\Delta^M}\!\left(\lambda_k - \beta_k\, V_{k+1}^\top \bigl(V_{k+1}\lambda_k + \rho\, V_{k+1}\lambda_0\bigr)\right)$$


where $\lambda_0 = \frac{1}{M}\mathbf{1}$ is a uniform prior, $\rho = 0.1$ regularizes toward equal weighting, and $\Pi_{\Delta^M}$ projects onto the probability simplex. The final balanced gradient $V_{k+1}\lambda_{k+1}$ is injected back into the autograd graph and propagated through the shared bottom layers.

**Pseudocode (Algorithm 3).**

```python
# MultiBalance: MGDA-style balancing at the representation level
# Single backward pass via copy-and-fork architecture

def multibalance_step(model, task_losses, optimizer,
                      u, lam, rho=0.1, gamma=0.01, lr_lam=1.0):
    """
    u   : list of M EMA norm scalars (initialized to 0)
    lam : task weight vector on simplex (initialized to 1/M each)
    """
    M = len(task_losses)

    # --- Forward: shared representation copied M times ---
    h = model.shared_bottom(inputs)            # [batch, n']
    h_copies = [h.clone().requires_grad_(True) for _ in range(M)]

    # --- Single backward: get M representation gradients ---
    raw_grads = []
    for m, (h_m, loss_fn) in enumerate(zip(h_copies, task_losses)):
        pred = model.task_heads[m](h_m)
        loss = loss_fn(pred, targets[m])
        loss.backward()
        raw_grads.append(h_m.grad.detach())    # g_{k+1,m}: [batch, n']

    # --- EMA rescaling ---
    v = []
    for m in range(M):
        norm_m = raw_grads[m].norm().item()
        u[m] = (1 - gamma) * u[m] + gamma * norm_m
        v_m = (u[m] / (norm_m + 1e-8)) * raw_grads[m]
        v.append(v_m)
    V = torch.stack(v)                         # [M, batch, n']

    # --- MGDA update for lambda (one projected gradient step) ---
    with torch.no_grad():
        Vlam = (lam.view(M, 1, 1) * V).sum(0)         # V @ lam: [batch, n']
        lam0 = torch.full((M,), 1.0 / M)
        Vlam0 = (lam0.view(M, 1, 1) * V).sum(0)       # V @ lam0

        # Gradient of objective w.r.t. lambda (use cosine similarity in practice)
        grad_lam = torch.stack([(V[m] * (Vlam + rho * Vlam0)).sum() for m in range(M)])
        lam = lam - lr_lam * grad_lam
        # Project onto simplex
        lam = simplex_projection(lam)

    # --- Inject balanced gradient back into shared trunk ---
    balanced_rep_grad = (lam.view(M, 1, 1) * V).sum(0)  # [batch, n']
    optimizer.zero_grad()
    h_final = model.shared_bottom(inputs)
    h_final.backward(balanced_rep_grad)        # single pass through shared params
    optimizer.step()

    return u, lam
```

**Pros and Cons.**
- Pros: Only ~0.42% QPS overhead vs. 68–82% for parameter-space MGDA; single backward pass via copy-and-fork; EMA prevents norm spike instability; theoretically grounded (Theorem 1 bounds parameter-level Pareto gap); validated at Meta on Feeds and Ads models.
- Cons: Balancing is confined to the shared representation bottleneck — conflicts inside task-specific subnetworks are not addressed; the copy-and-fork architecture requires refactoring existing model code; $\rho$ and $\gamma$ are additional hyperparameters.

---

### 2.10 GradCraft (KDD 2024)

**Intuition.** Previous methods address either magnitude (GradNorm, MetaBalance) or direction (PCGrad, MGDA) — but rarely both simultaneously. GradCraft takes a holistic two-stage approach: first equalize gradient magnitudes, then enforce that all pairs of adjusted gradients have non-negative dot products (global synergy). This combines the benefits of norm balancing and conflict resolution in a single unified procedure [^10].

**Math.** **Stage 1 (Magnitude Balancing).** Rescale each task gradient to match the maximum norm:


$$\mathbf{g}_i^{mag} = \mathbf{g}_i \cdot \frac{\max_j \|\mathbf{g}_j\|}{\|\mathbf{g}_i\|}$$


After this step, all $\|\mathbf{g}_i^{mag}\| = \max_j \|\mathbf{g}_j\|$ — a level playing field.

**Stage 2 (Direction Alignment).** Enforce global pairwise non-negativity: find adjusted gradients $\{\mathbf{g}_i^{final}\}$ such that:


$$\langle \mathbf{g}_i^{final}, \mathbf{g}_j^{final} \rangle \geq 0 \quad \forall i, j$$


Unlike PCGrad which handles this pairwise (and can create inconsistencies), GradCraft solves a global optimization problem to find the smallest adjustments to $\{\mathbf{g}_i^{mag}\}$ that achieve universal non-negativity. This is a more expensive but more internally consistent operation.

**Pseudocode.**

```python
# GradCraft
def gradcraft(gradients):
    T = len(gradients)

    # Stage 1: Magnitude balancing
    norms = [g.norm() for g in gradients]
    max_norm = max(norms)
    g_mag = [g * (max_norm / n) for g, n in zip(gradients, norms)]

    # Stage 2: Direction alignment
    # Iteratively project conflicting pairs until all pairs are synergistic
    g_final = [g.clone() for g in g_mag]
    converged = False
    while not converged:
        converged = True
        for i in range(T):
            for j in range(i + 1, T):
                dot = torch.dot(g_final[i], g_final[j])
                if dot < 0:
                    converged = False
                    # Project both toward each other to resolve conflict
                    norm_j_sq = g_final[j].norm() ** 2
                    norm_i_sq = g_final[i].norm() ** 2
                    g_i_new = g_final[i] - (dot / norm_j_sq) * g_final[j]
                    g_j_new = g_final[j] - (dot / norm_i_sq) * g_final[i]
                    g_final[i] = g_i_new
                    g_final[j] = g_j_new

    return sum(g_final)
```

**Pros and Cons.**
- Pros: Addresses both magnitude and direction; global synergy constraint is stronger than pairwise PCGrad; peer-reviewed and published at KDD 2024.
- Cons: Iterative Stage 2 is computationally expensive; convergence not always guaranteed in the iterative approach; $O(T^2)$ loop in Stage 2.

---

### 2.11 PUB — Parameter Update Balancing (2024)

**Intuition.** All gradient balancing methods so far operate on raw gradients $\mathbf{g}_i$. But in practice, models are trained with Adam, not vanilla SGD. Adam's adaptive moment estimates transform gradients before applying them: $\mathbf{g}_i^{Adam} \neq \mathbf{g}_i$. PUB makes a critical observation: **gradient balancing does not equal update balancing under Adam**. Two tasks with balanced gradients can still have highly imbalanced actual parameter updates once Adam applies its per-parameter learning rates. PUB closes this "momentum gap" by balancing the actual Adam updates, not the raw gradients [^11].

**Math.** For each task $i$, compute the Adam-adjusted update:


$$\Delta\theta_i = \text{Adam}(\mathbf{g}_i, m_i, v_i) = \frac{\hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$


where $m_i, v_i$ are the first and second moment estimates for task $i$'s gradients. PUB then finds the joint update direction:


$$\Delta\theta^* = \arg\max_{\Delta\theta} \min_{i \in [T]} \langle \Delta\theta_i, \Delta\theta \rangle \quad \text{s.t.} \quad \Delta\theta \in \text{conv}(\{\Delta\theta_i\})$$


This is the CAGrad/MGDA framework applied to Adam updates rather than raw gradients. The constraint that $\Delta\theta$ lies in the convex hull of task updates ensures the final step is a mixture of individually valid Adam steps.

**Pseudocode.**

```python
# PUB: Parameter Update Balancing (closes the momentum gap)
def pub_update(model, task_losses, optimizer):
    T = len(task_losses)
    task_updates = []

    # Step 1: Compute per-task Adam parameter updates
    for i, loss in enumerate(task_losses):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Simulate Adam update (without applying it)
        updates = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                # Adam-adjusted update: m / (sqrt(v) + eps)
                m = optimizer.state[p].get('exp_avg', p.grad.clone())
                v = optimizer.state[p].get('exp_avg_sq', p.grad.pow(2))
                adam_update = m / (v.sqrt() + 1e-8)
                updates[name] = adam_update
        task_updates.append(updates)

    # Step 2: Solve convex program to find consensus update
    # Maximize: min_i < delta_theta_i, delta_theta >
    # subject to: delta_theta in convex hull of {delta_theta_i}
    # (Frank-Wolfe or quadratic program)
    alphas = solve_min_norm_problem(task_updates)  # returns convex weights

    # Step 3: Apply weighted consensus update
    optimizer.zero_grad()
    for name, p in model.named_parameters():
        if any(name in u for u in task_updates):
            p.grad = sum(alphas[i] * task_updates[i][name] for i in range(T))

    optimizer.step()
```

**Pros and Cons.**
- Pros: First method to account for adaptive optimizers; more robust than PCGrad/CAGrad under Adam; theoretically principled momentum-aware balancing.
- Cons: Requires maintaining separate moment estimates per task (memory overhead); the inner optimization subproblem is more expensive than standard gradient balancing; limited large-scale industrial validation to date.

---

### 2.12 DRGrad (AAAI 2025)

**Intuition.** Even within a single task, different users generate gradients with varying compatibility with the primary objective. DRGrad operates at the finest recommendation-specific granularity: individual users [^12]. For each user, it routes gradients from auxiliary tasks only when they are likely to be cooperative with the primary task's objective for that specific user.

**Math.** DRGrad consists of three components:

**Router**: Estimates the compatibility of auxiliary task gradients for a given user $u$:


$$r_u^{(i)} = \sigma(W_r \mathbf{h}_u + b_r)$$


where $\mathbf{h}_u$ is the user representation. A high $r_u^{(i)}$ means task $i$'s gradient is likely helpful for user $u$.

**Updater**: Computes a balanced gradient using only the routed (cooperative) gradients:


$$\mathbf{g}_u^{balanced} = \mathbf{g}_u^{primary} + \sum_{i \neq \text{primary}} r_u^{(i)} \cdot \mathbf{g}_u^{(i)}$$


**Personalized Gate Network**: Learns per-user routing policies that adapt over training, allowing the system to identify which auxiliary tasks benefit which user segments.

**Pros and Cons.**
- Pros: Most fine-grained method available (user-level gradient routing); directly addresses the heterogeneous user population in recommendation.
- Cons: Complex architecture with multiple learned components; computationally expensive; limited large-scale industrial validation compared to simpler alternatives.

---

### 2.13 Meta Lattice (arXiv 2024)

**Full name.** Lattice — Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations (Meta, 2024).

**Intuition.** Large recommendation platforms often maintain separate models for different domains (e.g., Feed, Stories, Marketplace) and different objectives (CTR, CVR, revenue). This creates infrastructure fragmentation and prevents cross-domain knowledge sharing. Meta Lattice addresses Multi-Domain, Multi-Objective (MDMO) learning by consolidating fragmented model portfolios onto a single unified foundational network, rather than patching gradient conflicts after they arise [^13].

**Approach.** Lattice introduces two key components:

- **Lattice Filter**: Performs Pareto-optimal feature selection across domains. Rather than using all features for all domains, it identifies the feature subset that is most informative across the full portfolio of domain-objective pairs, reducing noise and redundancy.

- **Lattice Zipper**: Mixes various attribution windows for different objectives. Different objectives (e.g., immediate click vs. 7-day conversion) observe user behavior at different time horizons. The Zipper component blends these temporal views appropriately per objective.

The result is a single network that serves all domains and objectives, eliminating the need for separate per-domain or per-objective models.

**Results.** Meta reports 10% revenue gain and 20% capacity saving in production deployment [^13].

**Pros and Cons.**
- Pros: Structural unification of multi-domain and multi-objective learning; large-scale business impact validated at Meta production scale; addresses the broader MDMO problem, not just pairwise task gradient conflicts.
- Cons: Major infrastructure overhaul — not a drop-in replacement; requires multi-domain data alignment and coordinated feature engineering; organizational changes needed to maintain a unified model portfolio.

---

### 2.14 Other Notable Methods

**ConicGrad**: Constrains the combined gradient to lie inside an angular cone around the average gradient. Provides a softer version of MGDA's convex hull constraint.

**SAM-GS (Similarity-Aware Momentum Gradient Surgery)**: Extends PCGrad with momentum estimates, using gradient similarity as a continuous weight rather than a binary conflict flag. Reduces the aggressiveness of projection for near-orthogonal gradients.

**MTGR (Meituan Generative Recommendation)**: Combines multi-task gradient balancing with a generative recommendation backbone (DLRM-style), demonstrating that gradient balancing principles extend to sequence-based recommendation models.

---

## 3. Comparison Table

| Method | Year | Granularity | Direction? | Magnitude? | Scalable? | Key Insight |
|--------|------|-------------|-----------|-----------|-----------|-------------|
| MGDA | 2018 | Gradient | Yes | Partially | No | Pareto-stationary descent |
| GradNorm | 2018 | Loss | No | Yes | Yes | Match gradient norms to learning rates |
| PCGrad | 2020 | Gradient | Yes | No | No ($O(T^2)$) | Project out conflicting components |
| PLE | 2020 | Architecture | Yes | No | Yes | Structural separation of expert gradients |
| CAGrad | 2021 | Gradient | Yes | No | Moderate | Max-min fairness with proximity constraint |
| MetaBalance | 2022 | Per-layer gradient | Partially | Yes | Yes | Auxiliary aligns to primary per layer |
| SLGrad | 2023 | Per-sample gradient | Yes | No | No | Sample-level alignment to validation gradient |
| STEM-Net | 2024 | Embedding | Yes | No | Moderate | Stop-gradient at embedding level (AFTB) |
| MultiBalance | 2024 | Representation | Partially | Yes | Yes | Balance at bottleneck, single backward pass |
| GradCraft | 2024 | Gradient | Yes | Yes | No | Holistic two-stage balancing |
| PUB | 2024 | Adam update | Yes | Yes | Moderate | Balance updates, not raw gradients |
| DRGrad | 2025 | Per-user gradient | Yes | Partially | No | User-level gradient routing |

---

## 4. How to Choose

Selecting the right gradient balancing method depends on your system constraints, task relationships, and available engineering resources. Here is a decision guide:

- **Do you have more than ~5 tasks and need scalability?**
    - Yes:
        - **Is there a clear primary task with auxiliary helpers?**
            - Yes: Use **MetaBalance** (per-layer, primary-centric, scalable)
            - No: Use **MultiBalance** (single backward pass, representation-level, near-zero overhead)
        - **Are you limited to loss-level changes (no gradient access)?**
            - Yes: Use **GradNorm** (only requires per-task loss values)
    - No (small T, research or experimentation setting):
        - **Do you care about both direction and magnitude?**
            - Yes: Use **GradCraft** (holistic) or **PUB** (if using Adam)
            - No, direction only: Use **PCGrad** (simple, interpretable) or **CAGrad** (if you want worst-task guarantees)
            - No, magnitude only: Use **GradNorm** or **MetaBalance**

- **Is the conflict rooted in the embedding layer?**
    - Yes: Use **STEM-Net** (AFTB gating at embedding source)
    - No: Proceed to gradient-level methods above

- **Do you want structural rather than optimization-time solutions?**
    - Yes: Use **PLE** (architectural separation of expert pathways)

- **Do you have heterogeneous users with different task affinities?**
    - Yes: Consider **DRGrad** (user-level routing) or **SLGrad** (sample-level reweighting)

- **Are you using Adam (not SGD)?**
    - Yes: Strongly consider **PUB** — gradient balancing without accounting for Adam's moment estimates may be ineffective

- **Are you in a production system with strict latency/QPS constraints?**
    - Use **MultiBalance** (no additional backward passes) or **MetaBalance** (overhead is per-layer norm computation only)
    - Avoid **PCGrad**, **GradCraft**, **SLGrad**, **DRGrad** in latency-critical settings

---

## 5. Conclusion

Gradient balancing in multi-task recommendation has evolved from coarse loss-level reweighting (GradNorm, 2018) to fine-grained per-user routing (DRGrad, 2025). Each generation of methods addresses a distinct failure mode:

- **First generation** (GradNorm): Fix magnitude imbalance by adjusting task weights dynamically.
- **Second generation** (PCGrad, MGDA, CAGrad): Fix direction conflicts through geometric operations on gradient vectors.
- **Recommendation-specific** (MetaBalance, SLGrad, STEM-Net): Exploit the primary/auxiliary task structure and the embedding-centric nature of recommenders.
- **Scalable industrial** (PLE, MultiBalance): Prioritize production feasibility — structural solutions or single-pass approximations.
- **Holistic and adaptive** (GradCraft, PUB, DRGrad): Address both magnitude and direction simultaneously, or close gaps introduced by adaptive optimizers and user heterogeneity.

No single method dominates across all settings. The right choice depends on task count, available compute, whether a primary/auxiliary hierarchy exists, and the specific failure mode (magnitude, direction, or both). The comparison table and decision guide above should help practitioners map their constraints to an appropriate starting point.

As multi-task recommendation systems grow in complexity — more tasks, more diverse user populations, tighter latency budgets — the field will likely move toward **adaptive, data-driven selection** of balancing strategies: methods that automatically detect the current failure mode and apply the appropriate intervention, without human-specified hyperparameters or architectural changes.

---

## Citation
If you find this post helpful, please consider citing it as:
```bibtex
@article{wang2026mtlgradient,
  author = "Bing Wang",
  title = "Gradient Balancing in Multi-Task Recommendation: A Systematic Guide",
  journal = "bingzw.github.io",
  year = "2026",
  month = "March",
  url = "https://bingzw.github.io/posts/2026-03-08-mtl-gradient-balancing/"
}
```
or
```markdown
Bing Wang. (2026, March). Gradient Balancing in Multi-Task Recommendation: A Systematic Guide.
https://bingzw.github.io/posts/2026-03-08-mtl-gradient-balancing/
```

---

## References

[^1]: [Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML 2018*.](https://arxiv.org/abs/1711.02257)
[^2]: [Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. *NeurIPS 2020*.](https://arxiv.org/abs/2001.06782)
[^3]: [Sener, O., & Koltun, V. (2018). Multi-Task Learning as Multi-Objective Optimization. *NeurIPS 2018*.](https://arxiv.org/abs/1810.04650)
[^4]: [Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-Averse Gradient Descent for Multi-task Learning. *NeurIPS 2021*.](https://arxiv.org/abs/2110.14048)
[^5]: He, P., et al. (2022). MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks. *WWW 2022*.
[^6]: SLGrad (2023). Sample-Level Gradient Reweighting for Multi-Task Recommendation.
[^7]: Tang, J., et al. (2020). Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. *RecSys 2020*.
[^8]: He, F., et al. (2024). STEM: Unleashing the Power of Embeddings for Multi-Task Recommendation. *AAAI 2024*.
[^9]: [MultiBalance (2024). Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System. *Meta AI*.](https://arxiv.org/abs/2411.11871)
[^10]: Shi, X., et al. (2024). GradCraft: Holistic Gradient Balancing for Multi-Task Recommendation. *KDD 2024*.
[^11]: [PUB (2024). A Parameter Update Balancing Algorithm for Multi-task Ranking Models in Recommendation Systems.](https://arxiv.org/abs/2410.05806)
[^12]: DRGrad (2025). Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning Recommendations. *AAAI 2025*.
[^13]: Meta Lattice (2024). Lattice — Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations. *Meta AI*.
