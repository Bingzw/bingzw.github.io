---
title: "A Journey to Reinforcement Learning - Deep RL"
date: 2024-07-15T21:40:55-07:00
draft: False
description: "An summary of deep reinforcement learning algorithms"
tags: ['Machine learning', 'Deep Reinforcement Learning']
---
<p align="center">
<img src="/rf/deeprl.png" width="600" height="400"><br>
<p>
<!--more-->

*Image cited from [^1]*

This blog is a continuation of the last one: **A Journey to Reinforcement Learning - Tabular Methods**. The 
table of content structure and notations follow the same framework. Let's continue the journey from the last value based 
algorithm in model free setting.
## Model Free - when the environment is unknown
### Value Based (Continuing ...)
#### Deep Q Network (DQN)
Like what have been discussed in SARSA and $Q$ learning, they may not be a good fit when state or action space is too large 
or even continuous. Maintaining a $Q$ table is not feasible in such cases. Instead, we can apply function approximation to 
the $Q$ value. So far, the model powerful approach that can approximate almost any shape of function is **Neural Network**. 
Welcome to the realm of **deep reinforcement learning** (short for "deep learning" + "reinforcement learning").

The idea seems to be natural. Let's refine the DQN framework step by step. As an extension from $Q$ learning. Let's frame the 
problem with **continuous states** and **discrete actions** (the continuous actions case will be discussed later). The $Q$ value can 
be approximated by fitting a neural network where the state is fed into the network and get output of $Q$ value at each action.

<p align="center">
<img src="/rf/q_dqn.png" width="600" height="400"><br>
<em>Figure 1: Q learning vs DQN</em>
<p>

*Image cited from [^2]*

Given the network, what's the loss function? From the basic $Q$ learning, the $Q$ value is updated
via $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'}Q(s', a') - Q(s, a)]$. This indicates that we are actually 
"learning" the target $r + \gamma \max_{a'}Q(s', a')$. Therefore, the loss function can be defined as
$$
L(\theta, x) = \frac{1}{N}\sum_{i=1}^N[Q_{\theta}(s_i, a_i) - (r_i + \arg \max_{a'}Q_{\theta}(s', a'))]^2 \tag{3.4}
$$
given a sample $x = (s_i, a_i, r_i, s'_i)$.

When applying the backpropagation on (3.4), the target value is changing since it also depends on the parameter $\theta$ 
to be optimized. This differs from the typical deep learning where the label is given and fixed. Learning from a changing label 
problem may be very unstable and loss may fluctuate. To tackle this, the target $Q$ network is introduced.

The idea is to maintain two $Q$ neural nets: $Q_\theta$ and $Q_{\theta ^ *}$. The target net $Q_{\theta ^ *}$ is originally cloned from 
the net $Q_\theta$ and the (3.4) is reformatted as

$$
L(\theta, x) = \frac{1}{N}\sum_{i=1}^N[Q_{\theta}(s_i, a_i) - (r_i + \arg \max_{a'}Q_{\theta ^ *}(s', a'))]^2 \tag{3.5}
$$

The parameter $\theta ^ *$ is fixed in (3.5) and only copied from $Q_\theta$ periodically after a few gradient descent runs. In 
this way, the loss function is firstly optimized towards a fixed label (only $\theta$ is updated), then labels are updated (copy 
$\theta$ to $\theta ^ *$) and optimization continues.

Gathering the training samples purely from interacting with the system results in dependent samples due to the Markov property. In DQN, 
An **experience replay** (ER) component introduce a *replay buffer* that is sampling independent samples $x = (s_i, a_i, r_i, s'_i)$, 
thus also improving the training efficiency.

A general structure of DQN is
<p align="center">
<img src="/rf/dqn_flow.png" width="600" height="400"><br>
<em>Figure 2: DQN flow</em>
<p>

*Image cited from [^3]*


- Initialize the $Q_{\theta}$
- Initialize the target network $Q_{\theta ^ *}$ by cloning $\theta ^ *$ from $\theta$
- Initialize the replay buffer
- for episode $e$ from 1 to $K$, do:
  - Initialize the state $s$
  - for $t$ from 1 to $T$, do:
    - apply the $\epsilon$-greedy strategy to $Q_{\theta}$ to choose the action $a$ based on $s$
    - take the action and interact with the environment to get reward $r$ and $s'$
    - save the sample (s, a, r, s') to the replay buffer
    - if there are enough samples in the replay buffer, sample N data points $\\{(s_i, s_i, r_i, s_{i+1})\\}_{i=1,\dots,N}$
    - for each sample, calculate the target $y_i = r_i + \gamma \arg\max_{a} Q_{\theta ^ *}(s_{i+1}, a)$
    - minimize the loss function $L(\theta, x) = \frac{1}{N}\sum_{i=1}^N[y_i - Q_{\theta}(s_i, a_i)]^2$, update $Q_\theta$
    - Update $Q_{\theta ^ *}$ every $M$ steps
  - end for
- end for
- return $Q_{\theta}$ and derive the optimal policy

#### Advanced DQN
Though DQN extends the problem solving capability from finite states to infinite state space, original DQN still suffers 
from issues like overestimation. Below lists the efforts to further refine the DQN.
##### Double DQN
*Problem Addressed*: DQN suffers from overestimation bias when updating Q-values because it uses the same network to 
select and evaluate actions.

*Approach*: Double DQN separates action selection and action evaluation by using the training network to select actions and 
the target network to evaluate them. This reduces overestimation bias and leads to more accurate Q-value predictions.
##### Duel DQN
*Problem Addressed*: Traditional DQNs struggle to efficiently learn the value of states without requiring specific 
action-value pairs.

*Approach*: Introduces a dueling network architecture that separates the estimation of state values and advantage functions. 
This allows the model to learn which states are valuable without needing to learn the effect of each action in those states, 
leading to improved learning efficiency.
##### Prioritized Experience Replay
*Problem Addressed*: In standard experience replay, all transitions are sampled with equal probability, which can be 
inefficient as some experiences are more valuable for learning.

*Approach*: Prioritizes sampling of experiences that have higher expected learning progress, meaning transitions with 
a higher TD error are sampled more frequently. This focuses learning on more informative samples, speeding up 
the learning process.
##### Noisy Nets
*Problem Addressed*: Standard exploration methods (like epsilon-greedy) can be suboptimal or inefficient.

*Approach*: Replaces deterministic parameters in the neural network with parameterized noise, allowing the network to explore more 
effectively by adding noise directly to the weights. This leads to more effective exploration strategies that adapt over time.
##### Distributional DQN
*Problem Addressed*: Traditional DQNs focus on learning the expected return, which can overlook valuable distributional 
information about the returns.

*Approach*: Instead of estimating the expected value, estimate the distribution of the $Q$ function. This is to capture 
the uncertainty of the $Q$ function. The $Q$ function is then calculated as the expected value of the distribution.
##### Rainbow DQN [^4]
*Problem Addressed*: Individual improvements to DQN (like those above) each address specific limitations, but combining 
them could lead to even more robust performance.

*Approach*: Integrates several enhancements to DQN into a single framework: Double DQN, Duel DQN, Prioritized Experience Replay, 
Noisy Nets, Distributional Q-learning, and a few others. By combining these techniques, Rainbow DQN leverages the strengths of 
each approach to achieve superior performance in a unified model.

<p align="center">
<img src="/rf/rainbow.png" width="500" height="100"><br>
<em>Figure 3: Rainbow DQN</em>
<p>

*Image cited from [^4]*

### Policy Based
Value-based methods (like Q-learning and DQN) focus on estimating the value of state-action pairs to derive a policy. 
However, these methods can struggle with high-dimensional or continuous **action spaces**. Policy-based methods address 
these limitations by directly learning the policy itself. In policy based approaches, policies are often represented 
using parameterized functions, such as neural networks. These functions map states directly to probabilities over actions, 
allowing the agent to choose actions based on the learned distribution.
#### Policy Gradient
Suppose $\pi_{\theta}$ is the parameterized policy that is to be optimized, our objective is to maximize the expected 
cumulative reward from any starting state

<p align="center">
$$
J(\theta) = \mathbb{E}_{s_0}[V_{\pi_{\theta}}(s_0)] \tag{4.1}
$$
</p>

where the expectation is taken over state. Let's derive the derivative of (4.1).

<p align="center">
\begin{aligned}
\nabla_{\theta} V_{\pi_{\theta}}(s) &= \nabla_{\theta}  \sum_{a \in A} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a)\\
&= \sum_{a \in A} (\nabla_{\theta} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a) + \pi_{\theta}(a \mid s) \nabla_{\theta} Q_{\pi_{\theta}}(s, a)) \\
&= \sum_{a \in A} (\nabla_{\theta} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a) + \pi_{\theta}(a \mid s) \nabla_{\theta} \sum_{s', r} p(s', r \mid s, a)(r + \gamma V_{\pi_{\theta}}(s')) \\
&= \sum_{a \in A} (\nabla_{\theta} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a) + \gamma\pi_{\theta}(a \mid s) \sum_{s', r} p(s', r \mid s, a)\nabla_{\theta} V_{\pi_{\theta}}(s') \\
&= \sum_{a \in A} (\nabla_{\theta} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a) + \gamma\pi_{\theta}(a \mid s) \sum_{s'} p(s' \mid s, a)\nabla_{\theta} V_{\pi_{\theta}}(s') \hspace{2.4em} \text{(4.2)}
\end{aligned}
</p>

Let's define $\phi(s) = \sum_{a \in A} \nabla_{\theta} \pi_{\theta}(a \mid s)Q_{\pi_{\theta}}(s, a)$ and denote the probability 
of getting state $s'$ after $k$ steps from state $s$ under the policy $\pi_{\theta}$ as $d_{\pi_{\theta}}(s \rightarrow s', k)$, then continue
with equation (4.2) as

<p align="center">
\begin{aligned}
\nabla_{\theta} V_{\pi_{\theta}}(s) &= \phi(s) + \gamma\sum_{a}\pi_{\theta}(a \mid s) \sum_{s'} p(s' \mid s, a)\nabla_{\theta} V_{\pi_{\theta}}(s') \\
&= \phi(s) + \gamma\sum_{a}\sum_{s'}\pi_{\theta}(a \mid s)  p(s' \mid s, a)\nabla_{\theta} V_{\pi_{\theta}}(s') \\
&= \phi(s) + \gamma\sum_{s'}d_{\pi_{\theta}}(s \rightarrow s', 1)\nabla_{\theta} V_{\pi_{\theta}}(s') \\
&= \phi(s) + \gamma\sum_{s'}d_{\pi_{\theta}}(s \rightarrow s', 1) [\phi(s') + \gamma\sum_{s''}d_{\pi_{\theta}}(s' \rightarrow s'', 1)\nabla_{\theta} V_{\pi_{\theta}}(s'')] \\
&= \phi(s) + \gamma\sum_{s'}d_{\pi_{\theta}}(s \rightarrow s', 1) \phi(s') + \gamma^2\sum_{s''}d_{\pi_{\theta}}(s \rightarrow s'', 2)\nabla_{\theta} V_{\pi_{\theta}}(s'') \\
&= \phi(s) + \gamma\sum_{s'}d_{\pi_{\theta}}(s \rightarrow s', 1) \phi(s') + \gamma^2\sum_{s''}d_{\pi_{\theta}}(s \rightarrow s'', 2)\nabla_{\theta} V_{\pi_{\theta}}(s'') + \gamma^3\sum_{s'''}d_{\pi_{\theta}}(s \rightarrow s''', 3)\nabla_{\theta} V_{\pi_{\theta}}(s''')] \\
&= \dots \\
&= \sum_{x \in S} \sum_{k=0}^{\infty} \gamma^k d_{\pi_{\theta}}(s \rightarrow x, k)\phi(x) \hspace{16 em} \text{(4.3)}
\end{aligned}
</p>

Now the gradient of the expected cumulative reward function is

<p align="center">
\begin{aligned}
\nabla_{\theta} J_{\pi_{\theta}}(s) &= \nabla_{\theta} \mathbb{E}_{s_0} [V_{\pi_{\theta}}(s_0)] \\
&= \sum_{s \in S} \mathbb{E}_{s_0} [\sum_{k=0}^{\infty} \gamma^k d_{\pi_{\theta}}(s_0 \rightarrow s, k)\phi(s)] \\
&= \sum_{s \in S} \eta(s)\phi(s), \hspace{1 em} (\eta(s) = \mathbb{E}_{s_0} [\sum_{k=0}^{\infty} \gamma^k d_{\pi_{\theta}}(s_0 \rightarrow s, k)]) \\
&= (\sum_{s \in S} \eta(s)) \sum_{s \in S} \frac{\eta(s)}{\sum_{s \in S} \eta(s)} \phi(s) \\
& \propto \sum_{s \in S} \frac{\eta(s)}{\sum_{s \in S} \eta(s)} \phi(s) \\
&= \sum_{s \in S} \nu(s) \phi(s), \hspace{1 em} (\nu(s) = \frac{\eta(s)}{\sum_{s \in S} \eta(s)}) \\
&= \sum_{s \in S} \nu(s) \sum_{a \in A} Q_{\pi_{\theta}}(s, a) \nabla_{\theta} \pi_{\theta}(a \mid s) \\
&= \sum_{s \in S} \nu(s) \sum_{a \in A} \pi_{\theta}(a \mid s) Q_{\pi_{\theta}}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)} \\
&= \mathbb{E}_{\pi_{\theta}} [Q_{\pi_{\theta}}(s, a) \nabla_{\theta} \log(\pi_{\theta}(a \mid s))] \hspace{15 em} \text{(4.4)}
\end{aligned}
</p>

Thus, the expected cumulative reward function can be optimized by taking the gradient ascent from (4.4). Note that we may need to estimate 
the $Q_{\pi_{\theta}}(s, a)$ when calculating the gradient, a simple way is to apply Monte Carlo here and thus results in "REINFORCE" algorithm.

#### REINFORCE
- Initialize the policy parameter $\theta$
- for episode $e$ from 1 to $K$, do:
  - Initialize the state $s$
  - for $t$ from 1 to $T$, do:
    - sample the trajectories $\\{s_1, a_1, r_1, \dots, s_T, a_T, r_T\\}$ under the policy $\pi_{\theta}$
    - For any $t \in [1, T]$, calculate the total reward since time $t$, $\psi_t = \sum_{t'=t}^T \gamma^{t'-t}r_{t'}$
    - apply gradient ascent to update $\theta$, i.e. $\theta = \theta + \alpha \sum_t^T \psi_t \nabla_{\theta} \pi_{\theta}(a_t \mid s_t)$
  - end for
- end for
- return the policy $\pi_\theta$

This is a "policy version" monte carlo approach. Like the Monte Carlo algorithm, it's a simple online algorithm but may suffer from issues like
high variance, delayed rewards, inefficient sampling, noisy updates and non-stationary returns. To tackle these issues, more dedicated tools 
are needed.

#### Trust Region Policy Optimization (TRPO)
When using a deep neural network to fit the policy network, the policy gradient updates can be noisy with high variance. Thus resulting in
worse policy updates due to the fluctuation parameters. Is there any way to guarantee the monotonicity of the parameter updates? 

Let's rephrase the problem as something like this: Suppose the current policy is $\pi_{\theta}$ with parameter $\theta$, we would like 
to search for a new parameter $\theta'$ such that $J(\theta') \geq J(\theta)$.

<p align="center">
\begin{aligned}
J(\theta') - J(\theta) &= \mathbb{E}_{s_0}[V_{\pi_{\theta'}}(s_0)] - \mathbb{E}_{s_0}[V_{\pi_{\theta}}(s_0)] \\
&= \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} r(s_{t}, a_{t})] - \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} V_{\pi_{\theta}}(s_t) - \sum_{t=1}^{\infty} \gamma^{t} V_{\pi_{\theta}}(s_t)] \hspace{3 em} \text{(4.5)} \\
&= \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} r(s_{t}, a_{t})] + \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} (\gamma V_{\pi_{\theta}}(s_{t+1}) - V_{\pi_{\theta}}(s_t))] \\
&= \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} [r(s_{t}, a_{t}) + \gamma V_{\pi_{\theta}}(s_{t+1}) - V_{\pi_{\theta}}(s_t)]] \\
&= \mathbb{E}_{\pi_{\theta'}}[\sum_{t=0}^{\infty} \gamma^{t} A_{\pi_{\theta}}(s_{t}, a_{t})], \hspace{1 em} (A_{\pi_{\theta}}(s_{t}, a_{t}) = Q_{\pi_{\theta}}(s_{t}, a_{t}) - V_{\pi_{\theta}}(s_{t}))\\ 
&= \sum_{t=0}^{\infty} \gamma^{t} \mathbb{E}_{s_{t} \sim P_{t}^{\pi_{\theta'}}} \mathbb{E}_{a_{t} \sim \pi_{\theta'(\cdot \mid s_{t})}} [A_{\pi_{\theta}}(s_{t}, a_{t})] \\
&= \frac{1}{1 - \gamma} \mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta'}}} \mathbb{E}_{a_{t} \sim \pi_{\theta'(\cdot \mid s_{t})}} [A_{\pi_{\theta}}(s_{t}, a_{t})] \hspace{12 em} \text{(4.6)}
\end{aligned}
</p>
where the equation (4.5) holds because the start state $s_0$ does not depend on policy $\pi_{\theta'}$, thus the expectation can be 
rewritten under the policy $\pi_{\theta'}$. In (4.6), we applied that $\nu_{t}^{\pi_{\theta}} = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^{t} P_{t}^{\pi_{\theta}}$.

Therefore, the goal is to find a new policy such that equation (4.6) is non-negative to guarantee the monotonicity property.

However, it's challenging to solve (4.6) directly as $\pi_{\theta'}$ are being used to update policy strategy and sampling states simultaneously. A simple trick is to apply importance sampling using 
the old policy assuming new policy is kind of "similar" to the old one. Thus, the objective function in TPRO is

<p align="center">
\begin{aligned}
L_{TPRO}(\theta') &= \frac{1}{1 - \gamma} \mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta}}} \mathbb{E}_{a_{t} \sim \pi_{\theta'(\cdot \mid s_{t})}} [A_{\pi_{\theta}}(s_{t}, a_{t})] \\
&= \frac{1}{1 - \gamma} \mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta}}} \mathbb{E}_{a_{t} \sim \pi_{\theta(\cdot \mid s_{t})}} [\frac{\pi_{\theta'(a \mid s_{t})}}{\pi_{\theta(a \mid s_{t})}} A_{\pi_{\theta}}(s_{t}, a_{t})]
\end{aligned}
</p>

TPRO is actually trying to solve the following optimization problem.

<p align="center">
$$
\max_{\theta'} \mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta}}} \mathbb{E}_{a_{t} \sim \pi_{\theta(\cdot \mid s_{t})}} [\frac{\pi_{\theta'(a \mid s_{t})}}{\pi_{\theta(a \mid s_{t})}} A_{\pi_{\theta}}(s_{t}, a_{t})] \tag{4.7}
$$
</p>

<p align="center">
$$
s.t. \mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta}}} [D_{KL}( \pi_{\theta(\cdot \mid s_{t})}, \pi_{\theta'(\cdot \mid s_{t})})] \leq \delta \tag{4.8}
$$
</p>

To approximately solve the optimization problem, we can apply the Taylor approximation to the objective and its constraint like this.

<p align="center">
$$
\mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta}}} \mathbb{E}_{a_{t} \sim \pi_{\theta(\cdot \mid s_{t})}} [\frac{\pi_{\theta'(a \mid s_{t})}}{\pi_{\theta_k(a \mid s_{t})}} A_{\pi_{\theta_k}}(s_{t}, a_{t})] \approx g^T(\theta' - \theta_k) \tag{4.9}
$$
</p>

<p align="center">
$$
\mathbb{E}_{s_{t} \sim \nu_{t}^{\pi_{\theta_k}}} [D_{KL}( \pi_{\theta_k(\cdot \mid s_{t})}, \pi_{\theta'(\cdot \mid s_{t})})] \approx \frac{1}{2} (\theta' - \theta_k)^T H(\theta' - \theta_k) \tag{4.10}
$$
</p>
where $g$ denotes the gradient of the left hand side of equation (4.9) and $H$ represents the Hessian matrix of the left hand side of equation (4.10). The optimization problem can be solved by the conjugate gradient method with
the following formula $\theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{x^T Hx}}x$.

Therefore, the general TRPO algorithm is

- initialize the policy network parameters $\theta$ and value network parameters $\omega$
- for episode $e$ from 1 to $E$, do:
  - sample the trajectories $\\{s_1, a_1, r_1, \dots, \\}$ under the policy $\pi_{\theta}$
  - calculate the advantage $A(s_t, a_t)$ for each state, action pair. $A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l (r_{t+1} + \gamma V_{\omega}(s_{t+2}) - V_{\omega}(s_{t+1})), \lambda \in [0, 1]$
  - calculate the gradient $g$ of the objective function 
  - calculate the $x = H^{-1}g$
  - Find $i \in \\{1,2,\dots,K\\}$ and update the policy network parameter $\theta_{k+1} = \theta_k + \alpha^{i}\sqrt{\frac{2\delta}{x^T Hx}}x, \alpha \in (0, 1)$
  - Update the value network parameters by minimizing the square error: 
  <p align="center">
  $$ L(\omega) = \frac{1}{2} \mathbb{E}_t [G_t - V_{\omega} (s_t)]^2 $$
  </p>
- end for
- return the policy $\pi_\theta$

#### Proximal Policy Optimization (PPO)
Though TRPO is successful in many cases, the computation can be time consuming due to its complexity. To simplify the optimization process, PPO is taking the following two approaches for the 
objective (4.7) with the constraint (4.8).

##### PPO-penalty
Instead of optimizing a constraint objective, we can transform the objective (4.7) into an un-constraint optimization problem by using Lagrange multipliers.

<p align="center">
$$
\max_{\theta} \mathbb{E}_{s \sim \nu_{t}^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k(\cdot \mid s)}} [\frac{\pi_{\theta(a \mid s)}}
{\pi_{\theta_k(a \mid s)}} A_{\pi_{\theta_k}}(s, a) - \beta D_{KL}( \pi_{\theta_k(a \mid s)}, \pi_{\theta(a \mid s)})] \tag{4.11}
$$
</p>
where $d_k = D_{KL}^{\nu_{t}^{\pi_{\theta_k}}}(\pi_{\theta_k}, \pi_\theta)$ denotes the DL divergence between policies in two consecutive iterations. $\beta$ can be updated according to

    if d_k < delta / 1.5:
      beta_{k+1} = beta_k /2
    elif d_k > delta * 1.5:
      beta_{k+1} = beta_k * 2
    else:
      beta_{k+1} = beta_{k}

where $\delta$ is a hyper-parameter which is set in the beginning of learning.

##### PPO-Clip
The other way of joining the constraint into the objective is using clips, that is, to set boundaries for the objective function.

<p align="center">
$$
\max_{\theta} \mathbb{E}_{s \sim \nu_{t}^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k(\cdot \mid s)}} \left[min \\
\left(\frac{\pi_{\theta(a \mid s)}}{\pi_{\theta_k(a \mid s)}} A_{\pi_{\theta_k}}(s, a), clip \left(\frac{\pi_{\theta(a \mid s)}}
{\pi_{\theta_k(a \mid s)}}, 1-\epsilon, 1+\epsilon \right)A_{\pi_{\theta_k}}(s, a)\right) \right] \tag{4.12}
$$
</p>
where $clip(x, a, b) = max(min(x, b), a)$ and $\epsilon$ is a hyper-parameter. This makes the policy updates to be within the $[1-\epsilon, 1+\epsilon]$.

Therefore the sudo code for PPO is:

- initialize the policy network parameters $\theta$ and value network parameters $\omega$
- for episode $e$ from 1 to $E$, do:
  - sample the trajectories $\\{s_1, a_1, r_1, \dots, \\}$ under the policy $\pi_{\theta}$
  - calculate the advantage $A(s_t, a_t)$ for each state, action pair. $A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l (r_{t+1} + \gamma V_{\omega}(s_{t+2}) - V_{\omega}(s_{t+1})), \lambda \in [0, 1]$
  - Compute discounted cumulative rewards:$ G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$
  - calculate the gradient $g$ of the objective function 
  - update the policy network using stochastic gradient ascent
  - Update the value network parameters by minimizing the square error:
<p align="center">
$$ L(\omega) = \frac{1}{2} \mathbb{E}_t (G_t - V_{\omega} (s_t))^2 $$
</p>

- end for
- return the policy $\pi_\theta$

#### Cross-Entropy Method [Gradient Free]
#### Evolution Strategy [Gradient Free]
### Hybrid
A hybrid approach in reinforcement learning combines elements from both value-based and policy-based methods to leverage 
their respective strengths while mitigating their weaknesses.

- Value-based methods (e.g., Q-Learning, DQN) focus on learning a value function (like $Q(s, a)$) to guide decision-making 
but struggle in high-dimensional or continuous action spaces.
- Policy-based methods (e.g., REINFORCE) directly learn a policy $\pi(a|s)$, which works well for continuous actions but 
suffers from high variance in gradient estimation.

By integrating these approaches, a hybrid method learns both the value function (to stabilize learning and reduce variance) 
and the policy (to directly optimize actions). Actor-Critic algorithms, like A2C, PPO, and SAC, are popular examples of 
hybrid methods, combining a critic (value function) to evaluate actions and an actor (policy) to select actions. This synergy 
improves learning efficiency, stability, and scalability to complex environments.
#### Actor Critic (AC)
Actor-Critic is a class of reinforcement learning (RL) algorithms that combines the benefits of policy-based methods 
(like REINFORCE) and value-based methods (like Q-learning). It is a hybrid approach where two components — an actor and 
a critic — work together to optimize the policy. There are two components: actor and critic.

The actor is responsible for learning and outputting the policy, $\pi_\theta(a|s)$, which is a mapping from states to actions.
It is a parameterized function (e.g., a neural network) with parameters $\theta.$
Its goal is to directly improve the policy by maximizing the expected reward. The policy gradient (4.4) can be extended to

<p align="center">
$$g = \mathbb{E}_{\pi_{\theta}} [A_{\pi_{\theta}}(s, a) \nabla_{\theta} \log(\pi_{\theta}(a \mid s))] \tag{4.13}$$
</p>

when introducing the value function $V_\pi(s)$ as baseline, and the advantage function $A_{\pi_{\theta}}(s, a)$ is usually 
approximated by the temporal difference $\delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)$.

The critic evaluates how good the actions taken by the actor are, using a value function.
Common choices for the value function:
- State Value Function: $V_\pi(s) = \mathbb{E}_\pi [G_t | s_t = s]$
- Action-Value Function: $Q_\pi(s, a) = \mathbb{E}_\pi [G_t | s_t = s, a_t = a]$
- Advantage Function: $A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$

The critic guides the actor by providing feedback on its actions. When updating the critic net, the loss can be defined by temporal difference, i.e. 
$L_\omega = \frac{1}{2}(r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t))^2$. Therefore the gradient of critic loss 
is $$\nabla_\omega L_\omega = -(r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t))\nabla_\omega V_\omega(s_t) \tag{4.14}$$ Both the actor and critic loss 
are thus optimized by gradient descent.

<p align="center">
<img src="/rf/ac.png" width="300" height="150"><br>
<em>Figure 4: Actor Critic</em>
<p>

*Image cited from [^5]*

In all, the sudo code for actor-critic is:
- initialize the policy network parameters $\theta$ and value network parameters $\omega$
- for episode $e$ from 1 to $E$, do:
  - sample the trajectories $\\{s_1, a_1, r_1, \dots, \\}$ under the policy $\pi_{\theta}$
  - calculate the temporal difference by $\delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)$
  - calculate the gradient (4.13) and update the policy net parameters by $\theta$ 
  - calculate the gradient (4.14) and update the value net parameters by $\omega$
- end for
- return the policy $\pi_\theta$ and value function $V_\omega$

Actor-Critic serves as the foundation for many advanced RL algorithms and is widely used in solving complex decision-making problems.
For example, PPO (Proximal Policy Optimization) is an variant that improves stability by constraining the policy update step.
SAC (Soft Actor-Critic) extends actor-critic by incorporating entropy regularization to encourage exploration.

#### DDPG
Reinforce, actor-critic, TRPO and PPO are all about on-policy learning algorithms, while DQN is off-policy algorithm, it suffered from the discrete action space.
Deep Deterministic Policy Gradient (DDPG) is a model-free, off-policy reinforcement learning algorithm designed for continuous action spaces. 
It combines ideas from Deterministic Policy Gradient (DPG) and Deep Q-Learning (DQN), leveraging neural networks to approximate policies and value functions.

DDPG leverages the Actor-Critic Architecture. However, some unique key features in DDPG include:  

- Actor: Learns a deterministic policy $\mu_\theta(s)$ that maps states directly to actions.
- Critic: Estimates the action-value function $Q(s, a)$, which evaluates the quality of actions.

- Off-Policy Learning: Uses a replay buffer to store past experiences $(s, a, r, s')$, enabling sample efficiency and breaking correlations between samples.
- Target Networks: Employs target networks for both the actor and critic to stabilize learning by slowly updating their parameters towards the current networks.
- Continuous Action Space:Unlike DQN, which handles discrete actions, DDPG can handle high-dimensional continuous actions, making it suitable for tasks like robotics control.

DDPG sudo Algorithm:
- Initialize the actor network $\mu_\theta(s)$ and the critic network $Q_\phi(s, a)$ with random weights.
- Create target networks $\mu_{\theta'}(s)$ and $Q_{\phi'}(s, a)$ by copying the weights of the actor and critic networks.
- Initialize a replay buffer to store transitions $(s, a, r, s')$.
- For each episode:
  - At each step:
    - Select an action using the actor network with added noise for exploration: $a_t = \mu_\theta(s_t) + \mathcal{N}_t$ (where $\mathcal{N}_t$ is a noise process like Ornstein-Uhlenbeck).
    - Execute the action, observe reward $r_t$ and the next state $s_{t+1}$.
    - Store the transition $(s_t, a_t, r_t, s_{t+1})$ in the replay buffer.
    - Sample a minibatch of transitions from the replay buffer.
    - Update the critic by minimizing the Bellman error: $\mathcal{L}(\phi) = \mathbb{E}\left[\left(Q_\phi(s, a) - \left(r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))\right)\right)^2\right]$
    - Update the actor using the deterministic policy gradient: $\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q_\phi(s, a) \big|{a=\mu_\theta(s)} \right]$
    - Update the target networks:
      - $\phi' \leftarrow \tau \phi + (1 - \tau) \phi'$
      - $\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$
- Repeat until convergence.

#### SAC


## Reference
[^1]: Mao, Hongzi, et al. "Resource management with deep reinforcement learning." Proceedings of the 15th ACM workshop on hot topics in networks. 2016
[^2]: Sebastianelli, Alessandro, et al. "A Deep Q-Learning based approach applied to the Snake game." 2021 29th Mediterranean Conference on Control and Automation (MED). IEEE, 2021
[^3]: Muteba, K. F., Karim Djouani, and Thomas O. Olwal. "Deep reinforcement learning based resource allocation for narrowband cognitive radio-IoT systems." Procedia Computer Science 175 (2020): 315-324
[^4]: Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.
[^5]: [The Actor-Critic Reinforcement Learning algorithm](https://medium.com/intro-to-artificial-intelligence/a-link-between-cross-entropy-and-policy-gradient-expression-b2b308511867)