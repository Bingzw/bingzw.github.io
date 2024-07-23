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
#### Policy Gradient
#### TRPO/PPO
#### Cross-Entropy Method [Gradient Free]
#### Evolution Strategy [Gradient Free]
### Hybrid
#### DDPQ
#### Actor Critic (AC)
#### SAC


## Reference
[^1]: Mao, Hongzi, et al. "Resource management with deep reinforcement learning." Proceedings of the 15th ACM workshop on hot topics in networks. 2016
[^2]: Sebastianelli, Alessandro, et al. "A Deep Q-Learning based approach applied to the Snake game." 2021 29th Mediterranean Conference on Control and Automation (MED). IEEE, 2021
[^3]: Muteba, K. F., Karim Djouani, and Thomas O. Olwal. "Deep reinforcement learning based resource allocation for narrowband cognitive radio-IoT systems." Procedia Computer Science 175 (2020): 315-324
[^4]: Hessel, Matteo, et al. "Rainbow: Combining improvements in deep reinforcement learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.