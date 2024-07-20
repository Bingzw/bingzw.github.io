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

With such a neural network, how to define the loss function? From the basic $Q$ learning, the $Q$ value is updated
via $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'}Q(s', a') - Q(s, a)]$. This indicates that we are actually 
"learning" the target $r + \gamma \max_{a'}Q(s', a')$. Therefore, the loss function can be defined as
$$
L(\theta, x) = \frac{1}{N}\sum_{i=1}^N[Q_{\theta}(s_i, a_i) - (r_i + \arg \max_{a'}Q_{\theta}(s', a'))]^2 \tag{3.4}
$$
given a sample $x = (s_i, a_i, r_i, s'_i)$.

When applying the backpropagation on (3.4), the target value is changing since it also depends on the parameter $\theta$ 
to be optimized. This differs from the typical deep learning where the label is given and fixed. Learning from a changing label 
problem may be very unstable and loss may fluctuate. To tackle this, the target $Q$ network is introduced.

The idea is to maintain two $Q$ neural nets: $Q_\theta$ and $Q_{\theta*}$. The target net $Q_{\theta*}$ is originally cloned from 
the net $Q_\theta$ and the (3.4) is reformatted as

$$
L(\theta, x) = \frac{1}{N}\sum_{i=1}^N[Q_{\theta}(s_i, a_i) - (r_i + \arg \max_{a'}Q_{\theta*}(s', a'))]^2 \tag{3.5}
$$

The parameter $\theta*$ is fixed in (3.5) and only copied from $Q_\theta$ periodically after a few gradient descent runs. In 
this way, the loss function is firstly optimized towards a fixed label (only $\theta$ is updated), then labels are updated (copy 
$\theta$ to $\theta*$) and optimization continues.

Another point worth mentioning is about independent samples. To make this assumption valid, we need to build a replay buffer such 
that the interaction data points $x = (s_i, a_i, r_i, s'_i)$ can be reused, thus also improving the training efficiency.


#### Advanced DQN
### Policy Based
#### Policy Gradient
#### TRPO/PPO
#### Cross-Entropy Method [Gradient Free]
#### Evolution Strategy [Gradient Free]
### Hybrid
#### DDPQ
#### Actor Critic (AC)
#### SAC