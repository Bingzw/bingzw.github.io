---
title: "A Journey to Reinforcement Learning"
date: 2024-07-05T20:59:39-07:00
draft: False
description: "An summary of reinforcement learning algorithms"
tags: ['Machine learning', 'Deep Reinforcement Learning', 'Reinforcement Learning']
---
<p align="center">
<img src="/rf/rf.png" width="600" height="400"><br>
<em>Figure 1: basic RL model</em>
<p>
<!--more-->
*Image cited from [^1]*

## Basic Problem Statement of Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in 
an environment to maximize cumulative reward. Unlike supervised learning, where the model is trained on a fixed dataset, 
RL involves learning through interaction with the environment. Let's define some basic elements in RL domain.

- **Agent**: The learner or decision maker.
- **Environment**: The external system the agent interacts with.
- **State** $S$: A representation of the current situation of the agent.
- **Action** $A$: The set of all possible moves the agent can make.
- **Reward** $R$: A feedback signal from the environment to evaluate the agent's action.
- **Policy $\pi$**: A strategy used by the agent to determine the next action based on the current state. It's usually 
a probability function $\pi: S\times A$ -> $[0, 1]$
- **Value Function $V(s)$**: A function that estimates the expected cumulative reward from a given state.
- **Q-Function $Q(s, a)$**: A function that estimates the expected cumulative reward from a given state-action pair.

As it can be seen in *Figure 1*, the general workflow of RL involves 
1. Observe the state $s_t$ (and reward $r_t$), the state can be any representation of the current situation or context in which the agent operates.
Note that the state space $S$ can be either **finite** or **infinite**.
2. Based on the current policy $\pi$, the agent selects an action $a_t \in A$ to perform. The selection can be **deterministic** 
or **stochastic** depending on the policy.
3. The agent performs the selected action $a_t$ in the environment (can also be either **known** or **unknown**). 
4. After performing the action, the agent receives a reward $r_{t+1}$ from the environment and observes the next state $s_{t+1}$.
5. The agent updates its **policy** $\pi(a_t|s_t)$ and **value functions** $V(s_t)$ or $Q(s_t, a_t)$ based on the observed reward $r_t$ 
and next state $s_{t+1}$. The update rule varies depending on the RL algorithm used. (Note that the policy learned in step 5 
can be either the **same (on policy)** or **different (off policy)** with the ones in step 2)
6. The agent repeats again from step 1 and continues this iterative process until the policy converges, meaning it has 
learned an optimal or near-optimal policy that maximizes cumulative rewards over time.

Reinforcement Learning (RL) derives its name from the concept of "reinforcement" in behavioral psychology, where learning 
occurs through rewards and punishments. The agent learns to make decisions by receiving feedback in the form of rewards or 
penalties. Positive outcomes reinforce the actions that led to them, strengthening the behavior. The learning process is kind 
of a process of **"Trial and Error"**, where the agent explores different actions to discover which ones yield the highest rewards.
Long-term beneficial actions are reinforced through repeated positive outcomes.

Now let's start with the most influential and fundamental RL model - Markov Decision Process (MDP). Our fantastic journey begins here.
## Model Based - When the environment is given
### Dynamic Programming
## Model Free - When the environment is unknown
## Value Based
### On-Policy
#### SARSA
### Off-Policy
#### Q-Learning
#### Deep Q Network (DQN)
## Policy Based
### Gradient Based
#### Policy Gradient
#### TRPO/PPO
#### Actor Critic (AC)
### Gradient Free
#### Cross-Entropy Method
#### Evolution Strategy
## Summary
## Citation
## Reference
[^1]: [Souchleris, Konstantinos, George K. Sidiropoulos, and George A. Papakostas. "Reinforcement learning in game industry—review, prospects and challenges." Applied Sciences 13.4 (2023): 2443](https://www.mdpi.com/2076-3417/13/4/2443)





