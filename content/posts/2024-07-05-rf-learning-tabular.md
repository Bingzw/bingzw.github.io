---
title: "A Journey to Reinforcement Learning - Tabular Methods"
date: 2024-07-05T20:59:39-07:00
draft: False
description: "An summary of reinforcement learning algorithms basics, including mdp, dp, td, sarsa, q-learning"
tags: ['Machine learning', 'Reinforcement Learning']
---
<p align="center">
<img src="/rf/rf.png" width="600" height="400"><br>
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
  - A factor $\gamma$ between 0 and 1 that represents the difference in importance between future rewards and immediate rewards is 
  usually used to prioritize short-term rewards over long-term rewards
- **Policy $\pi$**: A strategy used by the agent to determine the next action based on the current state. It's usually 
a probability function $\pi: S\times A$ -> $[0, 1]$
- **Value Function $V_{\pi}(s)$**: A function that estimates the expected cumulative reward from a given state with the policy $\pi$.
- **Q-Function $Q_{\pi}(s, a)$**: A function that estimates the expected cumulative reward from a given state-action pair with the policy $\pi$.

As it can be seen in the above figure, the general workflow of RL involves 
1. Observe the state $s_t$ (and reward $r_t$), the state can be any representation of the current situation or context in which the agent operates.
Note that the state space $S$ can be either **finite** or **infinite**.
2. Based on the current policy $\pi$, the agent selects an action $a_t \in A$ to perform. The selection can be **deterministic** 
or **stochastic** depending on the policy.
3. The agent performs the selected action $a_t$ in the environment (can also be either **known** or **unknown**). 
4. After performing the action, the agent receives a reward $r_{t+1}$ from the environment and observes the next state $s_{t+1}$. 
The mechanism of moving from one state to another given a specific action is modeled by a probability function, denoted as $P(s_{t+1} \mid s_t, a_t)$.
5. The agent updates its **policy** $\pi(a_t \mid s_t)$ and **value functions** $V(s_t)$ or $Q(s_t, a_t)$ based on the observed reward $r_t$ 
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
Note that the below algorithms and equations are cited from [^3], [^4] and [^5].

### Markov Decision Process (MDP)
Let's be more specific about the above settings to realize the MDP. Assume the state space $S$ and action space $A$ are finite, 
the process is equipped with the markov property, that is, $P(s_{t+1} \mid h_t) = P(s_{t+1} \mid s_t, a_t)$, where $h_t = \\{ s_1, a_1, ... , s_t, a_t \\}$
denotes the history of states and actions.

Suppose the agent is interacting with the environment for a total $T$ steps (horizon). Let's define the total reward after time $t$ as
$$
G_t = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{T-t-1} r_T \tag{1.1}
$$
The value function is defined as 
$$
V_{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid s_t=s] \tag{1.2}
$$

The value action function $Q$ is defined as

$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \mid s_t=s, a_t=a] \tag{1.3}
$$

From the above definition, it's easy to derive the connection between $V$ and $Q$. In particular, when marginalizing the action, 
we are able to convert $Q$ function to $V$ value function.

$$
V_{\pi}(s) = \sum_{a \in A} \pi(a \mid s)Q_{\pi}(s, a) \tag{1.4}
$$

when converting $V$ to $Q$, we can see that

<p align="center">
\begin{aligned}
Q_{\pi}(s, a) &= \mathbb{E}_{\pi}[G_t \mid s_t=s, a_t=a] \\
&= \mathbb{E}_{\pi}[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3}+ ... \mid s_t=s, a_t=a] \\
&= \mathbb{E}_{\pi}[r_{t+1} \mid s_t=s, a_t=a] + \gamma \mathbb{E}_{\pi}[r_{t+2} + \gamma r_{t+3} + ... \mid s_t=s, a_t=a] \\
&= R(s, a) + \gamma \mathbb{E}_{\pi}[G_{t+1} \mid s_t=s, a_t=a] \\
&= R(s, a) + \gamma \mathbb{E}_{\pi}[\mathbb{E}_{\pi}[G_{t+1} \mid s_{t+1}] \mid s_t=s, a_t=a] \\
&= R(s, a) + \gamma \mathbb{E}_{\pi}[V_{\pi}(s_{t+1}) \mid s_t=s, a_t=a] \\
&= R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V_{\pi}(s') \hspace{15.5em} \text{(1.5)}
\end{aligned}
</p>

#### Bellman Expectation Equation
From (1.2), we can express the value function in an recursive way.
<p align="center">
\begin{aligned}
V_{\pi}(s) &= \mathbb{E}_{\pi}[G_t \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} \mid s_t=s] + \gamma\mathbb{E}_{\pi}[r_{t+2} + \gamma r_{t+3} + ... \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} \mid s_t=s] + \gamma\mathbb{E}_{\pi}[G_{t+1} \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} \mid s_t=s] + \gamma\mathbb{E}_{\pi}[\mathbb{E}_{\pi}[G_{t+1} \mid s_{t+1}] \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} \mid s_t=s] + \gamma\mathbb{E}_{\pi}[V_{\pi}(s_{t+1}) \mid s_t=s] \\
&= \mathbb{E}_{\pi}[r_{t+1} + \gamma V_{\pi}(s_{t+1}) \mid s_t=s] \hspace{19em} \text{(1.6)}
\end{aligned}
</p>
Similarly, the state action function can also be written recursively
$$
Q_{\pi}(s, a) = \mathbb{E}_{\pi}[r_{t+1} + \gamma Q_{\pi}(s_{t+1}, a_{t+1}) \mid s_t=s, a_t=a] \tag{1.7}
$$
Furthermore, equation (1.5) also expressed the current-future connection between $V$ and $Q$. So if we plug equation 
(1.5) in (1.4), then we would get 
$$
V_{\pi}(s) = \sum_{a \in A} \pi(a \mid s)(R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V_{\pi}(s')) \tag{1.8}
$$
which denotes the connection of value function at current state and future state.
On the other hand, if (1.4) is plugged in (1.5), we can see
$$
Q_{\pi}(s, a) = R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) \sum_{a' \in A} \pi(a' \mid s')Q_{\pi}(s', a') \tag{1.9}
$$
which builds the connection of the action value function between the current and future state action pairs. By comparing
(1.6) and (1.8), (1.7) and (1.9), it's easy to observe that (1.8) and (1.9) actually implements the expectation expression explicitly. 

A visualized interpretation of (1.8) and (1.9) are shown in the following backup diagram. 
<p align="center">
    <img src="/rf/backup.png"><br>
    <em>Figure 1: Backup Diagram</em>
</p>

#### Bellman Optimal Equation
The goal of reinforcement learning is to find the optimal policy $\pi^*$ such that the value function is maximized.

$$
\pi^*(s) = \arg \max_{\pi} V_{\pi}(s)
$$

when this is achieved, the optimal value function is $V^*(s) = max_{\pi}V_{\pi}(s), \forall s \in S$. At this time, 
the optimal value function can also be achieved by selecting the best action under the optimal policy

$$
V^* (s) = \max_{a} Q^* (s, a) \tag{1.10}
$$

where $Q^* (s, a) = \arg \max_{\pi} Q_{\pi}(s, a)$, $\forall s \in S, a \in A$.

Let's apply the optimal policy in (1.5) and we have

$$
Q^* (s, a) = R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V^*(s') \tag{1.11}
$$

When plugging (1.10) in (1.11), we get **Bellman optimal equation**

$$
Q^* (s, a) = R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) \max_{a'} Q^* (s', a') \tag{1.12}
$$

Similarly, plugging (1.11) in (1.10), the Bellman optimal value equation is

$$
V^* (s) = \max_{a} \left( R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V^* (s') \right) \tag{1.13}
$$

So far, we have introduced the main math modeling framework in RL. How shall we fit the real-life cases into this framework 
and get the best policy? What trade-offs have to be made in each scenario? Let's take a deep dive.
## Model Based - when the environment is given
Our journey begins with a typical MDP where the transition mechanism and the reward function are fully known. As seen in 
the above Bellman expectation and optimal equations, the problem can be resolved by tackling a similar subproblem recursively.
This is usually known as dynamic programming.
### Dynamic Programming
#### policy iteration
The key idea is to iteratively process two alternative steps: policy evaluation and policy improvement. Policy evaluation 
computes the value function $V^\pi$ for the current policy $\pi$. While policy improvement updates the policy $\pi$ using 
the greedy approach.

- initialize the policy $\pi(s)$ and value function $V(s)$
- while not stop
  - while $\delta > \theta$, do 
    - $\delta \leftarrow 0$
    - for each $s \in S$
      - $v \leftarrow V(s)$
      - $V(s) \leftarrow \sum_{a \in A} \pi(a \mid s)(R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V_{\pi}(s')) $ **(policy evaluation)**
      - $\delta \leftarrow \max(\delta, |v - V(s)|)$
  - end while
  - $\pi_{old} \leftarrow \pi$
  - for each $s \in S$
    - $\pi(s) \leftarrow \arg \max_{a} R(s, a) + \gamma \sum_{s'} p(s' \mid s, a)V(s')$ **(policy improvement)**
  - if $\pi_{old} = \pi$
    - stop and return $\pi$ and $V$

#### value iteration
It usually takes quite a significant amount of time to run policy evaluation, especially when the state and action space are
large enough. Is there a way to avoid too many policy evaluation process? The answer is value iteration. It's an iterative process
to update the Bellman optimal equation (1.13).

- initialize the value function $V(s)$
- while $\delta > \theta$, do 
  - $\delta \leftarrow 0$
  - for each $s \in S$
    - $v \leftarrow V(s)$
    - $V(s) \leftarrow \max_{a \in A} (R(s, a) + \gamma \sum_{s' \in S} p(s' \mid s, a) V_{\pi}(s')) $
    - $\delta \leftarrow \max(\delta, |v - V(s)|)$
- end while
- $\pi(s) \leftarrow \arg \max_{a} R(s, a) + \gamma \sum_{s'} p(s' \mid s, a)V(s')$
- return $V$ and $\pi$

It can be seen that value iteration doesn't own policy updates, it generates the optimal policy when the value function converges.

## Model Free - when the environment is unknown
In practical, the environment is hardly fully known or it's simply a blackbox most of the time. Thus dynamic programming (policy 
iteration & value iteration) might not helpful. In this section, we will introduce solutions originated from various kinds ideas.
### Value Based
This collection of algorithms aims optimizing the the value function $V$ or $Q$, which can then be used to derive the optimal 
policy. The policy is typically derived indirectly by selecting actions that maximize the estimated value. Value based methods
are usually simple to implement and understand. They are effective in environments with discrete and finite action spaces. 
Deriving optimal policies is clear and straightforward. However, they are usually struggling with high-dimensional or continuous 
action spaces. Trying function approximation (e.g., neural networks) may not work well due to unstable and divergence. Besides,
value based methods usually require extensive exploration to accurately estimate value functions.
#### Model Free Policy Evaluation: Monte Carlo & Temporal Difference (TD)
Value based approaches inherits the idea from dynamic programming. The difference now is that the environment is unknown such that
both policy iteration and value iteration are not feasible (transition probably unavailable now). Let's start with a simple question:
**how can we estimate the value function given a policy when the environment is unknown**. The idea is to **interact with the environment** 
and update the value function/policy based on the returned rewards. Depending on how heavily we rely on interacting with the 
environment, we have the Monte Carlo and Temporal difference method.
##### Monte Carlo
Monte Carlo methods rely on averaging returns of sampled episodes to estimate the expected value of states or state-action pairs. 
Unlike temporal difference (TD) methods, Monte Carlo methods do not bootstrap and instead use complete episodes to update value estimates.

- Initialize $V_{\pi}(s)$ arbitrarily for all states $s \in S$. 
- Initialize the total reward $S(s)$ and total visits $N(s)$
- for episode from 1 to $N$, do:
   - Generate an episode trajectory $(s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)$ following policy $\pi$.
   - For each state $s$ that first appearing in the episode trajectory:
     - Compute the return $G_t$  from state $s$: $G_t = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{T-t-1} r_T$
     - total reward at $s$ is $S(s) \leftarrow S(s) + G_t$
     - total count visiting $s$ is $N(s) \leftarrow N(s) + 1$
     - Update the value estimate $V_{\pi}(s)$ as the average of all observed returns for state $s$: $V_{\pi}(s) \leftarrow \frac{S(s)} {N(s)}$.
   - end for
- end for
- return the value function under policy $\pi$

It's worth noting that the monte carlo update can also be reformated in an incremental way, that is, 
$$
V_{\pi}(s) \leftarrow V_{\pi}(s) + \frac{1}{N(s)} (G_t - V_{\pi}(s)) \tag{3.1}
$$
It can be viewed to update the value function with an error term 
$G_t - V_{\pi}(s)$. As $V_{\pi}(s) \rightarrow G_t$, then the error term does not bring too much value and as a result $V(s)$ converges to
$G_t$. So $G_t$ can somehow be interpreted as the target of the update. In Monte Carlo, the updates does not start until the 
entire episode completes. This may not feasible in some cases where the interactive game never ends or more frequent updates 
are expected. To tackle this, we are happy to introduce temporal difference.
##### Temporal Difference
From (1.2), we can see that the Monte Carlo is approximating the target $G_t$ using (3.1). If we only interact with the environment
one step instead of completing the full episode trajectory. It's equivalent to reformat the (1.2) as (1.6), where the target is 
thus $r_{t} + \gamma V_{\pi}(s_{t+1})$. So the formula (3.1) can be written as
$$
V_{\pi}(s) \leftarrow V_{\pi}(s) + \alpha (r_{t+1} + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s)) \tag{3.2}
$$
where the $r_{t+1} + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s)$ is called temporal difference error. Since only one step reward is retrieved
from the system interaction, it's also noted as 1-step temporal difference, i.e. TD(1). What if we interact a few more steps with the environment?
The target $G_t$ would include more future steps rewards. A general representation of TD(k) is shown below.
<p align="center">
\begin{aligned}
&TD(1) \hspace{1em} \rightarrow \hspace{1em} G^{1}_t = r_{t+1} + \gamma V(s_{t+1}) \\
&TD(2) \hspace{1em} \rightarrow \hspace{1em} G^{2}_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2}) \\
&TD(k) \hspace{1em} \rightarrow \hspace{1em} G^{k}_t = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{k-1} r_{t+k} + \gamma^k V(s_{t+k}) \\
&TD(\infty) / MC \hspace{1em} \rightarrow \hspace{1em} G^{\infty}_t = r_{t+1} + \gamma r_{t+2}+ \dots + \gamma^{T-t-1} r_{T})
\end{aligned}
</p>

Compared with Monte Carlo, it's possible to updating the value function in an online fashion, meaning that update happens after 
every step of interaction. It's more efficient than updating after completing an episode. This also indicates that TD learning can be
applied to any piece of episode, which is more flexible. The estimation variance is lower but bias can be higher due to bootstrapping 
(updates based on estimated value of next state) 

Below compares the computation graph among MC, TD and DP. In MC, only the states on the episode trajectory are updated. On the
other hand, DP computes all the related next states when updating. TD compromise both MC and DP, it sampled only one or finite steps 
for updating.

<p align="center">
<img src="/rf/rf_dp_mc_td.png" width="900" height="600"><br>
<em>Figure 2: Visual Interpretation of DP, TD and MC</em>
<p>

*Image cited from [^2]*

#### SARSA
Now given a policy $\pi$, we have learned TD and MC can help evaluate the value function when the environment is unknown. 
The next natural question is How can we find the optimal policy? We can borrow the policy improvement idea from policy 
iteration in DP. Get the next policy by using the greedy search, i.e. $\pi_{i+1}(s) = \arg \max_{a} Q_{\pi_i}(s, a)$. 
Combining the generalized policy evaluation and the greedy policy improvement would allow us to build the generalized policy 
iteration.

<p align="center">
<img src="/rf/policy_iteration.png" width="250" height="150"><br>
<em>Figure 3: Generalized Policy Iteration</em>
<p>

Another difference from the policy iteration in DP is that we may never update some state-action pairs using the pure greedy 
policy improvement strategy. This is because we are now interacting with the environment instead of updating all the states action
explicitly with known transition probabilities. So we may need to allow some extent of exploration to mitigate such issue. One 
solution is to apply the $\epsilon$-greedy strategy. 

<p align="center">
\begin{aligned}
\pi(a \mid s) &= 
\begin{cases} 
      1 - \epsilon & \text{if } a = \arg \max_a Q(s, a) \\
      \epsilon &  \text{other actions in A}
\end{cases}
\end{aligned}
</p>

Thus the complete algorithm (TD policy evaluation on $Q$ + $\epsilon$-greedy policy improvement) is:

- Initialize the $Q(s, a)$
- for iteration from 1 to $N$, do:
  - get initial state $s$
  - apply the $\epsilon$-greedy strategy to choose the action $a$ based on $s$
  - for $t$ from 1 to $T$, do:
    - interact with the environment and get reward $r$ and $s'$
    - apply the $\epsilon$-greedy strategy to choose the next action $a'$ based on $s'$
    - $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$ **(TD policy evaluation)**
    - $s \leftarrow s', a \leftarrow a'$ **($\epsilon$-greedy policy improvement)**
  - end for
- end for
- return the value action $Q$

It's worth noting that once the state, action, reward, next state, next action $(s, a, r, s', a')$ is generated, the update is 
conducted once and then repeats the iterations. Therefore, is called **SARSA**. In a more general case, if we interact more steps
and update the $Q$ value using $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma r_{t+1} + \dots + \gamma^n Q(s_{t+n}, a_{t+n}) - Q(s_t, a_t)]$,
then it's **$n$ step SARSA**.

#### Q-Learning
If the way of updating the TD policy evaluation in SARSA is changed to 
$$ 
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \tag{3.3}
$$
Then we would get the Q learning algorithm.

- Initialize the $Q(s, a)$
- for iteration from 1 to $N$, do:
  - get initial state $s$
  - for $t$ from 1 to $T$, do:
      - apply the $\epsilon$-greedy strategy to choose the action $a$ based on $s$
      - interact with the environment and get reward $r$ and $s'$
      - $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$ 
      - $s \leftarrow s'$
  - end for
- end for
- return the value action $Q$ and derive the optimal policy

The major difference between Q learning and SARSA is about the way to update the $Q$ value. In (3.3), we are actually 
approximating the target value $r + \gamma \max_{a'} Q(s', a')$. So the goal here is to learn
the optimal policy $\pi(a \mid s) = \max_{a'} Q(s, a')$. Let's call the policy used in (3.3) as the target policy. 
The data used to learn the target policy $(s, a, r, s')$ is however generated from another policy that interacts with the 
environment, called the behavior policy ($\epsilon$-greedy). It's easy to see that the target policy is different from the 
behavior policy. This is also called **off-policy** learning.

On the other hand, the behavior policy fully aligns with the target policy in SARSA, which means $a'$ and $a$ are generated
from the same policy. This kind of learning is called **on-policy** learning.

One interesting common area among DP, SARSA, Q-learning is that they all assume finite or discrete state and action space.
Optimizing such problem can be seen as updating a table with an entry for each possible state or state-action pair. Thus, 
these algorithms are also called **tabular methods**.

**To be continued ...**

## Summary
So far, we have introduced the basic RL setting and basic tabular methods that are suitable to simple cases. However, most
real problems involve large quantity or infinite state or action space. More powerful tools are thus needed. In next blog, 
we will start the journey to deep reinforcement learning, an area that utilize the fancy deep learning techniques to tackle 
more complicated problems.

## Citation
If you find this post helpful, please consider citing it as:
```bibtex
@article{wang2024rflearningtabular,
  author = "Bing Wang",
  title = "A Journey to Reinforcement Learning - Tabular Methods",
  journal = "bingzw.github.io",
  year = "2024",
  month = "July",
  url = "https://bingzw.github.io/posts/2024-07-05-rf-learning-tabular/"
}
```
or 
```markdown
Bing Wang. (2024, June). A Journey to Reinforcement Learning - Tabular Methods. 
https://bingzw.github.io/posts/2024-07-05-rf-learning-tabular/
```

## Reference
[^1]: [Souchleris, Konstantinos, George K. Sidiropoulos, and George A. Papakostas. "Reinforcement learning in game industry—review, prospects and challenges." Applied Sciences 13.4 (2023): 2443](https://www.mdpi.com/2076-3417/13/4/2443)
[^2]: [An intuitive guide to reinforcement learning](https://roboticseabass.com/2020/08/02/an-intuitive-guide-to-reinforcement-learning/)
[^3]: [Easy RL](https://datawhalechina.github.io/easy-rl/)
[^4]: [Hands On RL](https://hrl.boyuai.com/)
[^5]: [Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
