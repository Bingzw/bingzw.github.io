---
title: "Recommendation Models"
date: 2024-06-06T22:04:53-07:00
draft: false
description: "An introduction to recommendation models"
tags: ['Machine learning', 'Deep Learning', 'Recommendation']
---
<p align="center">
<img src="/recommendation_model/reco.jpeg" width="600" height="400"><br>
<p>
<!--more-->
*Image cited from [vecteezy](https://www.vecteezy.com/)*

## Introduction to Recommendation Models

In the digital age, the sheer volume of available content and products can be overwhelming for users. Recommendation 
systems play a crucial role in helping users discover relevant items by filtering through vast amounts of data. These 
systems are integral to various industries, including e-commerce, streaming services, social media, and online 
advertising.

Recommendation models are algorithms designed to suggest items to users based on various factors such as past behavior, 
preferences, and item characteristics. The goal is to enhance user experience by providing personalized recommendations, 
thereby increasing engagement and satisfaction.

In most cases, the goal is trying to find the most appealing item for the user given all user demographic, behavior and 
interest information. A well-designed recommendation system usually results in better personalization, user engagement, 
and revenue generation. Let's take a look at the most popular modeling approaches.

## Collaborative Filtering (CF)

Collaborative Filtering is one of the most famous recommendation models so far. It's about to guess the user's interest 
based on the behaviors and preferences of other users with similar tastes/characteristics. This basic approach is 
straightforward and only requires user-item interaction history.

### Memory Based CF
Suppose the recommendation is purely derived from historical records and it does not involve any predictive modelings. 
Let's assume a table of all ratings of users on items. Let $r_{u,i}$ denotes the missing rating of user $u$ on item $i$. 
$S_u$ be a set of users that share similar characteristics with user $u$. $P_j$ denotes a set of items that are close to 
item $j$. The goal is to guess the missing rating $r_{u,i}$. Let's start expressing the "predicted" rating from either user-user 
filtering or item-item filtering.

#### User User Filtering
Assume that users with a similar profile share a similar taste. We are predicting the missing rating as the weighted 
average from ratings of similar users:
<p align="center">
$r_{u,i} = \bar{r}_u + \frac{\sum_{u'\in S_u}sim(u, u') * (r_{u',i} - \bar{r}_{u'})}{\sum_{u'\in S_u}sim(u, u')}$
<p>
where $\bar{r}_u$ is the average rating of user $u$ and $sim(u, u')$ represents the similarity score between user $u$ 
and $u'$. A common choice is to calculate the cosine similarity between two user rating vectors.

#### Item Item Filtering
Assume that the customers will prefer products that share a high similarity with those already well appreciated. The 
missing rating is thus predicted as the weighted average of a set of similar products:
<p align="center">
$r_{u,i} = \frac{\sum_{j \in P_i}sim(j, i) * r_{u,j}}{\sum_{j \in P_i}sim(j, i)}$
<p>

It's straightforward to observe that user-based filtering is to check what other users think of the same product, while 
item-based filtering is aggregating what the user thinks of other items. Essentially, both are weighted linear 
combination of observed ratings. Can we do better than weighted average?

### Model Based CF
Unlike the memory based CF which is trying to fill in the missing cells in the rating matrix, model-based collaborative 
filtering is to predict user preferences for items based on past interactions. It often relies on the concept of 
latent factors. These are hidden features that influence user preferences and item characteristics. By uncovering these 
latent factors, the model can predict the likelihood of a user liking an item. 

Let $R$ denotes the user item interaction rating matrix, $R \in \mathbb{R}^{m*n}$, where $m$ is the dimension of user 
space and $n$ denotes the dimension of item space.

#### Clustering
Given user or item embeddings, a simple approach is to apply the K nearest neighbours to find the K closest users or 
items depending on the similarity metrics used.

#### Matrix Factorization
The basic idea is to decompose the user-item interaction matrix into two lower-dimensional matrices:
- **User Matrix (U)**: Represents users in terms of latent factors, $U \in \mathbb{R}^{m * p}$, where $p$ is the dimension
of the latent space.
- **Item Matrix (V)**: Represents items in terms of latent factors, $V \in \mathbb{R}^{n * p}$

The interaction matrix $R$ is approximated as the product of these two matrices: $R \approx U \cdot V^T$. In practice, we
are usually optimizing for a weighted matrix factorization objective[^1]
<p align="center">
$$L(\theta) = \sum_{(i, j)\in obs} \omega_{i, j} (R_{i, j} - U_i \cdot V_{j}^{T})^2 + \omega_{0} \sum_{(i, j) \notin obs}
(U_i \cdot V_{j}^{T})^2$$
<p>
where $\omega_0$ is a hyper-parameter that weights the two terms so that the objective is not 
dominated by one or the other, and $\omega_{i, j}$ is a function of the frequency of user $i$ and item $j$.

Common optimization algorithms to minimize the above objective function includes **stochastic gradient descent** and 
**weighted alternating least squares (WALS)**.

Recently, a few efforts on deep learning have also been proposed on matrix factorization. For example,
##### Deep Auto-Encoders[^2]
It proposed a model called Collaborative Denoising Auto-Encoder (CDAE) that extends the traditional denoising autoencoder 
architecture to the collaborative filtering domain. CDAE is designed to handle implicit feedback, where user preferences 
are inferred from user behavior rather than explicit ratings. The model incorporates both user-specific and item-specific 
factors, leveraging the rich user interaction data to learn better representations for recommendation tasks.

<p align="center">
    <img src="/recommendation_model/autoencoder_mf.png"><br>
    <em>Figure 1: deep auto-encoder</em>
</p>

##### Neural Collaborative Filtering[^3]
NCF replaces these linear latent factor models with non-linear neural networks, allowing for a more expressive 
representation of user-item interactions. By doing so, NCF can capture more complex patterns in the data that traditional 
methods might miss. The core idea is to use multi-layer perceptrons (MLPs) to model the interaction function between 
users and items, providing a more flexible and powerful framework for learning user preferences.

The general architecture of NCF includes embedding layers for users and items, followed by one or more hidden layers 
that learn the interaction between these embeddings. The final output layer predicts the user's preference for a given 
item. This approach not only improves the accuracy of recommendations but also enables the integration of additional 
features, such as user demographics or item attributes, into the model. Refer to the DLRM or DeepFM for more details.

<p align="center">
    <img src="/recommendation_model/ncf_mf.png"><br>
    <em>Figure 2: neural collaborative filtering</em>
</p>

## Content Based Recommendation

## Ranking As Recommendations

### Wide & Deep a3

### DeepFM

### Deep & Cross Network (DCN)

### Attention Factorization Machine (AFM)

### Deep Interest Network (DIN)

### Deep Learning Recommendation Model (DLRM)

### Entire Space Multi-Task Model (ESMM)

### Multi-gate Mixture of Experts (MMOE)

### Progressive Layered Extraction (PLE)

### Behavior Sequence Transformer (BST)

## Reference
[^1]: [Collaborative Filtering and Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
[^2]: [Kuchaiev, Oleksii, and Boris Ginsburg. "Training deep autoencoders for collaborative filtering." arXiv preprint arXiv:1708.01715 (2017)](https://arxiv.org/pdf/1708.01715)
[^3]: [He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017](https://arxiv.org/pdf/1708.05031)