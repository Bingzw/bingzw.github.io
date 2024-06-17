---
title: "Recommendation Models"
date: 2024-06-06T22:04:53-07:00
draft: false
description: "An introduction to recommendation models"
tags: ['Machine learning', 'Deep Learning', 'Recommendation', 'Ranking', 'Personalization']
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

$$r_{u,i} = \frac{\sum_{j \in P_i}sim(j, i) * r_{u,j}}{\sum_{j \in P_i}sim(j, i)}$$

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

$$L(\theta) = \sum_{(i, j)\in obs} \omega_{i, j} (R_{i, j} - U_i \cdot V_{j}^{T})^2 + \omega_{0} \sum_{(i, j) \notin obs}
(U_i \cdot V_{j}^{T})^2$$

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

## Content Based Models (Ranking As Recommendation)
Unlike the collaborative filtering that purely relies on user-item interaction data. Content-based recommendation models 
focus on the attributes of items to suggest similar items to users based on their past interactions. These models analyze 
the content (such as text, keywords, categories, or features) associated with the items and create a profile for each 
user based on the features of the items they have shown interest in. Compared with collaborative filtering, it's easier 
to handle the cold start issue when item features are known. However, its recommendation may strictly adheres to the 
user's profile, potentially limiting diversity. There are quite a few popular model frameworks in this area and we would 
focus on the models based on deep neural architecture.

### Wide & Deep[^4]

Wide & Deep learning combined wide linear models and deep neural networks to achieve both memorization and generalization. 
The model consists of two components:

1. **Wide Component**: This part is a generalized linear model (GLM) that excels at memorization by capturing feature 
interactions using cross-product feature transformations. The wide component is effective at handling sparse features 
and explicitly memorizing frequent co-occurrence patterns. It can be represented as:

    $$y_{\text{wide}} = \mathbf{w}^T \mathbf{x} + b $$

    where $\mathbf{x}$ represents the input features, $\mathbf{w}$ represents the learned weights, and $b$ is 
    the bias term.

2. **Deep Component**: This part is a feed-forward neural network that excels at generalization by capturing high-level 
and non-linear feature interactions. The deep component uses dense embeddings to represent categorical features, and it 
learns implicit interactions through multiple hidden layers. The output of the deep component can be represented as:
   
   $$\mathbf{h}^L = f^L(\mathbf{W}^L \mathbf{h}^{L-1} + \mathbf{b}^L)$$
   
   where $\mathbf{h}^L$ is the activation of the $L$-th layer, $\mathbf{W}^L$ and $\mathbf{b}^L$ are the weights and biases 
   of the $L$-th layer, respectively, and $f^L$ is the activation function.
   The final output of the deep component, $ y_{\text{deep}} $, is the output of the last layer of the deep neural network. It is given by:
   
   $$
   y_{\text{deep}} = \mathbf{W}^O \mathbf{h}^{L} + \mathbf{b}^O
   $$
   
   where $ \mathbf{W}^O $ and $ \mathbf{b}^O $ are the weights and bias of the output layer, and $ \mathbf{h}^L $ is the activation of the last hidden layer.
   
   The final prediction is a weighted sum of the outputs from the wide and deep components:
   
   $$\hat{y} = \sigma(y_{\text{wide}} + y_{\text{deep}})$$
   
   where $\sigma$ is the sigmoid function used to squash the output to a probability score.

<p align="center">
   <img src="/recommendation_model/wide_deep.png" width="800" height="600"><br>
   <em>Figure 3: wide & deep model</em>
</p>

### DeepFM[^5]

DeepFM addresses the challenge of capturing both low-order and high-order feature interactions in recommendation systems. 
The model consists of two interconnected components:

1. **FM Component**: This part captures low-order feature interactions using Factorization Machines, which are effective 
for handling sparse data and modeling pairwise feature interactions without manual feature engineering. The FM component 
can be represented as:

   $$
   y_{\text{FM}} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
   $$

   where $w_0$ is the global bias, $w_i$ is the weight of the $i$-th feature, $\mathbf{v}_i$ and $\mathbf{v}_j$ are 
   latent vectors for the $i$-th and $j$-th features, respectively, and $x_i$ and $x_j$ are input feature values.

2. **Deep Component**: This part captures high-order feature interactions through a deep neural network. The deep 
component uses embeddings to represent input features and learns complex interactions through multiple hidden layers. 
The output of the deep component can be represented as:

   $$\mathbf{h}^L = f^L(\mathbf{W}^L \mathbf{h}^{L-1} + \mathbf{b}^L)$$

   where $\mathbf{h}^L$ is the activation of the $L$-th layer, $\mathbf{W}^L$ and $\mathbf{b}^L$ are the weights and 
   biases of the $L$-th layer, respectively, and $f^L$ is the activation function.
   The final output of the deep component, $ y_{\text{deep}} $, is the output of the last layer of the deep neural network. It is given by:

   $$
   y_{\text{deep}} = \mathbf{W}^O \mathbf{h}^{L} + \mathbf{b}^O
   $$

   where $ \mathbf{W}^O $ and $ \mathbf{b}^O $ are the weights and bias of the output layer, and $ \mathbf{h}^L $ is the 
   activation of the last hidden layer.

The final prediction is a combination of the outputs from the FM and deep components:

$$\hat{y} = \sigma(y_{\text{FM}} + y_{\text{deep}})$$

where $\sigma$ is the sigmoid function used to squash the output to a probability score.

<p align="center">
   <img src="/recommendation_model/deepfm.png" width="600" height="400"><br>
   <em>Figure 4: deepfm model</em>
</p>

### Deep & Cross Network v2 (DCN v2)[^6]
DCN-V2 starts with an embedding layer, followed by a cross network containing multiple cross layers that models explicit 
feature interactions, and then combines with a deep network that models implicit feature interactions. The function class 
modeled by DCN-V2 is a strict superset of that modeled by DCN. The overall model architecture is depicted in Fig. 5, with 
two ways to combine the cross network with the deep network: (1) stacked and (2) parallel.

1. **Cross Component**: The Cross Network V2 enhances the original cross network by introducing a more flexible mechanism 
to capture feature interactions. The cross layer in DCN V2 can be represented as:

   $$
   x_{l+1} = x_0 \odot (W_l \cdot x_l + b_l) + x_l
   $$

   where $x_l$ is the input to the l-th cross layer. $W_l$ and $b_l$ are the weight matrix and bias vector of the l-th 
   cross layer. $x_{l+1}$ is the output of the $l+1$-th cross layer. $\odot$ is the Hadamard product.

2. **Deep Component**: The deep network in DCN V2 captures high-order feature interactions through a series of dense layers. 
This part of the network learns abstract representations of the input features, enabling the model to generalize well.

   $$ 
   h_{l} = f_{l}(W_{l} h_{l-1} + b_{l})
   $$

   where $h_{l}$ is the activation of the $l$-th layer, $W_{l}$ and $b_{l}$ are the weights and 
   biases of the $l$-th layer, respectively, and $f_{l}$ is the activation function.
   
The final prediction is a combination of the outputs from either the hidden layer (stacked structure) or the  concatenation
of cross and deep network outputs.

<p align="center">
   <img src="/recommendation_model/dcn_v2.png" width="600" height="400"><br>
   <em>Figure 5: deep cross network v2</em>
</p>

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
[^4]: [Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016](https://arxiv.org/pdf/1606.07792)
[^5]: [Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017)](https://arxiv.org/pdf/1703.04247)
[^6]: [Wang, Ruoxi, et al. "Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021](https://arxiv.org/pdf/2008.13535)