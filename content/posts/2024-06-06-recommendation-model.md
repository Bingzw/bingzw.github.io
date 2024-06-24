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

### Deep Learning Recommendation Model (DLRM)[^6]
Deep Learning Recommendation Model (DLRM) is an advanced machine learning framework designed by Facebook AI to tackle 
the complex challenge of personalized recommendations at scale. It is particularly suited for large-scale recommendation 
systems in environments such as social media platforms, e-commerce, and online advertising. The DLRM combines the 
strengths of collaborative filtering and content-based methods by utilizing both dense and sparse features to provide 
highly accurate and scalable recommendations.

<p align="center">
   <img src="/recommendation_model/dlrm.png" width="600" height="400"><br>
   <em>Figure 5: deep learning recommendation model</em>
</p>

Key components of DLRM includes
1. **Sparse Features**: These are categorical variables (e.g., user ID, item ID) which are typically represented using 
embeddings. Embeddings transform sparse categorical data into dense vectors of continuous numbers, making them suitable 
for neural network processing, denoted as $\mathbf{x}_s[i]$.
The raw sparse features are transformed to sparse embeddings as follows. Let
$$
\mathbf{e}_i = \text{Embedding}(\mathbf{x}_s[i])
$$
where $\mathbf{e}_i$ is the embedding vector for the $i$-th sparse feature, $\text{Embedding}$ denotes an embedding lookup table

2. **Dense Features**: These are numerical variables (e.g., user age, item price) that are used directly in their raw 
form or normalized form, denoted as $\mathbf{x}_d$.

3. **Bottom MLP (Multilayer Perceptron)**: Processes the dense features to capture high-level representations.
$$
\mathbf{h}_d = \text{BottomMLP}(\mathbf{x}_d)
$$

4. **Interaction Layer**: This layer captures the interactions between different features (both sparse and dense). It 
uses a dot product to compute the pairwise interactions among features.
$$
\mathbf{z} = \left[ \mathbf{h}_d, \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n \right]
$$
where $\mathbf{z}$ is the concatenated vector of dense feature representation and embeddings.

5. **Top MLP**: Combines the processed dense features and interactions from the interaction layer to make the final prediction.
$$
\hat{y} = \sigma(\text{TopMLP}(\mathbf{z}))
$$
where $\sigma$ is an activation function, typically a sigmoid function for binary classification tasks.

### Deep & Cross Network v2 (DCN v2)[^7]
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
   <em>Figure 6: deep cross network v2</em>
</p>

### Deep Interest Network (DIN)[^8]
Deep Interest Network (DIN) is a neural network-based model designed for personalized recommendation systems, particularly 
in the context of e-commerce and advertising. Unlike traditional recommendation models that primarily focus on user-item interactions, 
DIN leverages a user's historical behavior to make more accurate and contextually relevant recommendations. The key 
innovation in DIN is its ability to capture user interests dynamically and use this information to influence the recommendation 
process.

<p align="center">
   <img src="/recommendation_model/din.png" width="800" height="600"><br>
   <em>Figure 7: deep interest network</em>
</p>

In particular, the DIN contains the following key components: 

1. **Embedding Layer**: User behaviors and items are represented as dense vectors through an embedding layer. Let 
$(\{e_{1}, e_{2}, \ldots, e_{n}\})$ be the sequence of embeddings for user behaviors, and $e_{target}$ be the embedding of the target item.

2. **Local Activation Unit**: This unit applies the attention mechanism to compute the relevance of each user behavior 
in the sequence with respect to the target item. 
Let 
$$
\alpha_{i} = \frac{\exp(\text{score}(e_{i}, e_{target}))}{\sum_{j=1}^{n} \exp(\text{score}(e_{j}, e_{target}))}
$$
where $\text{score}(e_{i}, e_{target})$ is the activation weight output from a feed-forward
network measuring the relevance of behavior $e_{i}$ to the target item.

3. **Interest Extractor Layer**: Combines the weighted behavior embeddings to form a user interest representation.
$$
u = \sum_{i=1}^{n} \alpha_{i} e_{i}
$$

4. **Prediction Layer**: The user interest representation is concatenated with the target item embedding and fed into a neural network to predict the user's interaction with the target item.
$$
\hat{y} = \sigma(W[u, e_{target}, e_{profile}, e_{context}] + b)
$$
where $[u, e_{target}, e_{profile}, e_{context}]$ denotes the concatenation of the user interest representation, profile
embedding, context embedding and the target item embedding, $W$ is a weight matrix, $b$ is a bias term, and $\sigma$ is an activation function (e.g., sigmoid).

### Multi-gate Mixture of Experts (MMOE)[^9]
Multi-gate Mixture-of-Experts (MMoE) is an advanced multi-task learning (MTL) framework designed to model and leverage 
task relationships for improved performance across multiple tasks. The MMoE architecture combines the benefits of 
mixture-of-experts models with the flexibility of multi-gate mechanisms, allowing the model to dynamically allocate 
computational resources based on the specific needs of each task. This approach is particularly useful in scenarios 
where tasks are interrelated and can benefit from shared learning while maintaining task-specific adaptations.

<p align="center">
   <img src="/recommendation_model/mmoe.png" width="800" height="600"><br>
   <em>Figure 8: multi-gate mixture of experts</em>
</p>

Key components of the model architecture are:
1. **Experts**: A set of neural network sub-models that serve as specialized units for feature extraction. Each expert 
is trained to capture different aspects of the input data. Let $\mathbf{x}$ represent the input data, $E$ represent 
the number of experts, and $T$ represent the number of tasks. The output from expert is represented as
$$
\mathbf{h}^i = f_e^i(\mathbf{x}), \quad \text{for } i = 1, 2, \ldots, E
$$
where $f_e^i$ is the function of the $i$-th expert, and $\mathbf{h}^i$ is the output of the $i$-th expert.

2. **Multi-gate Mechanism**: Separate gating networks for each task that dynamically select and weight the contributions 
of different experts based on the input data and task requirements.
$$
\mathbf{g}^j = \sigma(\mathbf{W}^j \mathbf{x}), \quad \text{for } j = 1, 2, \ldots, T
$$
where $\mathbf{W}^j$ are the parameters of the gating network for task $j$, and $g^j$ is the gating output distribution 
of gating weights to each expert assigned by task $j$. To combine the experts output and gating weights, we have
$$\mathbf{t}^j = \sum_{i=1}^{E} g_i^j \cdot \mathbf{h}^i$$
where $\mathbf{t}^j$ is the combined output for task $j$, and $g_i^j$ is the weight for the $i$-th expert assigned by 
the gating network of task $j$.

3. **Task-specific Layers**: Layers that process the combined outputs from the experts, tailored to the specific requirements of each task.
$$
\hat{y}^j = f_o^j(\mathbf{t}^j)
$$
where $f_o^j$ is the task-specific output layer for task $j$, and $\hat{y}^j$ is the predicted output for task $j$.

### Behavior Sequence Transformer (BST)[^10]
The Behavior Sequence Transformer (BST) is a novel neural network architecture designed for modeling user behavior 
sequences in recommendation systems. BST leverages the power of Transformer models, which have achieved significant 
success in natural language processing (NLP), to capture the sequential patterns and contextual dependencies in user 
interactions over time. This approach enhances the ability to predict user preferences and improve recommendation accuracy.

<p align="center">
   <img src="/recommendation_model/bst.png" width="800" height="600"><br>
   <em>Figure 9: behavior sequence transformer</em>
</p>

Key components of BST include:
1. **Input Layer**: 
- **User Behavior Sequence**: A sequence of items or actions representing user interactions over time.
- **Contextual Features**: Additional information such as timestamps, device types, and other relevant context.

2. **Embedding Layer**:
$$
\mathbf{E}_x = \mathbf{W}_e \cdot \mathbf{x}
$$
where $\mathbf{W}_e$ is the embedding matrix and $\mathbf{x}$ is the input feature vector.

3. **Positional Encoding**:
$$
\mathbf{E}_p = \text{PE}(pos)
$$
where $\text{PE}(pos)$ is the positional encoding function that adds position-specific information to the embeddings.

4. **Transformer Encoder**:
$$
\mathbf{H}_i = \text{LayerNorm}(\text{MultiHeadAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) + \mathbf{E}_i)
$$
where $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are the query, key, and value matrices, and $\mathbf{E}_i$ is the 
input embedding at layer $i$. A Feed-Forward Networks ($\text{FFN}$) is added on top of the self attention layer to further enhance the
model with non-linearity.
$$
\mathbf{O}_i = \text{LayerNorm}(\text{FFN}(\mathbf{H}_i) + \mathbf{H}_i)
$$
In practice, we usually stack multiple transformer layers. So the output of last layer is thus fed as the input of the 
next layer.

5. **Output Layer**:
$$
\hat{y} = \sigma(\mathbf{W}_o \cdot \mathbf{O}_L)
$$
where $\mathbf{W}_o$ is the weight matrix of the output layer, $\mathbf{O}_L$ is the output of the last Transformer encoder layer, and $\sigma$ is the activation function.

## Summary
So far, we have summarized the main popular recommendation models. These summaries highlight the unique strengths and 
specific applications of each recommendation model, reflecting their advancements and contributions to the field. The 
modern recommendation systems usually are built with multiple stages and are composed with both simple and complex models. 
Users may choose the best model depending on the business requirements and system architecture design. 

## Citation
If you find this post helpful, please consider citing it as:
```bibtex
@article{wang2024recommendationmodel,
  author = "Bing Wang",
  title = "Recommendation Models",
  journal = "bingzw.github.io",
  year = "2024",
  month = "June",
  url = "https://bingzw.github.io/posts/2024-06-06-recommendation-model/"
}
```
or 
```markdown
Bing Wang. (2024, June). Recommendation Models. 
https://bingzw.github.io/posts/2024-06-06-recommendation-model/
```

## Reference
[^1]: [Collaborative Filtering and Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
[^2]: [Kuchaiev, Oleksii, and Boris Ginsburg. "Training deep autoencoders for collaborative filtering." arXiv preprint arXiv:1708.01715 (2017)](https://arxiv.org/pdf/1708.01715)
[^3]: [He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017](https://arxiv.org/pdf/1708.05031)
[^4]: [Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016](https://arxiv.org/pdf/1606.07792)
[^5]: [Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017)](https://arxiv.org/pdf/1703.04247)
[^6]: [Naumov, Maxim, et al. "Deep learning recommendation model for personalization and recommendation systems." arXiv preprint arXiv:1906.00091 (2019)](https://arxiv.org/pdf/1906.00091)
[^7]: [Wang, Ruoxi, et al. "Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021](https://arxiv.org/pdf/2008.13535)
[^8]: [Zhou, Guorui, et al. "Deep interest network for click-through rate prediction." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018](https://arxiv.org/pdf/1706.06978)
[^9]: [Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
[^10]: [Chen, Qiwei, et al. "Behavior sequence transformer for e-commerce recommendation in alibaba." Proceedings of the 1st international workshop on deep learning practice for high-dimensional sparse data. 2019](https://arxiv.org/pdf/1905.06874)