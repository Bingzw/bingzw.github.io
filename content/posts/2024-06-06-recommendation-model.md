---
title: "Recommendation Model"
date: 2024-06-06T22:04:53-07:00
draft: true
description: "An introduction to recommendation models"
tags: ['Machine learning', 'Deep Learning', 'Recommendation']
---

<!--more-->

## Introduction to Recommendation Models

In the digital age, the sheer volume of available content and products can be overwhelming for users. Recommendation systems play a crucial role in helping users discover relevant items by filtering through vast amounts of data. These systems are integral to various industries, including e-commerce, streaming services, social media, and online advertising.

Recommendation models are algorithms designed to suggest items to users based on various factors such as past behavior, preferences, and item characteristics. The goal is to enhance user experience by providing personalized recommendations, thereby increasing engagement and satisfaction.

In most cases, the goal is trying to find the most appealing item for the user given all user demographic, behavior and interest information. A well-designed recommendation system usually results in better personalization, user engagement, and revenue generation. Let's take a look at the most popular modeling approaches.

## Early Models

### Collaborative Filtering (CF)

Collaborative Filtering is one of the most famous recommendation mdoels so far. It's about to guess the user's interest based on the behaviors and preferences of other users with similar tastes/characteristics.

#### Memory Based CF
Suppose the recommendation is purely derived from historical records and it does not involve any predictive modelings. Let's assume a table of all ratings of users on items. Let $r_{u,i}$ dentoes the missing rating of user $u$ on item $i$. $S_u$ be a set of users that share similar characteristics with user $u$. The goal is to guess the missing rating $r_{u,i}$. In particular, it can be derived from either user-user filtering or item-item filtering.
##### User Based CF
We are predicting the missing rating as the weighted average from ratings of similar users:
<p align="center">
$r_{u,i} = \bar{r}_u + \frac{\sum_{u'\in S_u}sim(u, u') * (r_{u',i} - \bar{r}_{u'})}{\sum_{u'\in S_u}sim(u, u')}$
<p>
where $\bar{r}_u$ is the average rating of user $u$ and $sim(u, u')$ represents the similarity score between user $u$ and $u'$. A common choice is to calculate the cosine simialrity between two user rating vectors.


#### Model Based CF

#### Hybrid

## Modern Recommendation Models

### Wide & Deep

### DeepFM

### Deep & Cross Network (DCN)

### Attention Factorization Machine (AFM)

### Deep Interest Network (DIN)

### Deep Learning Recommendation Model (DLRM)

### Entire Space Multi-Task Model (ESMM)

### Multi-gate Mixture of Experts (MMOE)

### Progressive Layered Extraction (PLE)

### Behavior Sequence Transformer (BST)
