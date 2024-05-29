---
title: "Causality"
date: 2024-05-28T16:19:16-07:00
draft: false
description: "An introduction to causality inference"
tags: ['causal inference', 'statistics']
---

This is a test post for the causality section.

<!--more-->

## What Is Causality?
Different from common ML tasks that focus on summarizing the patterns in the data, expecting to make predictions on 
similar unseen data. Causality, on the other hand, is about understanding the underlying mechanisms that drive the data. 
It is about to predict how the people/agent/system would react if it were intervened in a certain way. Think about the 
example of having two parallel universes with different light, temperature, and humidity conditions. What would happen 
to a plant if it were moved from one universe to another? This is a causal question.

### Causal Formulation
To understand how the outcome would change in different conditions, we need to define the following notations to 
describe the causal relations.

- $Y_i$: the outcome of interest for individual $i$. For example, the health status of a patient.
- $A_i$: an indicator of whether the individual $i$ received the treatment. For example, a new drug.
- $X_i$: a set of covariates that may affect both the treatment assignment and the outcome. For example, health status, 
age, gender etc
- $U_i$: unobserved factors that may affect both the treatment assignment and the outcome. For example, genetic factors.
- $Y_i^{a}$: the potential outcome of individual $i$ if he/she received the treatment $A = a, a \in \\{0, 1\\}$. For 
example, the health status of a patient if he/she received the new drug ($a=1$).

In practice, the data $(Y_i, A_i, X_i)$ is usually drawn i.i.d from a population and they are formulated into a causal 
graph as shown below.
![simple_causal_model_diagram](/causality/simple_causal_model_diagram.png)
The mechanism of the causal graph is often reasoning by structural causal models (SCM). It models the underlying data 
generation process with an ordered sequence of functions that map the parent nodes to the child nodes. For example, the 
SCMs of the above causal graph can be written as:
$P(Y, A, X) = P(Y | A, X) P(X | A) P(A)$. This equation describes the probabilistic dependencies between the observed 
variables. Now the question is how do we define the effect of the new drug $A$ on health outcome $Y$?

### Average Treatment Effect
Let's continue with the above drug trail context and notations. Consider a simple case with a binary treatment $A$ 
(1: treated, 0: untreated) and a binary outcome $Y$ (1: recovered, 0: not recovered). $Y^{a=1}$ (Y under the treatment 
a = 1) be the outcome

## Causal Effect Estimation

### Randomized Controlled Trials

### Causal Inference

#### Inverse Probability Weighting

#### Standardization

#### Stratification

#### Propensity Score Matching

#### Instrumental Variables

#### Difference-In-Differences

## Root Cause Optimization


## Appendix

### Causal Hierarchy: Observation, Intervention, Counterfactual

#### Causation vs Association

#### Intervention vs Counterfactual

### Do Calculus

## Reference





