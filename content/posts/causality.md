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

![causal_model_diagram](/causality/causal_model_diagram.png)
The mechanism of the causal graph is often reasoning by structural causal models (SCM). It models the underlying data 
generation process with an ordered sequence of functions that map the parent nodes to the child nodes. For example, the 
SCMs of the above causal graph can be written as:
$P(Y, A, X) = P(Y | A, X) P(X | A) P(A)$. This equation describes the probabilistic dependencies between the observed 
variables. Now the question is how do we define the effect of the new drug $A$ on health outcome $Y$?

### Average Treatment Effect
Let's continue with the above drug trail context and notations. Consider a simple case with a binary treatment $A$ 
(1: treated, 0: untreated) and a binary outcome $Y$ (1: recovered, 0: not recovered). 
- $Y^{a=1}$ (Y under the treatment 
a = 1) be the outcome variable that would have been observed if the patient had received the treatment. 
- $Y^{a=0}$ (Y 
under the treatment a = 0) be the outcome variable that would have been observed if the patient had not received the
treatment. 

When estimating the causal effect on the **population**, it's usually to consider the average treatment effect (ATE), 
$ATE = E[Y^{a=1}] - E[Y^{a=0}]$, denoting the average causal effect of the treatment on the outcome.

## Causal Effect Estimation
With the causal formulation and the average treatment effect defined, the next step is to estimate the causal effect
from the observed data. Multiple methods have been developed for this purpose, each relying on different assumptions 
and is suitable for different scenarios. Some of the most common methods include: _randomized controlled trials (A/B 
testing)_, _confounder adjustments methods_, _instrumental variables_, and _difference-in-differences_ etc.

### Randomized Controlled Trials (A/B Testing)
The most common approach to estimate the causal effect is to run an experiment, usually in the following general steps.
1. Objective: Define the research question and the hypothesis (usually built with treatment metrics) to be tested.
2. Design: Randomly assign the subjects into two groups, the treatment group (A) and the control group (B). The 
treatment group receives the treatment, while the control group does not.
3. Randomization: Randomly assign the subjects to the treatment and control groups to ensure the groups are comparable.
4. Run test: Implement the A/B test and let it run for a sufficient period to gather an adequate amount of data
5. Analysis: Analyze the data and calculate the treatment effect.
6. Decisions: Make decisions based on the test results.

Randomized controlled trials are considered the gold standard for causal inference because they can eliminate the 
confounding bias by randomizing the treatment assignment. However, they are not always feasible due to ethical, 
practical, or financial reasons. Therefore, other methods are developed to estimate the causal effect from observational
data.

### Causal Inference - Confounder Adjustments Methods
Before diving into the confounder adjustments methods, let's first clarify the confounding bias. Consider the following 
example: 
![confounder_selection_bias](/causality/confounder_selection_bias.png)
A study is conducted to investigate the effect of a new drug on the health outcome of patients. The study 
finds that patients who received the new drug have a higher recovery rate than those who did not. However, the study 
also finds that the patients who received the new drug are younger and healthier than those who did not. Therefore, the
observed effect of the new drug on the health outcome may be **confounded** by the age and health status of the patients.
To estimate the causal effect of the new drug on the health outcome, we need to _adjust_ for the confounding factors, 
such as age and health status.

#### Key Assumptions
All the confounder adjustments methods rely on the following key assumptions:
- **Exchangeability**: the treatment assignment is independent of the potential outcomes given the covariates.
- **Positivity**: the probability of receiving the treatment is positive for all levels of the covariates.
- **Consistency**: the individual's potential outcome is the same as the observed outcome when the treatment is received.

#### Inverse Probability Weighting
- Key Idea: Assigning weights to observations in order to create a pseudo-population that approximates the population 
that would have been obtained through random assignment in a randomized control trial (RCT).
- Steps:
  1. Estimate the propensity score, $pps(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Calculate the inverse probability weights, $w_i = \frac{P(A_i)}{pps(x_i)}$.
  3. Fit a marginal structure model $E[Y^a] = \beta_0 + \beta_1 a$ with the inverse probability weights $w_i$.
  4. Estimate the average treatment effect, $ATE = \hat{\beta_1}$.

#### Standardization
- Key Idea: Standardize the outcome under different treatment conditions to estimate the average treatment effect.
- Steps:
  1. Fit the outcome model with treatment and covariates, $P[Y^a | X=x] = P(Y | A=a, X=x)$. For example, the linear model
representation is $Y^A = \beta_0 + \beta_1 A + \beta_2 X$.
  2. Estimate the expectation of the outcome under different treatment conditions: 
$E[Y^a] = \int_x P(Y|A=a, x)*f_X(x)dx$.
  3. Calculate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

#### Stratification
- Key Idea: Stratify the data into different strata based on the covariates and estimate the average treatment effect.
- Steps:
  1. Estimate the propensity score, $pps(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Fit the outcome model with treatment and propensity score as the covariates, 
$P[Y^a | X=x] = P(Y | A=a, pps(x))$. A simple linear representation is $Y^A = \beta_0 + \beta_1 A + \beta_2 * pps(X)$.
  3. Estimate the expectation of the outcome under different treatment conditions:
$E[Y^a] = \int_x P(Y|A=a, pps(x))*f_X(x)dx$
  4. Calculate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

#### Propensity Score Matching
- Key Idea: Match the treated and control units based on the propensity score to estimate the average treatment effect.
- Steps:
  1. Estimate the propensity score, $pps(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Match the treated and control units based on the propensity score. (Nearest Neighbour, Kernel Matching, etc)
  3. Estimate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

### Causal Inference - Instrumental Variables

### Difference-In-Differences

## Root Cause Optimization


## Appendix

### Causal Hierarchy: Observation, Intervention, Counterfactual

#### Causation vs Association

#### Intervention vs Counterfactual

### Do Calculus

## Reference

https://www.youtube.com/watch?v=Od6oAz1Op2k





