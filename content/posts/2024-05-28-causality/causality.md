---
title: "Causality Introduction"
date: 2024-05-28T16:19:16-07:00
draft: false
description: "An introduction to causality inference"
tags: ['causal inference', 'statistics']
---
![causality_icon](/causality/causality_icon.png)
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
graph as shown below[^1].

<p align="center">
    <img src="/causality/causal_model_diagram.png" width="600" height="400"><br>
    <em>Figure 1: causal inference elements</em>
</p>

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

<p align="center">
    <img src="/causality/confounder_selection_bias.png"><br>
    <em>Figure 2: confounder example</em>
</p>

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

#### General Flow of Causal Inference

In general, causal inference is to build a pseudo-population that mimics the randomized experimentation from the observed
data. The assumptions taken and the modeling approaches applied are to ensure the pseudo-population is unbiased and
consistent with the true population. The general flow of causal inference can be summarized as follows:

   1. **Define the Problem**: What are the causal effect of interest? What are the treatment and outcome variables?
   2. **Acquire Data**: Collect the data that contains the treatment, outcome, and covariates.
   3. **Build the Causal Model**: Fit the causal model that describes the causal relationship
   4. **Estimate the Causal Effect**: Estimate the counterfactual causal effect using the trained model

<p align="center">
    <img src="/causality/causal_inference_flow.png" width="800" height="500"><br>
    <em>Figure 3: Causal Inference Modeling Flow</em>
</p>
We follow the same modeling process in figure 3. First defining the outcome, treatment, and covariates, then building the
causal model using the sample data. Next to estimate the counterfactual performance for both treated and untreated in the
treatment and control groups. The dashed figure in the above graph represents the unobserved counterfactuals that are 
estimated from the counterfactual model. Finally, the causal effect is then estimated by comparing the observed outcome 
with the estimated counterfactual outcome.

Let's dive deep into how the causal model[^3] can be trained from the observed data.

#### Inverse Probability Weighting
- Key Idea: Assigning weights to observations in order to create a pseudo-population that approximates the population 
that would have been obtained through random assignment in a randomized control trial (RCT).
- Steps:
  1. Estimate the propensity score, $g(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Calculate the inverse probability weights, $w_i = \frac{P(A_i)}{g(x_i)}$.
  3. Fit a marginal structure model $E[Y^a] = \beta_0 + \beta_1 a$ with the inverse probability weights $w_i$.
  4. Estimate the average treatment effect, $ATE = \hat{\beta_1}$.

#### Standardization
- Key Idea: Standardize the outcome under different treatment conditions to estimate the average treatment effect.
- Steps:
  1. Fit the outcome model with treatment and covariates, $P[Y^a | X=x] = P(Y | A=a, X=x)$. For example, the linear model
  representation is $Y^A = \beta_0 + \beta_1 A + \beta_2 X + \epsilon$.
  2. Estimate the expectation of the outcome under different treatment conditions: 
  $E[Y^a] = \int_x P(Y|A=a, x)*f_X(x)dx$.
  3. Calculate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

#### Stratification
- Key Idea: Stratify the data into different strata based on the covariates and estimate the average treatment effect.
- Steps:
  1. Estimate the propensity score, $g(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Fit the outcome model with treatment and propensity score as the covariates, 
  $P[Y^a | X=x] = P(Y | A=a, g(x))$. A simple linear representation is $Y^A = \beta_0 + \beta_1 A + \beta_2 * g(X) + \epsilon$.
  3. Estimate the expectation of the outcome under different treatment conditions:
  $E[Y^a] = \int_x P(Y|A=a, g(x))*f_X(x)dx$
  4. Calculate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

#### Propensity Score Matching
- Key Idea: Match the treated and control units based on the propensity score to estimate the average treatment effect.
- Steps:
  1. Estimate the propensity score, $g(x) = P(A = 1 | X = x)$, the probability of receiving the treatment given the 
  covariates.
  2. Match the treated and control units based on the propensity score. (Nearest Neighbour, Kernel Matching, etc)
  3. Estimate the average treatment effect, $ATE = E[Y^{a=1}] - E[Y^{a=0}]$.

#### Double Machine Learning (DML)
In all above methods, we have observed that two components are needed to estimate the causal effect: the propensity score
$g(x) = P(A = 1 | X = x)$ and the outcome model $P[Y^a | X=x] = P(Y | A=a, X=x)$. However, the estimation of these
components may be biased, leading to biased estimates of the causal effect. To address this issue, double machine learning
(DML) is proposed.
- Key Idea: Use machine learning algorithms to estimate the nuisance functions and statistical estimation techniques to
estimate the causal effect.
- Steps:
  1. Divide the data into K-folds (cross fitting). For each fold $i$, train the propensity score model and the outcome model on
  folds other than $i$, i.e. $g^{-i}(x)$ and $P^{-i}(Y^a | X)$. 
  2. Estimating the augmented inverse probability of treatment weighted estimator (AIPTW) for fold $i$: 
$\hat{\tau_i} = \frac{1}{n_i} \sum_{i} \hat{P}^{-i}(Y^{a=1} | x_i) - \hat{P}^{-i}(Y^{a=0} | x_i) + A_i\frac{Y_i - \hat{P}^{-i}(Y^{a=1} | x_i)}{\hat{g}^{-i}(x_i)} - (1-A_i)\frac{Y_i - \hat{P}^{-i}(Y^{a=0} | x_i)}{1-\hat{g}^{-i}(x_i)}$
  3. Aggreagate the estimates from all folds to get the final estimate of the causal effect, $\hat{\tau} = \frac{1}{N} \sum_{i} n_i \hat{\tau_i}$.


### Causal Inference - Instrumental Variables
All above approaches rely on the assumption of no unobserved confounding, which may not be satisfied in practice. When 
there is potential endogeneity or unobserved confounding in observational data, instrumental variables can be used to
estimate the causal effect.
- Key Idea: Estimating the causal effect of a treatment or intervention when there is potential endogeneity or 
unobserved confounding in observational data. It relies on the use of an instrumental variable, which is a variable 
that is correlated with the treatment but not directly with the outcome, except through its influence on the treatment.
- Assumptions:
  1. **Relevance**: the instrumental variable is correlated with the treatment.
  2. **Exogeneity**: the instrumental variable is independent of the unobserved confounders.
  3. **Exclusion Restriction**: the instrumental variable affects the outcome only through the treatment.
- Steps:
  1. Estimate the first-stage regression: $A = \alpha_0 + \alpha_1 Z + \epsilon_1$.
  2. Estimate the second-stage regression: $Y = \beta_0 + \beta_1 \hat{A} + \epsilon_2$.
  3. Calculate the average treatment effect, $ATE = \hat{\beta_1}$.

So far, we have discussed the major methods for estimating the causal effect from observational data. Each method has its
own assumptions and limitations, there is no clear winner and the choice of method depends on the specific context and 
data characteristics. In practice, it is often recommended to use multiple methods to estimate the causal effect and
compare the results to ensure the robustness of the conclusions.

## Root Cause Optimization
In the above sections, we designed a hypothetical causality problem of estimating the effect of taking a new drug. The 
treatment variable $A$ can be defined as $A_i = 1$ if the patient takes the new drug and $A_i = 0$ if the patient does not.
More generally, the treatment variable can be formulated in this way $A = 1_{\{f(\theta) < \mu\}}$, where $f(\theta)$ is 
the function that maps the treatment parameters $\theta$ to the treatment assignment, and $\mu$ is the threshold that 
determines the treatment assignment. $f(\theta)$ is often a known metric since causal inference usually starts with some 
suspected causal mechanisms. However, the threshold $\mu$ is often unknown and needs to be optimized to maximize the
causal effect. Therefore, the root cause optimization is to find the optimal threshold $\mu$ that maximizes the causal
effect.
<p align="center">
$argmax_{\mu} E[Y^{A=1}] - E[Y^{A=0}]$, 
where $A = 1_{\{f(\theta) < \mu\}}$ with given $f(\theta)$
<p>

To solve the optimization problem, we can use the causal inference methods discussed above to estimate the causal effect
under different thresholds and find the optimal threshold that maximizes the causal effect. Popular optimization algorithms
such as grid search, random search and bayesian optimization can be used to find the optimal threshold.

## Appendix
### Do Calculus
While we formulate the causal effect from the counterfactual perspective, there is another powerful tool that called 
do-calculus that estimates the causal effect by setting interventions on the treatment variable. In particular, it's 
trying to apply interventions $A=a$ to the causal graph that deletes the edges pointing to $A$, and then conditioning on
the observed variables to estimate the causal effect. The average treatment effect can be written as:
<p align="center">
$ATE = E[Y^{a=1}] - E[Y^{a=0}] = E[Y|do(A=1)] - E[Y|do(A=0)]$
<p>

Please refer to the do-calculus literature[^2] for more details.

## Citation
If you find this post helpful, please consider citing it as:
```bibtex
@article{wang2024causality,
  author = "Bing Wang",
  title = "Causality Introduction",
  journal = "bingzw.github.io",
  year = "2024",
  month = "May",
  url = "https://bingzw.github.io/posts/2024-05-28-causality/causality/"
}
```
or 
```markdown
Bing Wang. (2024, May). Causality Introduction. 
https://bingzw.github.io/posts/2024-05-28-causality/causality
```

## Reference
[^1]: [Causal Inference: Explained](https://www.youtube.com/watch?v=Od6oAz1Op2k)
[^2]: Pearl, Judea (2000). Causality: Models, Reasoning, and Inference.
[^3]: [Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)









