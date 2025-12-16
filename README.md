# CME 295: Transformers & Large Language Models
## Comprehensive Study Guide: Lectures 5 & 6
### Preference Tuning & Reasoning Models

**Based on lectures by Afshine Amidi & Shervine Amidi**  
**Stanford University - Fall 2025**

---

# Table of Contents

1. [Introduction and Learning Objectives](#introduction)
2. [Part I: Preference Tuning (Lecture 5)](#part-i-preference-tuning)
   - Chapter 1: The Need for Preference Tuning
   - Chapter 2: Preference Data
   - Chapter 3: RLHF
   - Chapter 4: DPO
3. [Part II: Reasoning Models (Lecture 6)](#part-ii-reasoning-models)
   - Chapter 5: Limitations of Vanilla LLMs
   - Chapter 6: Chain of Thought
   - Chapter 7: GRPO and Test-Time Scaling
   - Chapter 8: DeepSeek R1
4. [Part III: Synthesis and Comparison](#part-iii-synthesis)
5. [Key Formulas Reference](#key-formulas)

---

# Introduction

Welcome back! This comprehensive guide covers two critical lectures in the LLM training pipeline: **Preference Tuning** (Lecture 5) and **Reasoning Models** (Lecture 6). Since you're returning after a 2-month hiatus, each concept will be explained with the necessary background.

## The LLM Training Pipeline: A Quick Refresher

Before diving into the details, let's refresh our understanding of the complete LLM training pipeline. This context is essential because preference tuning and reasoning training build upon earlier stages.

**The pipeline consists of three major stages:**

1. **Pretraining**: Starting from an initialized model (random weights), we train on massive amounts of text data using next-token prediction. The model learns to predict what word comes next given all previous words. This gives the model "basic knowledge" about language, code, facts, and patterns. Think of this as the model learning to read and understand the world. Pretraining typically uses trillions of tokens and creates what we call a "base model."

2. **Finetuning (Supervised Fine-Tuning or SFT)**: The pretrained model is then trained on curated instruction-following data—pairs of (instruction, desired_response). This transforms it from a text completion engine into an assistant that can follow instructions. The model learns what a "good response" looks like for different types of requests.

3. **Preference Tuning**: Finally, we align the model with human preferences, teaching it not just what to say, but how to say it in a way that humans prefer. This is where the model learns to be helpful, harmless, and honest—the focus of Lecture 5.

## What You'll Learn

By the end of this guide, you will understand:
- Why preference tuning is necessary and how it differs from supervised learning
- The complete RLHF pipeline including reward modeling and policy optimization
- Mathematical foundations: Bradley-Terry model, PPO, DPO, and GRPO
- How reasoning models work and the concept of test-time scaling
- The DeepSeek R1 training recipe and its innovations
- Practical trade-offs between different alignment approaches

---

# Part I: Preference Tuning (Lecture 5)

## Chapter 1: The Need for Preference Tuning

### 1.1 Why SFT Isn't Enough

Even after supervised fine-tuning (SFT), models can "misbehave." The SFT model learns from examples of good responses, but it doesn't explicitly learn what BAD responses look like. This means it can still generate problematic outputs.

**Example from the lectures:**

> **User**: "Suggest a new activity I could do with my teddy bear."
>
> **SFT Model (Bad Response)**: "I'd suggest you do not spend much time with your teddy bear at all."

This response is technically coherent English but unhelpful and dismissive. The model hasn't learned that this type of response is undesirable. This is where preference tuning comes in—we need to **inject negative signals** to teach the model what NOT to do.

### 1.2 The Core Idea: Learning from Comparisons

The fundamental insight behind preference tuning is that **it's easier for humans to compare two responses ("A is better than B") than to generate perfect responses from scratch**. This comparative approach scales better and provides clearer training signals.

**Three key reasons why preference tuning matters:**

1. **Comparison is easier than generation**: Humans can quickly judge "A is better than B" but struggle to articulate exactly what a perfect response should contain. Try it yourself—it's much easier to say "this essay is better than that one" than to write the perfect essay from scratch.

2. **SFT data distribution is fragile**: Even small inconsistencies in SFT training data can cause the model to "mess up." The distribution of training examples heavily influences model behavior in unpredictable ways. If your SFT data has even a few bad examples, the model might learn to imitate them.

3. **Quality matters enormously**: SFT data quality is critical but hard to guarantee at scale. Preference tuning provides a more robust way to inject quality signals. However, the lectures note that "model misbehaving" can also be a good wake-up call to check SFT data quality.

---

## Chapter 2: Preference Data—The Foundation

### 2.1 What is Preference Data?

Preference data consists of **observations** where each observation is a (prompt, response) pair. The key is how we organize and label these observations to create training signal.

**Three Types of Preference Data:**

| Type | Description | Example |
|------|-------------|---------|
| **Pointwise** | Each response gets an absolute score independently | Obs1=0.4, Obs2=0.9, Obs3=0.1, Obs4=0.2 |
| **Pairwise** | Responses compared in pairs | Obs1 < Obs2, Obs1 > Obs3, Obs2 > Obs4 |
| **Listwise** | Complete rankings of multiple responses | Obs2 > Obs1 > Obs4 > Obs3 |

**Pairwise data is the most common format** and what RLHF/DPO typically use. It's more reliable because humans are better at relative judgments than absolute ones.

### 2.2 How to Collect Pairwise Preference Data

The recipe involves two main steps:

**Step 1: Generate pairs of responses for the same prompt**
- **Input prompts**: Obtained from logs of real user interactions or from a reference distribution of questions
- **Output responses**: Generated by the SFT model using different sampling temperatures, synthetic generation, or human rewrites

**Step 2: Label which response is "chosen" (y_w) and which is "rejected" (y_l)**
- **Human rating**: Gold standard but expensive and time-consuming
- **Proxies**: LLM-as-a-judge, BLEU scores, ROUGE scores, or other automatic metrics
- **Scale variants**: Binary (just better/worse) vs. nuanced scales (1-5 ratings)

---

## Chapter 3: RLHF—Reinforcement Learning from Human Feedback

### 3.1 The RL Formulation for LLMs

Before diving into RLHF, let's understand how we frame language model training as a reinforcement learning problem.

**Quick Refresher on Reinforcement Learning:**
In RL, an agent takes actions in an environment to maximize cumulative reward. The agent has a policy (strategy for choosing actions) that it improves over time based on the rewards it receives.

**Mapping RL Concepts to LLMs:**

| RL Concept | LLM Interpretation |
|------------|-------------------|
| **Agent** | The LLM itself |
| **State (s_t)** | Input so far (prompt + tokens generated so far) |
| **Action (a_t)** | Next token to generate |
| **Environment** | The token vocabulary |
| **Policy π_θ(a_t\|s_t)** | Probability distribution over next tokens—**this IS the LLM!** |
| **Reward (r_t)** | Human preference signal |

**The Goal**: Learn a policy π_θ that generates responses aligned with human preferences while staying close to its original capabilities.

### 3.2 RLHF Overview: The Two-Step Process

RLHF was popularized by the **InstructGPT paper** (Ouyang et al., 2022). It made GPT-3 much more helpful and safe.

**Step 1 — Reward Modeling: "Distinguish good from bad!"**
- Input: (prompt x, response ŷ)
- Output: Quantitative score r(x, ŷ)

**Step 2 — Reinforcement Learning: "Align the model!"**
- Input: prompt x
- Output: response ŷ that maximizes reward while staying close to base model

### 3.3 Step 1: Reward Modeling—The Mathematical Foundation

#### The Bradley-Terry Model

The reward model is trained using the **Bradley-Terry formulation** (1952), a statistical model originally developed for ranking sports teams.

**Core Formula:**

$$p(y_i > y_j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)$$

Where:
- **r_i, r_j**: Reward scores for responses y_i and y_j
- **σ(x) = 1/(1 + e^(-x))**: The sigmoid function—maps any real number to (0, 1)

**Intuition**: When r_i >> r_j (response i has much higher reward), σ(r_i - r_j) → 1, meaning we're very confident i is better. When r_i ≈ r_j, σ(r_i - r_j) ≈ 0.5, representing uncertainty.

#### Training the Reward Model

**Loss Function:**

$$\mathcal{L}(\theta) = -\mathbb{E}[\log \sigma(r(x, \hat{y}_w) - r(x, \hat{y}_l))]$$

Where:
- **ŷ_w**: The "winner" (chosen/preferred response)
- **ŷ_l**: The "loser" (rejected response)

**This loss minimizes when**: r(x, ŷ_w) >> r(x, ŷ_l), i.e., the reward model correctly assigns much higher scores to preferred responses.

**Reward Model Specs:**
- **Data**: O(10,000) observations with human ratings
- **Architecture**: Pretrained LLM with classification head instead of next-token prediction head

### 3.4 Step 2: Reinforcement Learning—Policy Optimization

**The Goal**: Change the weights of the LLM to penalize bad answers and promote good answers using the frozen reward model.

**The Objective Function:**

$$\mathcal{L}(\theta) = \text{[Maximize rewards]} + \text{[Don't deviate too much from base model]}$$

**Why both terms?**

1. **Avoid "reward hacking"**: Without constraints, the model might find degenerate solutions that score high on the reward model but are nonsensical

2. **Training stability**: Large policy updates can destabilize training and cause the model to "forget" its language capabilities

**Training Specifications:**
- **Data**: O(100,000) observations—10x more than reward modeling
- **Labels**: Scores given by the (frozen) reward model
- **Initialization**: Start from the SFT model weights
- **Model states**: LLM (policy) is trainable 🔥; Reward Model is frozen ❄️

### 3.5 PPO: Proximal Policy Optimization

PPO (Schulman et al., 2017) is the most common RL algorithm used in RLHF.

#### Advantages, Not Just Rewards

PPO actually optimizes **advantages** rather than raw rewards:

$$\text{Advantage} \approx \text{Reward} - \text{Baseline (Value Function)}$$

**What is a Value Function?**
- Operates at the token level
- Estimates "what would be the expected reward if we continue following this policy?"
- Trained jointly with the policy
- Uses **Generalized Advantage Estimation (GAE)** for stability

**Why advantages?** Subtracting the baseline reduces variance in gradient estimates.

#### PPO-Clip: The Most Common Variant

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

Where:
- **r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)**: Ratio of new to old policy probabilities
- **Â_t**: Estimated advantage
- **ε**: Clipping hyperparameter (typically 0.1-0.2)

**Intuition**: The clipping prevents the ratio from going too far from 1, keeping updates conservative and stable.

#### PPO-KL Penalty: An Alternative

$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t - \beta \cdot \text{KL}[\pi_{\theta_{old}}(\cdot|s_t) || \pi_\theta(\cdot|s_t)]\right]$$

**Important Terminology:**
- **"old"**: Model from the previous RL iteration
- **"ref"**: Base/reference model (typically the SFT model, stays fixed)

### 3.6 Challenges with RL-Based Approaches

1. **Multi-stage complexity**: Requires training a separate reward model first
2. **Hyperparameter sensitivity**: Many parameters to tune
3. **Training instability**: RL training can be unstable
4. **Monitoring difficulty**: Hard to know what metrics to track
5. **Diversity requirements**: Need diverse completions to learn from
6. **Model overhead**: PPO needs 4 models: policy, value, reward model, base model

### 3.7 Best-of-N (BoN): A Workaround Without RL

**Core Idea**: Skip the RL step and leverage the reward model at inference time.

**Strategy:**
1. Given a prompt, generate N different outputs using the SFT model
2. Score each output using the reward model
3. Return the highest-scoring response

**Example:**

| Response | RM Score | Rank |
|----------|----------|------|
| "How about you both watch a movie together?" | 0.8 | 1 ✓ |
| "Take your teddy bear on a picnic" | 0.2 | 2 |
| "I'd suggest you do not spend much time..." | -2 | 3 |

**Trade-off**: N× inference cost but avoids all RL training complexity.

---

## Chapter 4: DPO—Direct Preference Optimization

### 4.1 Motivation: Can We Skip RL Entirely?

Given all the challenges with RL, researchers asked: "Can we train in a supervised fashion?"

**The DPO Insight** (Rafailov et al., 2023): *"Your Language Model is Secretly a Reward Model"*

DPO shows that we can derive a closed-form solution that bypasses explicit reward modeling and RL optimization.

### 4.2 The DPO Loss Function

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Where:
- **π_θ**: The policy we're training
- **π_ref**: The reference policy (typically SFT model, frozen)
- **y_w**: Preferred/winning response
- **y_l**: Dispreferred/losing response
- **β**: Temperature parameter

**Key Properties:**
- ✅ No separate reward model needed!
- ✅ Operates directly on preference data
- ✅ Similar to Bradley-Terry with implicit reward

### 4.3 Where Does DPO Come From?

**5-Step Derivation:**

1. **Start from PPO objective**: Maximize expected reward minus KL penalty
2. **Derive optimal policy**: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
3. **Identify implicit reward**: r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)
4. **Write Bradley-Terry**: The Z(x) terms cancel out!
5. **Infer DPO loss**: Take negative log likelihood

### 4.4 RLHF vs DPO Comparison

| Aspect | RLHF (PPO) | DPO |
|--------|------------|-----|
| Training stages | Multi-stage | Single-stage |
| Models needed | 4 (policy, value, reward, ref) | 2 (policy, ref) |
| Complexity | High | Low |
| Data | Generates new responses | Uses fixed preference dataset |
| Performance | Varies by task | Varies by task |

**Performance**: No universal consensus—results vary by task and implementation.

---

# Part II: Reasoning Models (Lecture 6)

## Chapter 5: Limitations of Vanilla LLMs

### 5.1 Strengths and Weaknesses

**Strengths:**
- Great at imitation and idea generation
- Amazing at generating and debugging code

**Weaknesses:**
- **Limited reasoning** ← Focus of Lecture 6
- Static knowledge
- Cannot perform actions ← Focus of Lectures 7 & 8
- Hard to evaluate

### 5.2 What is "Reasoning"?

**Tentative Definition**: Reasoning = Ability to solve a problem through a sequence of logical steps

| Type | Example |
|------|---------|
| NOT Reasoning | "What is the course code of Stanford's Transformers & LLMs class?" (fact recall) |
| Reasoning | "The bear was born in 2020. How old is this bear now?" (requires calculation) |

---

## Chapter 6: Chain of Thought and the New Paradigm

### 6.1 The Core Idea: Think Before Answering

**Chain of Thought (CoT) Prompting** (Wei et al., 2022): Teach the model to explain its reasoning before providing an answer.

**Without CoT:**
> Q: How old will the bear be next year?  A: 5.

**With CoT:**
> Q: How old will the bear be next year?  A: It will be one year older than its age this year, which was 4. Hence, it will be 5.

**Key Insight for Reasoning Models**: Do CoT but at a **MUCH larger scale**.

### 6.2 The New Paradigm

**Traditional**: Question → LLM → Answer

**Reasoning Model**: Question → LLM → [Reasoning Chain] → Answer

**Output = Reasoning + Answer**

The reasoning chain is explicit, often in tags like `<think>...</think>`.

### 6.3 Timeline of Reasoning Model Releases

| Date | Release |
|------|---------|
| Sep 2024 | OpenAI o1-preview |
| Dec 2024 | Google Gemini 2.0 Flash Thinking |
| Jan 2025 | DeepSeek R1 |
| Feb 2025 | Claude 3.7 Sonnet, Grok 3 Beta |
| Jun 2025 | Magistral |

### 6.4 How to Spot a Reasoning Model

- Shows "thinking" or "thought summary" in the interface
- Complete chain of thought usually hidden from users
- API pricing separates "reasoning tokens" from regular tokens

---

## Chapter 7: GRPO and Test-Time Scaling

### 7.1 Test-Time Scaling

**Core Idea**: Incentivize the model to reason before answering by allocating more compute at inference time.

**Key Considerations:**
- Reasoning chains are hard to write from scratch—SFT data by hand is impractical
- We don't want to limit the model to human-written reasoning
- Natural verifiable reward exists: "Did it solve the problem?" → yes/no

**Solution**: Use RL with verifiable rewards!

### 7.2 Verifiable Rewards for Reasoning

**Reward = Formatting + Accuracy**

| Component | Check |
|-----------|-------|
| Formatting | Does output contain proper `<think>...</think>` delimiters? |
| Accuracy - Code | Do all test cases pass? |
| Accuracy - Math | Does final answer match ground truth? |

No learned reward model needed!

### 7.3 GRPO: Group Relative Policy Optimization

GRPO (Shao et al., 2024, from DeepSeekMath) is the RL algorithm of choice for reasoning models.

**Key Formula:**

$$\text{Advantage} \approx \text{Reward} - \text{Average(reward of group)}$$

Unlike PPO which uses a learned value function, GRPO uses the **average reward of a GROUP** of sampled outputs for the same prompt.

### 7.4 GRPO vs PPO Comparison

**Similarities:**
- Both use probability ratios (new/old policy)
- Both use clipping to prevent large updates

**Differences:**

| Aspect | GRPO | PPO |
|--------|------|-----|
| KL penalty | Explicit term | Via clipping |
| Advantage baseline | Group average | Learned value function |
| Models needed | 3 (policy, reward, ref) | 4 (adds value model) |

### 7.5 Reasoning-Based Benchmarks

**Coding**: HumanEval, CodeForces, SWE-bench
- Task: Solve coding problem or fix bug
- Verification: Run test cases

**Math**: AIME, GSM8K
- Task: Solve challenging math problems
- Verification: Compare final answer to ground truth

### 7.6 Pass@k Metric

**Definition**: Probability that at least 1 of k attempts succeeds

$$\text{Pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

Where n = total samples, c = correct samples

**Variants:**
- **Pass@k**: When checking is easy or higher latency OK
- **Pass@1**: When single-generation quality matters
- **Cons@k**: Majority voting comparison

---

## Chapter 8: DeepSeek R1 Training Recipe

### 8.1 R1-Zero: Proof of Concept

R1-Zero demonstrates that reasoning abilities can emerge **purely from RL**, without any SFT on reasoning data.

**Training:**
1. Pretrain model → V3-Base (MoE: ~671B total, ~37B active)
2. GRPO with reasoning data → R1-Zero

**Template for R1-Zero:**
```
A conversation between User and Assistant. The assistant first thinks about 
the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think></think> and 
<answer></answer> tags respectively.
```

**Benefits**: Reasoning abilities without SFT!
**Challenges**: Chains have formatting/readability issues

### 8.2 The Full R1 Pipeline (5 Stages)

1. **Pretrain → V3-Base**: MoE architecture (~671B total, ~37B active)

2. **"Small-scale" SFT with reasoning data**
   - Data: Long CoTs from R1-Zero, rewritten by humans

3. **GRPO with reasoning data**
   - Reward = Formatting + Accuracy + Language Consistency

4. **"Large-scale" SFT with reasoning AND non-reasoning data**
   - ~600k pairs: Math, coding, logic (rejection sampling)
   - ~200k pairs: General data (reuses V3 SFT data)

5. **GRPO with reasoning AND non-reasoning data → R1**
   - Reasoning: Reward = Formatting + Accuracy
   - General: Reward = Helpfulness + Harmlessness

### 8.3 The Increasing Output Length Problem

**Observation**: During GRPO training, response length keeps increasing.

**Why it happens**: In GRPO, gradient contribution is proportional to token count. Short successful outputs get strong positive gradients; long outputs get diluted gradients. This creates incentive for longer outputs.

**Mitigations:**
- **DAPO**: Equalize token-level contributions
- **Dr. GRPO**: Alternative normalization scheme
- Difficulty-based adjustments
- Diversity encouragement

### 8.4 Distillation for Reasoning

**Traditional Distillation (Lecture 2):**
- Teacher produces soft probability distributions
- Student matches token-level distributions
- Goal: Match next token distribution

**Reasoning Distillation (R1-Distill):**
- R1 generates entire reasoning traces
- Smaller models learn via SFT on these traces
- Goal: Learn complete reasoning patterns

**Results**: Distilled models achieve competitive performance with efficient compute!

---

# Part III: Synthesis and Comparison

## Complete Training Pipeline Summary

**For Traditional Models:**
```
Base Model → Pretrain → SFT → Preference Tuning (RLHF or DPO) → Final Model
```

**For Reasoning Models (R1 Path):**
```
Base Model → Pretrain → Small SFT → GRPO (Reasoning) → Large SFT → GRPO (Full) → Final Model
```

## Method Comparison Table

| Method | Key Advantage | Key Challenge | Models Required |
|--------|---------------|---------------|-----------------|
| **RLHF (PPO)** | Proven track record; fine control | Complex setup; many hyperparameters | 4 |
| **DPO** | Simple supervised training; no RL | Fixed dataset; no exploration | 2 |
| **Best-of-N** | No training; inference-time only | N× inference cost | 2 |
| **GRPO** | No value model; simpler than PPO | Length bias; needs group sampling | 3 |

---

# Key Formulas Reference

## Bradley-Terry Model
$$p(y_i > y_j) = \sigma(r_i - r_j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}}$$

## Reward Model Loss
$$\mathcal{L}_{RM}(\theta) = -\mathbb{E}[\log \sigma(r(x, \hat{y}_w) - r(x, \hat{y}_l))]$$

## PPO-Clip Objective
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

## DPO Loss
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

## GRPO Advantage
$$\hat{A}_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j$$

## Pass@k
$$\text{Pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

---

# Key Takeaways

## From Lecture 5 (Preference Tuning):
- Preference data (comparing A vs B) is easier to collect than perfect demonstrations
- RLHF uses reward modeling + PPO to align models
- DPO simplifies preference tuning to supervised learning
- Both approaches have trade-offs; neither is universally superior

## From Lecture 6 (Reasoning Models):
- Test-time scaling allows models to "think longer" on hard problems
- GRPO is preferred over PPO for reasoning (no value model needed)
- Verifiable rewards (correct/incorrect) enable RL without human labeling
- Distillation can transfer reasoning abilities to smaller models

---

**— End of Study Guide —**

*References:*
- Ouyang et al., 2022 - "Training language models to follow instructions with human feedback"
- Schulman et al., 2017 - "Proximal Policy Optimization Algorithms"
- Rafailov et al., 2023 - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- Wei et al., 2022 - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Shao et al., 2024 - "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
- DeepSeek-AI, 2025 - "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
