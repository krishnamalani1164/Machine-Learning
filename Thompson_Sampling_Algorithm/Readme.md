# Thompson Sampling Algorithm

## Overview
Thompson Sampling is a Bayesian multi-armed bandit algorithm that uses probability matching for exploration-exploitation by sampling from posterior distributions.

## Key Concepts

### Beta-Binomial Model
For binary rewards (0/1), use Beta distribution as conjugate prior:
```
Prior: Beta(α, β)
Posterior: Beta(α + successes, β + failures)
```

### Thompson Sampling Rule
```
θᵢ ~ Beta(αᵢ, βᵢ)
Select arm: argmax(θᵢ)
```

Where `θᵢ` is sampled reward probability for arm i.

## Algorithm Steps
1. **Initialize priors** - Set Beta(1,1) for each arm
2. **Sample from posteriors** - Draw θᵢ ~ Beta(αᵢ, βᵢ) for each arm
3. **Select arm** with highest sampled value
4. **Observe reward** (0 or 1)
5. **Update posterior** - Increment α (success) or β (failure)
6. **Repeat** until stopping criterion

## Beta Distribution Properties
```
Mean: α / (α + β)
Variance: αβ / ((α + β)²(α + β + 1))
```

### Prior Update Rules
```
Success (reward = 1): α ← α + 1
Failure (reward = 0): β ← β + 1
```

## Example
```
Round 5: Three arms A, B, C

Arm A: α=3, β=2 → Beta(3,2)
Arm B: α=1, β=4 → Beta(1,4)  
Arm C: α=2, β=1 → Beta(2,1)

Sample:
θA ~ Beta(3,2) = 0.75
θB ~ Beta(1,4) = 0.18
θC ~ Beta(2,1) = 0.82

Select Arm C (highest sample)
If reward=1: C becomes Beta(3,1)
If reward=0: C becomes Beta(2,2)
```

## Gaussian Thompson Sampling
For continuous rewards with normal distribution:
```
Prior: N(μ₀, σ₀²)
Likelihood: N(μ, σ²)
Posterior: N(μₙ, σₙ²)

μₙ = (σ²μ₀ + nσ₀²x̄) / (σ² + nσ₀²)
σₙ² = σ²σ₀² / (σ² + nσ₀²)
```

## Multi-Parameter Extensions

### Contextual Thompson Sampling
```
θ ~ N(μ, Σ)
μₜ₊₁ = (Σ₀⁻¹μ₀ + Σᵗ⁻¹x̄ₜ) / (Σ₀⁻¹ + Σᵗ⁻¹)
```

### Linear Thompson Sampling
For linear payoff models with feature vectors.

## Theoretical Properties
- **Bayesian regret bound**: O(√(K ln T))
- **Asymptotically optimal** - Matches lower bounds
- **Self-tuning** - Automatically balances exploration/exploitation
- **Information-theoretic optimality**

## Advantages
- **Natural exploration** - Probabilistic arm selection
- **No hyperparameters** - Uses problem structure
- **Flexible priors** - Incorporates domain knowledge
- **Computationally efficient** - Simple sampling operations
- **Robust performance** - Works well across problem types

## Implementation Variants

### Basic Thompson Sampling
```python
# Pseudo-code
for t in range(T):
    theta = [beta.rvs(alpha[i], beta[i]) for i in arms]
    arm = argmax(theta)
    reward = pull_arm(arm)
    if reward == 1:
        alpha[arm] += 1
    else:
        beta[arm] += 1
```

### Optimistic Thompson Sampling
Uses inflated variance for more exploration.

## Applications
- **Online advertising** - Ad placement optimization
- **Clinical trials** - Adaptive treatment allocation
- **Recommendation systems** - Content personalization  
- **A/B testing** - Dynamic traffic allocation
- **Portfolio optimization** - Asset allocation
- **Hyperparameter tuning** - ML model optimization
- **Game playing** - Strategy selection
- **Dynamic pricing** - Price point selection

## vs Other Bandit Algorithms
| Algorithm | Strategy | Regret | Computation |
|-----------|----------|--------|-------------|
| Thompson Sampling | Bayesian sampling | O(√(K ln T)) | O(K) |
| UCB1 | Confidence bounds | O(√(K ln T)) | O(K) |
| ε-greedy | Random exploration | O(K ln T) | O(K) |
| LinUCB | Linear confidence | O(√(T ln T)) | O(d²) |

## Best Practices
- **Choose appropriate priors** based on domain knowledge
- **Monitor convergence** of posterior distributions
- **Handle non-stationary** environments with discounting
- **Scale features** for contextual variants
