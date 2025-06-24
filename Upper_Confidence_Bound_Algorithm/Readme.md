# Upper Confidence Bound (UCB) Algorithm

## Overview
UCB is a multi-armed bandit algorithm that balances exploration and exploitation by selecting actions with highest upper confidence bounds.

## Key Formula

### UCB1 Selection Rule
```
UCB(i) = X̄ᵢ + √(2 ln(t) / nᵢ)
```

Where:
- `X̄ᵢ` = Average reward of arm i
- `t` = Total number of rounds played
- `nᵢ` = Number of times arm i has been selected
- `ln(t)` = Natural logarithm of total rounds

## Algorithm Components

### Exploitation Term
```
X̄ᵢ = Σ(rewards from arm i) / nᵢ
```
Favors arms with high observed rewards.

### Exploration Term  
```
√(2 ln(t) / nᵢ)
```
Increases uncertainty bound for less-explored arms.

## Algorithm Steps
1. **Initialize** - Play each arm once
2. **Calculate UCB** for each arm using formula
3. **Select arm** with highest UCB value
4. **Update** average reward and counts
5. **Repeat** until stopping criterion

## Confidence Bound Intuition
- **High average reward** → Higher UCB (exploitation)
- **Few selections** → Higher UCB (exploration)  
- **More total rounds** → Increased exploration pressure

## Example
```
Round t=10, Arms: A, B, C

Arm A: X̄=0.6, n=4 → UCB = 0.6 + √(2×ln(10)/4) = 1.28
Arm B: X̄=0.8, n=5 → UCB = 0.8 + √(2×ln(10)/5) = 1.48  
Arm C: X̄=0.4, n=1 → UCB = 0.4 + √(2×ln(10)/1) = 2.55

Select Arm C (highest UCB despite lowest average)
```

## UCB Variants

### UCB1-Tuned
```
UCB(i) = X̄ᵢ + √(ln(t)/nᵢ × min(1/4, Vᵢ + √(2ln(t)/nᵢ)))
```
Where `Vᵢ` is the sample variance of arm i.

### UCB2
Uses exponential scheduling for arm selection intervals.

## Theoretical Guarantees
- **Regret bound**: O(√(K ln(t)))
- **Optimal convergence** to best arm
- **Logarithmic regret** growth

## Properties
- **Parameter-free** - No tuning required
- **Anytime algorithm** - Can stop at any round
- **Theoretical backing** - Proven regret bounds
- **Simple implementation** - Easy to code

## Applications
- **Online advertising** - Ad selection optimization
- **A/B testing** - Treatment allocation  
- **Recommendation systems** - Content selection
- **Resource allocation** - Server/bandwidth assignment
- **Clinical trials** - Treatment arm selection
- **Game playing** - Move selection in MCTS

## vs Other Bandit Algorithms
| Algorithm | Exploration Strategy | Regret Bound |
|-----------|---------------------|--------------|
| UCB1 | Confidence intervals | O(√(K ln t)) |
| ε-greedy | Random exploration | O(K ln t) |
| Thompson Sampling | Bayesian sampling | O(√(K ln t)) |
