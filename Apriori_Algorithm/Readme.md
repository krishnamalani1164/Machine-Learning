# Apriori Algorithm for Association Rule Learning

## Overview
Apriori is a classic algorithm for mining frequent itemsets and generating association rules in transactional data.

## Key Concepts

### Support
Frequency of an itemset in the dataset.
```
Support(A) = |transactions containing A| / |total transactions|
```

### Confidence
Strength of implication A → B.
```
Confidence(A → B) = Support(A ∪ B) / Support(A)
```

### Lift
Measures how much more likely B is given A.
```
Lift(A → B) = Support(A ∪ B) / (Support(A) × Support(B))
```

## Algorithm Steps
1. **Find frequent 1-itemsets** (items meeting minimum support)
2. **Generate candidates** for k-itemsets from frequent (k-1)-itemsets
3. **Prune candidates** using Apriori property
4. **Count support** for remaining candidates
5. **Repeat** until no more frequent itemsets found
6. **Generate rules** from frequent itemsets

## Apriori Property
If an itemset is infrequent, all its supersets are also infrequent.

## Example
```
Transaction DB: {A,B}, {A,C}, {B,C}, {A,B,C}
Min Support = 50% (2/4)

Frequent 1-itemsets: {A}:3, {B}:3, {C}:3
Frequent 2-itemsets: {A,B}:2, {A,C}:2, {B,C}:2
Frequent 3-itemsets: {A,B,C}:1 (pruned - below min support)

Rule: A → B
Support = 2/4 = 50%
Confidence = 2/3 = 67%
Lift = 0.5/(0.75×0.75) = 0.89
```

## Applications
- Market basket analysis
- Web usage patterns  
- Bioinformatics
- Recommendation systems
