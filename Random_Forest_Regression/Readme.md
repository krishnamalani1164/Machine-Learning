# Random Forest

## Theory
Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses bagging (bootstrap aggregating) and random feature selection to reduce overfitting and improve generalization.

## Key Concepts

### Ensemble Prediction
```
Classification: ŷ = mode(tree₁(x), tree₂(x), ..., treeₙ(x))
Regression: ŷ = (1/n) Σ(i=1 to n) treeᵢ(x)
```

### Bootstrap Sampling
```
For each tree: Sample m observations with replacement from original dataset
```

### Feature Randomness
```
At each split: Randomly select √p features (classification) or p/3 features (regression)
Where p = total number of features
```

### Out-of-Bag (OOB) Error
```
OOB Error = Average error on samples not used in bootstrap for each tree
```

## Algorithm Steps
1. **Bootstrap**: Create n random samples with replacement
2. **Build Trees**: Train decision tree on each bootstrap sample
3. **Random Features**: At each node, consider only random subset of features
4. **No Pruning**: Grow trees deep (low bias, high variance)
5. **Aggregate**: Combine predictions via voting (classification) or averaging (regression)

## Key Parameters
- **n_estimators**: Number of trees (more trees = better performance, slower)
- **max_features**: Features considered at each split (√p for classification)
- **max_depth**: Maximum tree depth (default: None)
- **min_samples_split**: Minimum samples to split node (default: 2)
- **bootstrap**: Whether to use bootstrap sampling (default: True)

## Advantages
- Reduces overfitting compared to single decision trees
- Handles missing values and mixed data types
- Provides feature importance scores
- Works well out-of-the-box with minimal tuning
- Robust to outliers and noise

## Feature Importance
```
Importance = Average decrease in impurity across all trees for each feature
```

## Use Cases
- Classification: Image recognition, spam detection, medical diagnosis
- Regression: House prices, stock prediction, demand forecasting
- Feature selection and ranking
- Any tabular data problem
