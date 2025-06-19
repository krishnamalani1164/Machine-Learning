# Decision Tree

## Theory
Decision trees are tree-like models that make decisions by splitting data based on feature values. They create a hierarchy of if-else conditions to classify or predict outcomes, mimicking human decision-making processes.

## Key Formulas

### Entropy (Classification)
```
Entropy(S) = -Σ(i=1 to c) pᵢ log₂(pᵢ)
```
Where pᵢ = proportion of samples belonging to class i

### Information Gain
```
Information Gain = Entropy(parent) - Σ(weighted_entropy(children))
```

### Gini Impurity (Alternative to Entropy)
```
Gini(S) = 1 - Σ(i=1 to c) pᵢ²
```

### Mean Squared Error (Regression)
```
MSE = (1/n) Σ(yᵢ - ȳ)²
```
Where ȳ = mean of target values in node

### Variance Reduction (Regression)
```
Variance Reduction = Variance(parent) - Σ(weighted_variance(children))
```

## Algorithm Steps
1. **Start**: Begin with root node containing all data
2. **Split**: Find best feature and threshold that maximizes information gain
3. **Partition**: Split data into subsets based on chosen feature
4. **Recurse**: Repeat process for each child node
5. **Stop**: When stopping criteria met (max depth, min samples, etc.)
6. **Predict**: Assign majority class (classification) or mean value (regression)

## Splitting Criteria
- **Classification**: Information Gain, Gini Impurity
- **Regression**: MSE reduction, Variance reduction

## Key Parameters
- **max_depth**: Maximum tree depth (prevents overfitting)
- **min_samples_split**: Minimum samples required to split
- **min_samples_leaf**: Minimum samples in leaf node
- **max_features**: Maximum features considered for splitting
- **criterion**: Splitting criterion (gini, entropy, mse)

## Advantages
- Easy to understand and interpret
- Handles both numerical and categorical data
- No assumptions about data distribution
- Automatic feature selection
- Handles missing values well

## Disadvantages
- Prone to overfitting
- Unstable (small data changes → different trees)
- Biased toward features with more levels
- Struggles with linear relationships

## Use Cases
- Medical diagnosis (symptom-based decisions)
- Credit approval (risk assessment)
- Customer segmentation
- Feature selection and data exploration
- Rule extraction from data
