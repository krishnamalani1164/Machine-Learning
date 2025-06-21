# Decision Forest Classification

## Theory
Decision Forest (Random Forest for Classification) is an ensemble method that combines multiple decision trees using bagging and random feature selection. It reduces overfitting and improves accuracy by aggregating predictions from diverse trees through majority voting.

## Key Formulas

### Ensemble Prediction
```
ŷ = mode(tree₁(x), tree₂(x), ..., treeₙ(x))
```

### Probability Estimation
```
P(class = c|x) = (1/n) Σ(i=1 to n) P_treeᵢ(class = c|x)
```

### Out-of-Bag (OOB) Accuracy
```
OOB_Accuracy = (1/n) Σ(I(yᵢ = ŷᵢ_oob))
```

### Feature Importance (Gini-based)
```
Importance(feature) = Σ(all trees) Σ(all nodes) p(node) × Gini_decrease
```
Where p(node) = proportion of samples reaching the node

### Bootstrap Sampling
```
Each tree trained on: Sample n observations with replacement
```

### Random Feature Selection
```
At each split: Randomly select √p features (default for classification)
Where p = total number of features
```

## Individual Tree Splitting

### Gini Impurity
```
Gini(S) = 1 - Σ(i=1 to c) pᵢ²
```

### Information Gain
```
Information_Gain = Gini(parent) - Σ(weighted_Gini(children))
```

### Entropy (Alternative)
```
Entropy(S) = -Σ(i=1 to c) pᵢ log₂(pᵢ)
```

## Algorithm Steps
1. **Bootstrap**: Create n random samples with replacement from training data
2. **Build Trees**: For each bootstrap sample:
   - Grow decision tree without pruning
   - At each node, randomly select √p features
   - Choose best split from selected features only
3. **Combine**: Store all n trained trees
4. **Predict**: 
   - Run input through all trees
   - Use majority vote for final classification
   - Calculate class probabilities as average

## Key Parameters
- **n_estimators**: Number of trees (more trees = better performance, slower)
- **max_features**: Features considered at each split
  - √p (default for classification)
  - log₂(p)
  - Custom number or fraction
- **max_depth**: Maximum tree depth (default: None - fully grown)
- **min_samples_split**: Minimum samples to split node (default: 2)
- **min_samples_leaf**: Minimum samples in leaf (default: 1)
- **bootstrap**: Use bootstrap sampling (default: True)
- **criterion**: Splitting criterion (gini, entropy)

## Voting Mechanisms

### Hard Voting (Majority Vote)
```
Predicted_class = mode(predictions from all trees)
```

### Soft Voting (Probability-based)
```
Predicted_class = argmax(average probabilities from all trees)
```

## Feature Importance Calculation
1. **Node Importance**: Weighted impurity decrease at each split
2. **Tree Importance**: Sum over all nodes in tree
3. **Forest Importance**: Average over all trees
4. **Normalization**: Scale to sum to 1.0

## Advantages
- Reduces overfitting compared to single decision trees
- Handles missing values naturally
- Provides feature importance rankings
- Works with mixed data types (numerical + categorical)
- Robust to outliers and noise
- Parallelizable training
- Good performance out-of-the-box

## Disadvantages
- Less interpretable than single decision tree
- Can overfit with very noisy data
- Biased toward categorical variables with more categories
- Memory intensive (stores multiple trees)
- Prediction slower than single tree

## Use Cases
- **Medical Diagnosis**: Multi-symptom classification
- **Image Classification**: Object recognition
- **Text Classification**: Document categorization, spam detection  
- **Fraud Detection**: Transaction classification
- **Customer Segmentation**: Marketing applications
- **Bioinformatics**: Gene expression classification
- **Any multi-class classification problem**
