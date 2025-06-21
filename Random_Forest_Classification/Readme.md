# Random Decision Forest Classification

## Theory
Random Decision Forest is an ensemble learning method that builds multiple decision trees using random sampling of both data (bootstrap) and features. It combines predictions through majority voting to create a robust classifier that reduces overfitting and improves generalization.

## Key Formulas

### Ensemble Classification
```
ŷ = majority_vote(tree₁(x), tree₂(x), ..., treeₙ(x))
ŷ = argmax_c Σ(i=1 to n) I(treeᵢ(x) = c)
```

### Class Probability Estimation
```
P(y = c|x) = (1/n) Σ(i=1 to n) I(treeᵢ(x) = c)
```

### Bootstrap Sampling Formula
```
Bootstrap sample size = n (same as original dataset)
Each sample drawn with replacement
```

### Random Feature Selection
```
Features per split = √p (classification default)
Where p = total number of features
```

### Out-of-Bag Error
```
OOB_Error = (1/m) Σ(i=1 to m) I(yᵢ ≠ ŷᵢ_oob)
Where m = number of OOB predictions
```

### Feature Importance (Mean Decrease Impurity)
```
Importance(Xⱼ) = (1/n) Σ(i=1 to n) Σ(t∈Tᵢ) p(t) × ΔI(t,Xⱼ)

Where:
- n = number of trees
- t = node in tree Tᵢ
- p(t) = proportion of samples reaching node t
- ΔI(t,Xⱼ) = impurity decrease when splitting on feature Xⱼ
```

## Tree Building Process

### Gini Impurity (Default Criterion)
```
Gini(S) = 1 - Σ(i=1 to k) (pᵢ)²
Where pᵢ = proportion of samples belonging to class i
```

### Information Gain
```
IG = Gini(parent) - Σ(weighted_Gini(children))
```

### Entropy (Alternative Criterion)
```
Entropy(S) = -Σ(i=1 to k) pᵢ log₂(pᵢ)
```

## Algorithm Steps
1. **Initialize**: Set number of trees (n_estimators)
2. **For each tree**:
   - Create bootstrap sample (sample n instances with replacement)
   - Build decision tree:
     - At each node, randomly select √p features
     - Find best split among selected features using Gini/Entropy
     - Split node and repeat recursively
     - No pruning (grow trees deep)
3. **Store Forest**: Keep all trained trees
4. **Prediction**:
   - Pass input through all trees
   - Collect all predictions
   - Return majority vote (hard voting)

## Randomness Sources

### Data Randomness (Bagging)
- Bootstrap sampling creates diverse training sets
- Each tree sees ~63.2% of original data
- Remaining ~36.8% used for OOB evaluation

### Feature Randomness
- Random subset of features at each split
- Reduces correlation between trees
- Default: √p features for classification

## Key Parameters
- **n_estimators**: Number of trees (100-1000 typical)
- **max_features**: Features per split ('sqrt', 'log2', int, float)
- **max_depth**: Maximum tree depth (None = unlimited)
- **min_samples_split**: Minimum samples to split (default: 2)
- **min_samples_leaf**: Minimum samples in leaf (default: 1)
- **bootstrap**: Use bootstrap sampling (default: True)
- **criterion**: Split criterion ('gini', 'entropy')
- **random_state**: Seed for reproducibility

## Voting Mechanisms

### Hard Voting (Classification)
```
Final_prediction = mode(all_tree_predictions)
```

### Soft Voting (Probability-based)
```
Final_prediction = argmax(average_class_probabilities)
```

## Performance Characteristics
- **Bias**: Lower than single decision tree
- **Variance**: Much lower than single decision tree
- **Overfitting**: Resistant due to averaging effect
- **Out-of-Bag**: Built-in validation without separate test set

## Advantages
- Excellent generalization performance
- Handles overfitting naturally
- Works with missing values
- Provides reliable feature importance
- Handles mixed data types
- Parallelizable training
- No need for extensive hyperparameter tuning

## Disadvantages
- Less interpretable than single tree
- Memory intensive (stores multiple trees)
- Slower prediction than single models
- Can still overfit with extremely noisy data
- Biased toward categorical features with many levels

## Use Cases
- **Healthcare**: Disease diagnosis, medical imaging
- **Finance**: Credit scoring, fraud detection
- **E-commerce**: Product recommendation, customer segmentation
- **Computer Vision**: Image classification, object detection
- **NLP**: Text classification, sentiment analysis
- **IoT**: Sensor data classification
- **Bioinformatics**: Gene expression analysis
