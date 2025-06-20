# K-Nearest Neighbors (KNN)

## Theory
K-Nearest Neighbors is a lazy learning algorithm that makes predictions based on the k closest training examples in the feature space. It assumes that similar instances have similar outputs and uses local information for decision making.

## Key Formulas

### Distance Metrics

#### Euclidean Distance (Most Common)
```
d(x, y) = √(Σ(i=1 to n)(xᵢ - yᵢ)²)
```

#### Manhattan Distance
```
d(x, y) = Σ(i=1 to n)|xᵢ - yᵢ|
```

#### Minkowski Distance (Generalized)
```
d(x, y) = (Σ(i=1 to n)|xᵢ - yᵢ|^p)^(1/p)
```
Where p=1 (Manhattan), p=2 (Euclidean)

#### Cosine Distance
```
d(x, y) = 1 - (x·y)/(||x|| ||y||)
```

### Prediction Formulas

#### Classification (Majority Vote)
```
ŷ = mode(y₁, y₂, ..., yₖ)
```

#### Classification (Weighted Vote)
```
ŷ = argmax_c Σ(i=1 to k) wᵢ × I(yᵢ = c)
where wᵢ = 1/d(x, xᵢ)
```

#### Regression (Simple Average)
```
ŷ = (1/k) Σ(i=1 to k) yᵢ
```

#### Regression (Weighted Average)
```
ŷ = Σ(i=1 to k) wᵢyᵢ / Σ(i=1 to k) wᵢ
where wᵢ = 1/d(x, xᵢ)
```

## Algorithm Steps
1. **Store**: Keep all training data (lazy learning)
2. **Calculate**: Compute distances from query point to all training points
3. **Sort**: Order training points by distance
4. **Select**: Choose k nearest neighbors
5. **Predict**: 
   - Classification: Majority vote of k neighbors
   - Regression: Average of k neighbor values

## Key Parameters
- **k**: Number of neighbors (odd numbers avoid ties in classification)
- **distance_metric**: How to measure similarity (euclidean, manhattan, etc.)
- **weights**: Uniform (equal) or distance-based weighting
- **algorithm**: Method for finding neighbors (brute, ball_tree, kd_tree)

## Choosing k
- **Small k**: More sensitive to noise, complex decision boundaries
- **Large k**: Smoother decision boundaries, may lose local patterns
- **Rule of thumb**: k = √n (where n = number of samples)
- **Cross-validation**: Test different k values to find optimal

## Advantages
- Simple to understand and implement
- No assumptions about data distribution
- Works well with small datasets
- Naturally handles multi-class problems
- Can capture complex decision boundaries

## Disadvantages
- Computationally expensive at prediction time
- Sensitive to irrelevant features (curse of dimensionality)
- Requires feature scaling
- Memory intensive (stores all training data)
- Poor performance with high-dimensional sparse data

## Use Cases
- Recommendation systems (collaborative filtering)
- Pattern recognition and image classification
- Outlier detection
- Missing value imputation
- Text mining and document classification
