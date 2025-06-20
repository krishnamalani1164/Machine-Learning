# Linear Support Vector Machine (SVM)

## Theory
Linear SVM finds the optimal hyperplane that separates classes with maximum margin. It seeks the decision boundary that maximizes the distance between the closest points of different classes (support vectors).

## Key Formulas

### Decision Function
```
f(x) = wᵀx + b
```

### Classification Rule
```
ŷ = sign(wᵀx + b) = {
  +1  if wᵀx + b ≥ 0
  -1  if wᵀx + b < 0
}
```

### Margin
```
Margin = 2/||w||
```

### Distance from Point to Hyperplane
```
distance = |wᵀx + b|/||w||
```

### Optimization Problem (Hard Margin)
```
Minimize: (1/2)||w||²

Subject to: yᵢ(wᵀxᵢ + b) ≥ 1, ∀i
```

### Optimization Problem (Soft Margin)
```
Minimize: (1/2)||w||² + C Σξᵢ

Subject to: 
yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
ξᵢ ≥ 0, ∀i
```

### Dual Problem
```
Maximize: Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼxᵢᵀxⱼ

Subject to:
0 ≤ αᵢ ≤ C
Σαᵢyᵢ = 0
```

### Support Vector Condition
```
Support vectors: αᵢ > 0
Margin vectors: 0 < αᵢ < C
```

## Key Concepts

### Hyperplane
Linear decision boundary: wᵀx + b = 0

### Support Vectors
Training points that lie on the margin boundary. Only these points determine the hyperplane.

### Margin
Width of the "street" between classes. SVM maximizes this margin.

### Slack Variables (ξᵢ)
Allow misclassification in soft margin SVM. Measure how much a point violates the margin.

## Key Parameters
- **C**: Regularization parameter
  - High C: Hard margin, less tolerance for misclassification
  - Low C: Soft margin, more tolerance for misclassification
- **class_weight**: Handle imbalanced classes
- **max_iter**: Maximum iterations for optimization

## Algorithm Steps
1. **Formulate**: Set up quadratic optimization problem
2. **Solve**: Find optimal αᵢ values using SMO algorithm
3. **Extract**: Calculate w and b from support vectors
4. **Predict**: Use f(x) = wᵀx + b for classification

## Advantages
- Effective for high-dimensional data
- Memory efficient (uses only support vectors)
- Works well when classes are clearly separated
- Mathematically robust optimization

## Disadvantages
- Sensitive to feature scaling
- No probabilistic output
- Performance degrades with noise and overlapping classes
- Choice of C parameter is crucial

## Use Cases
- Text classification (spam detection, sentiment analysis)
- Image classification (linearly separable features)
- Bioinformatics (gene classification)
- Document classification
- Any binary classification with clear linear separation
