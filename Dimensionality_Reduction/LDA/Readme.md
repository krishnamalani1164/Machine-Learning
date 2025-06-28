# Linear Discriminant Analysis (LDA) Theory

## Overview

LDA is a supervised dimensionality reduction technique that finds linear combinations of features that best separate different classes. Unlike PCA which maximizes variance, LDA maximizes class separability.

## Mathematical Formulation

### Problem Setup
- **Data**: X ∈ ℝⁿˣᵈ with class labels y ∈ {1,2,...,C}
- **Goal**: Find projection directions that maximize between-class separation while minimizing within-class scatter
- **Output**: Transformed data in (C-1)-dimensional space (maximum)

### Core Optimization Problem

**Objective**: Find direction w that maximizes Fisher's criterion:

```
J(w) = (wᵀSᵦw) / (wᵀSᵨw)
```

Where:
- **Sᵦ**: Between-class scatter matrix
- **Sᵨ**: Within-class scatter matrix

**Solution**: w is the eigenvector of Sᵨ⁻¹Sᵦ with largest eigenvalue.

## Scatter Matrices

### Within-Class Scatter Matrix
```
Sᵨ = Σᵢ₌₁ᶜ Σⱼ∈Cᵢ (xⱼ - μᵢ)(xⱼ - μᵢ)ᵀ
```
- **μᵢ**: Mean of class i
- **Cᵢ**: Set of samples in class i
- **Measures**: Compactness within each class

### Between-Class Scatter Matrix
```
Sᵦ = Σᵢ₌₁ᶜ nᵢ(μᵢ - μ)(μᵢ - μ)ᵀ
```
- **nᵢ**: Number of samples in class i
- **μ**: Overall mean of all data
- **Measures**: Separation between class means

### Total Scatter Matrix
```
Sₜ = Sᵨ + Sᵦ = Σⱼ₌₁ⁿ (xⱼ - μ)(xⱼ - μ)ᵀ
```

## Generalized Eigenvalue Problem

### Mathematical Formulation
LDA reduces to solving:
```
Sᵦw = λSᵨw
```

**Equivalent form** (when Sᵨ is invertible):
```
Sᵨ⁻¹Sᵦw = λw
```

### Multiple Discriminants
For multi-class problems:
- **Maximum discriminants**: min(C-1, d) where C = number of classes
- **Discriminant functions**: w₁, w₂, ..., wₖ (ordered by eigenvalue magnitude)
- **Projection matrix**: W = [w₁, w₂, ..., wₖ]

## Geometric Interpretation

### Fisher's Linear Discriminant
- **Goal**: Find line (in 2D) that best separates classes
- **Criterion**: Maximize ratio of between-class variance to within-class variance
- **Intuition**: Classes should be far apart with small within-class spread

### Projection View
- Project data onto lower-dimensional subspace
- **Subspace**: Spanned by discriminant vectors
- **Optimization**: Best class separation in projected space

## Theoretical Properties

### Optimality
- **Fisher's Criterion**: LDA maximizes Fisher's discriminant ratio
- **Bayes Error**: Under certain assumptions, LDA minimizes Bayes classification error
- **Gaussian Classes**: Optimal for equal covariance Gaussian classes

### Dimensionality Reduction
- **Maximum dimensions**: At most (C-1) meaningful discriminants
- **Information**: Preserves class-discriminative information
- **Loss**: May lose within-class structure information

## Assumptions

### Key Assumptions
1. **Gaussian Distribution**: Classes follow multivariate Gaussian distributions
2. **Equal Covariance**: All classes have same covariance matrix (homoscedasticity)
3. **Linear Separability**: Classes can be separated by linear boundaries
4. **No Outliers**: Method sensitive to outliers

### Violation Consequences
- **Unequal covariances**: Quadratic discriminant analysis (QDA) more appropriate
- **Non-Gaussian**: Robust variants or non-parametric methods needed
- **Non-linear separation**: Kernel LDA or non-linear methods required

## Relationship to Other Methods

### LDA vs PCA
| Aspect | LDA | PCA |
|--------|-----|-----|
| **Supervision** | Supervised (uses labels) | Unsupervised |
| **Objective** | Maximize class separation | Maximize variance |
| **Dimensions** | ≤ (C-1) | ≤ min(n,d) |
| **Use Case** | Classification | General dimensionality reduction |

### LDA as Classification
LDA can be used directly for classification:
- **Decision Rule**: Assign to class with highest discriminant score
- **Linear Boundaries**: Creates linear decision boundaries
- **Probabilistic**: Can provide class probabilities under Gaussian assumption

## Computational Aspects

### Standard Algorithm
1. **Compute class means**: μᵢ for each class i
2. **Compute scatter matrices**: Sᵨ and Sᵦ
3. **Solve eigenvalue problem**: Sᵨ⁻¹Sᵦw = λw
4. **Select discriminants**: Top (C-1) eigenvectors
5. **Project data**: Y = XW

### Complexity
- **Time**: O(d³) for matrix inversion + O(Cd²) for scatter computation
- **Space**: O(d²) for scatter matrices
- **Constraint**: Requires Sᵨ to be invertible

### Numerical Issues
- **Singular Sᵨ**: When d > n or classes perfectly separable
- **Solutions**: 
  - Regularization: Sᵨ + εI
  - Pseudo-inverse
  - PCA preprocessing

## Extensions and Variants

### Regularized LDA
- **Problem**: Singular or ill-conditioned Sᵨ
- **Solution**: Sᵨ + λI (ridge regularization)
- **Benefit**: Improves numerical stability

### Quadratic Discriminant Analysis (QDA)
- **Assumption**: Different covariance matrices per class
- **Decision boundary**: Quadratic instead of linear
- **Trade-off**: More flexible but requires more parameters

### Kernel LDA
- **Idea**: Apply LDA in high-dimensional feature space
- **Kernel trick**: K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
- **Benefit**: Can handle non-linear class boundaries

### Flexible Discriminant Analysis
- **Extension**: Non-parametric regression instead of linear
- **Benefit**: More flexible class boundaries
- **Methods**: Splines, neural networks, etc.

## Performance Characteristics

### Strengths
- **Supervised**: Uses class information effectively
- **Interpretable**: Linear combinations have clear meaning
- **Efficient**: Fast computation for well-conditioned problems
- **Optimal**: Under Gaussian assumptions with equal covariances

### Limitations
- **Strong assumptions**: Gaussian, equal covariance requirements
- **Limited dimensions**: At most (C-1) discriminants
- **Outlier sensitive**: Can be heavily influenced by outliers
- **Small sample**: Problems when n < d (high-dimensional data)

## Evaluation Metrics

### Classification Performance
- **Accuracy**: Overall classification rate
- **Class-specific**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed error analysis

### Dimensionality Reduction Quality
- **Discriminant Ratios**: λᵢ values indicate separation quality
- **Between/Within Ratio**: Overall separation measure
- **Visualization**: 2D/3D plots of discriminant space

## Key Insight

LDA finds the optimal linear projection for classification by explicitly maximizing the ratio of between-class to within-class variance. This makes it particularly effective when the goal is both dimensionality reduction and classification, especially under the assumption of Gaussian classes with equal covariances.
