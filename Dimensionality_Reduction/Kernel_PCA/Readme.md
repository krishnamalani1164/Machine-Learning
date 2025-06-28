# Kernel PCA Theory

## Overview

Kernel PCA extends standard PCA to capture non-linear relationships by implicitly mapping data to a higher-dimensional feature space using kernel functions, then performing PCA in that space without explicitly computing the mapping.

## Mathematical Formulation

### Problem Setup
- **Data**: X ∈ ℝⁿˣᵈ
- **Feature Map**: φ: ℝᵈ → ℝᴴ (possibly infinite-dimensional)
- **Goal**: Perform PCA on φ(X) without explicitly computing φ
- **Key Insight**: Use kernel trick to avoid high-dimensional computations

### Kernel Trick
Instead of computing φ(x) explicitly, use kernel function:
```
K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
```

**Kernel Matrix**: K ∈ ℝⁿˣⁿ where Kᵢⱼ = K(xᵢ, xⱼ)

## Theoretical Foundation

### Standard PCA in Feature Space
In feature space, we want to find eigenvectors of covariance matrix:
```
C = (1/n) Σᵢ₌₁ⁿ φ(xᵢ)φ(xᵢ)ᵀ
```

**Eigenvalue equation**: Cv = λv

### Key Insight: Eigenvector Representation
Any eigenvector v can be written as:
```
v = Σᵢ₌₁ⁿ αᵢφ(xᵢ) = Φᵀα
```

Where Φ = [φ(x₁), φ(x₂), ..., φ(xₙ)]ᵀ and α = [α₁, α₂, ..., αₙ]ᵀ

### Dual Formulation
Substituting into eigenvalue equation:
```
(1/n)ΦΦᵀ(Φᵀα) = λ(Φᵀα)
```

Multiplying by Φ:
```
(1/n)ΦΦᵀΦΦᵀα = λΦΦᵀα
```

Since ΦΦᵀ = K (kernel matrix):
```
(1/n)K²α = λKα
```

**Simplified**: Kα = nλα

## Algorithm

### Centering in Feature Space
Must center data in feature space: φ̃(x) = φ(x) - μφ

**Centered kernel matrix**:
```
K̃ = K - 1ₙK - K1ₙ + 1ₙK1ₙ
```

Where 1ₙ is n×n matrix of ones divided by n.

### Standard Kernel PCA Steps
1. **Compute kernel matrix**: K with Kᵢⱼ = K(xᵢ, xⱼ)
2. **Center kernel matrix**: K̃ (centering in feature space)
3. **Eigendecomposition**: K̃α = λα
4. **Normalize eigenvectors**: αᵀK̃α = nλ
5. **Project new data**: For new point x, projection is Σᵢ αᵢK(xᵢ, x)

## Common Kernel Functions

### Polynomial Kernel
```
K(x, y) = (xᵀy + c)ᵈ
```
- **c**: Constant term
- **d**: Polynomial degree
- **Feature space**: All monomials up to degree d

### Gaussian (RBF) Kernel
```
K(x, y) = exp(-γ||x - y||²)
```
- **γ**: Bandwidth parameter (γ = 1/(2σ²))
- **Feature space**: Infinite-dimensional
- **Property**: Universal approximator

### Sigmoid Kernel
```
K(x, y) = tanh(αxᵀy + β)
```
- **α, β**: Parameters
- **Relation**: Similar to neural network activation

### Linear Kernel
```
K(x, y) = xᵀy
```
- **Special case**: Reduces to standard PCA
- **Feature space**: Original space

## Theoretical Properties

### Mercer's Theorem
**Condition**: Kernel function must be positive semi-definite
```
∫∫ K(x, y)f(x)f(y)dxdy ≥ 0  for all f ∈ L²
```

**Implication**: Guarantees existence of feature map φ

### Reproducing Kernel Hilbert Space (RKHS)
- **Feature space**: RKHS associated with kernel
- **Reproducing property**: K(x, ·) represents evaluation functional
- **Norm**: ||f||²ₕ has specific structure in RKHS

### Eigenvalue Properties
- **Non-negative**: All eigenvalues λᵢ ≥ 0
- **Finite non-zero**: At most n non-zero eigenvalues
- **Ordering**: λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0

## Geometric Interpretation

### Non-linear Principal Components
- **Standard PCA**: Linear combinations of original features
- **Kernel PCA**: Non-linear combinations through kernel mapping
- **Curves**: Principal components can be non-linear curves/surfaces

### Feature Space Perspective
- **High-dimensional**: Data mapped to (possibly infinite) dimensional space
- **Linear PCA**: Performed in high-dimensional feature space
- **Projection**: Back to input space gives non-linear components

## Advantages and Limitations

### Advantages
- **Non-linear**: Captures non-linear relationships
- **Kernel trick**: Avoids explicit high-dimensional computation
- **Flexibility**: Various kernels for different data types
- **No assumptions**: Fewer distributional assumptions than linear PCA

### Limitations
- **Computational cost**: O(n³) for eigendecomposition of n×n matrix
- **Memory**: Requires storing n×n kernel matrix
- **Parameter selection**: Kernel parameters need tuning
- **Interpretability**: Less interpretable than linear PCA
- **Out-of-sample**: Requires storing training data for new projections

## Parameter Selection

### Kernel Parameters
- **RBF γ**: Cross-validation or heuristics (median distance)
- **Polynomial d**: Usually low values (2, 3, 4)
- **Grid search**: Systematic parameter exploration

### Number of Components
- **Eigenvalue decay**: Look for significant drop in eigenvalues
- **Cross-validation**: Use downstream task performance
- **Visualization**: For 2D/3D plots, use first 2-3 components

## Computational Aspects

### Complexity
- **Training**: O(n³) for eigendecomposition
- **Memory**: O(n²) for kernel matrix storage
- **Prediction**: O(n) for each new point projection

### Approximation Methods
- **Nyström approximation**: Use subset of data points
- **Random features**: Approximate kernel with random projections
- **Incomplete Cholesky**: Low-rank approximation of kernel matrix

### Centering Computation
```python
# Centering kernel matrix
K_centered = K - np.mean(K, axis=0) - np.mean(K, axis=1)[:, None] + np.mean(K)
```

## Applications

### Dimensionality Reduction
- **Non-linear manifolds**: Data on curved surfaces
- **Complex relationships**: When linear PCA insufficient
- **Preprocessing**: For non-linear classification/regression

### Visualization
- **2D/3D embedding**: Non-linear dimensionality reduction for plotting
- **Cluster visualization**: Reveal non-linear cluster structure
- **Manifold learning**: Discover underlying non-linear structure

### Feature Extraction
- **Image processing**: Capture non-linear image features
- **Signal processing**: Non-linear signal components
- **Bioinformatics**: Gene expression, protein analysis

## Relationship to Other Methods

### Kernel PCA vs Standard PCA
| Aspect | Standard PCA | Kernel PCA |
|--------|--------------|------------|
| **Linearity** | Linear | Non-linear |
| **Complexity** | O(d³) | O(n³) |
| **Memory** | O(d²) | O(n²) |
| **Interpretability** | High | Low |
| **Flexibility** | Limited | High |

### Connection to Other Kernel Methods
- **Kernel Ridge Regression**: Same kernel trick principle
- **SVM**: Related feature space mapping
- **Gaussian Processes**: RKHS framework connection

## Practical Considerations

### When to Use Kernel PCA
- **Non-linear data**: Clear non-linear structure in data
- **Sufficient samples**: n not too large (due to O(n³) complexity)
- **Exploration**: Discovering unknown non-linear patterns
- **Preprocessing**: For subsequent non-linear analysis

### Alternatives
- **Manifold learning**: t-SNE, UMAP for visualization
- **Autoencoders**: Neural network-based non-linear PCA
- **Isomap, LLE**: Other non-linear dimensionality reduction

## Key Insight

Kernel PCA elegantly extends PCA to non-linear settings by using the kernel trick to implicitly work in high-dimensional feature spaces. This allows discovery of non-linear principal components without the computational burden of explicitly computing high-dimensional mappings, making it a powerful tool for non-linear dimensionality reduction and feature extraction.
