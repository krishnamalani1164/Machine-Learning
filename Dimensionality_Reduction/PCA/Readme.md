# Principal Component Analysis (PCA) Theory

## Overview

PCA finds orthogonal directions of maximum variance in data to reduce dimensionality while preserving as much information as possible.

## Mathematical Formulation

### Problem Setup
- **Data**: X ∈ ℝⁿˣᵈ (n samples, d features)
- **Goal**: Find k orthogonal directions (k < d) capturing maximum variance
- **Output**: Transformed data Y ∈ ℝⁿˣᵏ

### Core Optimization Problem

**Objective**: Find direction w₁ that maximizes variance:

```
max Var(Xw₁) = max w₁ᵀCw₁
subject to ||w₁|| = 1
```

Where C = (1/n)XᵀX is the covariance matrix.

**Solution**: w₁ is the eigenvector of C with largest eigenvalue λ₁.

### Sequential Approach

For subsequent components:
- w₂ maximizes variance subject to being orthogonal to w₁
- w₃ maximizes variance subject to being orthogonal to w₁, w₂
- And so on...

**Result**: Principal components are eigenvectors of covariance matrix ordered by eigenvalue magnitude.

## Key Mathematical Properties

### Eigenvalue Decomposition
```
C = QΛQᵀ
```
- **Q**: Matrix of eigenvectors (principal components)
- **Λ**: Diagonal matrix of eigenvalues
- **Eigenvalues**: λ₁ ≥ λ₂ ≥ ... ≥ λₑ ≥ 0

### Variance Explained
- **Total variance**: tr(C) = Σλᵢ
- **Variance by k-th component**: λₖ/Σλᵢ  
- **Cumulative variance**: Σᵢ₌₁ᵏ λᵢ/Σᵢ₌₁ᵈ λᵢ

### Reconstruction
- **Forward transformation**: Y = XW (W = first k eigenvectors)
- **Reconstruction**: X̂ = YWᵀ
- **Reconstruction error**: ||X - X̂||²F = Σᵢ₌ₖ₊₁ᵈ λᵢ

## Theoretical Guarantees

### Optimality (Eckart-Young Theorem)
PCA provides the **optimal rank-k approximation** of X in terms of:
- Frobenius norm: ||X - X̂||²F
- Spectral norm: ||X - X̂||₂

### Minimization Perspective
PCA can be viewed as solving:
```
min ||X - XWWᵀ||²F
subject to WᵀW = I
```

### Maximum Likelihood (Probabilistic PCA)
Under Gaussian assumptions:
- **Model**: x = Wz + μ + ε
- **z ~ N(0,I)**: Latent variables  
- **ε ~ N(0,σ²I)**: Noise
- **ML solution**: W spans same subspace as first k eigenvectors

## Geometric Interpretation

### Projection View
- PCA projects data onto k-dimensional subspace
- **Subspace**: Span of first k eigenvectors
- **Projection**: Orthogonal projection minimizing reconstruction error

### Variance Maximization
- First PC: Direction of maximum variance
- Second PC: Direction of maximum remaining variance (orthogonal to first)
- Subsequent PCs: Iteratively maximize remaining variance

## Assumptions and Limitations

### Key Assumptions
1. **Linearity**: Relationships between variables are linear
2. **Gaussian distribution**: Optimal under multivariate Gaussian assumption
3. **Variance = Importance**: High variance directions are most important

### Limitations
- **Linear method**: Cannot capture non-linear relationships
- **Global method**: Single transformation for entire dataset
- **Sensitive to scaling**: Requires standardization for different units
- **Interpretability**: Components are linear combinations, not original features

## Computational Aspects

### Standard Algorithm
1. **Center data**: X̃ = X - μ (μ = sample mean)
2. **Compute covariance**: C = (1/n)X̃ᵀX̃
3. **Eigendecomposition**: C = QΛQᵀ
4. **Select components**: W = first k columns of Q
5. **Transform**: Y = X̃W

### Complexity
- **Time**: O(d³) for eigendecomposition + O(nd²) for covariance
- **Space**: O(d²) for covariance matrix

### Numerical Considerations
- **SVD approach**: More numerically stable than eigendecomposition
- **X = UΣVᵀ**: Principal components are columns of V
- **Eigenvalues**: λᵢ = σᵢ²/(n-1)

## Alternative Formulations

### Singular Value Decomposition
```
X̃ = UΣVᵀ
```
- **V**: Principal component directions
- **U**: Principal component scores (up to scaling)
- **Σ**: Related to square root of eigenvalues

### Dual PCA
When n < d, compute eigenvectors of X̃X̃ᵀ instead of X̃ᵀX̃:
- **Complexity**: O(n³) vs O(d³)
- **Useful**: When samples << features

## Extensions

### Kernel PCA
- **Idea**: Apply PCA in feature space via kernel trick
- **Kernel**: K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)
- **Benefit**: Captures non-linear relationships

### Sparse PCA
- **Goal**: Find components with few non-zero loadings
- **Trade-off**: Sparsity vs explained variance
- **Benefit**: Improved interpretability

### Robust PCA
- **Problem**: Standard PCA sensitive to outliers
- **Solution**: L₁ norm or other robust measures
- **Decomposition**: X = L + S (low-rank + sparse)

## Key Insight

PCA exploits the fact that high-dimensional data often lies near a lower-dimensional linear subspace. It finds the "best" such subspace by maximizing preserved variance, which under Gaussian assumptions is equivalent to minimizing information loss.
