# Dimensionality Reduction Theory

A concise guide to the theoretical foundations of dimensionality reduction techniques.

## Overview

Dimensionality reduction transforms high-dimensional data **X** ∈ ℝⁿˣᵈ to lower-dimensional representation **Y** ∈ ℝⁿˣᵏ (where k < d) while preserving essential information.

**Goal**: Find mapping **f**: ℝᵈ → ℝᵏ that minimizes information loss.

## The Curse of Dimensionality

**Key Problems**:
- **Volume Concentration**: Data concentrates near surface of high-dimensional spaces
- **Distance Concentration**: All pairwise distances become similar
- **Sparsity**: Exponentially more data needed as dimensions increase

**Mathematical**: For d-dimensional unit sphere, volume ratio (1-ε)ᵈ → 0 as d increases.

## Linear Methods

### Principal Component Analysis (PCA)
**Objective**: Find directions of maximum variance
- **Method**: Eigendecomposition of covariance matrix **C = XᵀX**
- **Solution**: Principal components are eigenvectors with largest eigenvalues
- **Variance Explained**: λₖ/Σλᵢ for k-th component
- **Optimal**: Best linear reconstruction in least-squares sense

### Linear Discriminant Analysis (LDA)
**Objective**: Maximize class separation
- **Criterion**: **max (wᵀSᵦw)/(wᵀSᵨw)**
- **Sᵦ**: Between-class scatter matrix
- **Sᵨ**: Within-class scatter matrix
- **Solution**: Generalized eigenvalue problem

### Factor Analysis
**Model**: **x = Λf + ε**
- **Λ**: Factor loadings
- **f**: Latent factors ~ N(0,I)
- **ε**: Noise ~ N(0,Ψ)

### Independent Component Analysis (ICA)
**Objective**: Find statistically independent components
- **Model**: **x = As** (s = independent sources)
- **Method**: Maximize non-Gaussianity or minimize mutual information

## Non-Linear Methods

### Manifold Hypothesis
**Assumption**: High-dimensional data lies on lower-dimensional manifold **M** ⊆ ℝᵈ with intrinsic dimension k << d.

### t-SNE
**Objective**: Preserve local neighborhood structure
- **High-dim similarities**: **pᵢⱼ ∝ exp(-||xᵢ-xⱼ||²/2σ²)**
- **Low-dim similarities**: **qᵢⱼ ∝ (1+||yᵢ-yⱼ||²)⁻¹**
- **Loss**: KL divergence **KL(P||Q)**
- **Property**: Excellent for visualization, preserves clusters

### UMAP
**Foundation**: Riemannian geometry + algebraic topology
- **Method**: Fuzzy topological representation → cross-entropy optimization
- **Advantage**: Preserves both local and global structure
- **Speed**: Faster than t-SNE

### Locally Linear Embedding (LLE)
**Assumption**: Each point is linear combination of neighbors
1. **Weights**: **min Σᵢ ||xᵢ - Σⱼ Wᵢⱼxⱼ||²**
2. **Embedding**: **min Σᵢ ||yᵢ - Σⱼ Wᵢⱼyⱼ||²**

### Isomap
**Principle**: Preserve geodesic (manifold) distances
1. Build k-nearest neighbor graph
2. Compute shortest paths (geodesic approximation)
3. Apply classical multidimensional scaling

### Autoencoders
**Architecture**: Encoder **f**: ℝᵈ → ℝᵏ, Decoder **g**: ℝᵏ → ℝᵈ
- **Loss**: **||x - g(f(x))||²**
- **Advantage**: Can learn complex non-linear mappings
- **Variants**: Variational autoencoders (VAE), sparse autoencoders

## Information Theory Perspective

### Mutual Information
**Goal**: Maximize **I(X;Y) = H(X) - H(X|Y)** between original and reduced data

### Information Bottleneck
**Objective**: **min I(X;T) - βI(T;Y)**
- Balance compression and prediction accuracy
- **β** controls trade-off

### Rate-Distortion Theory
**Trade-off**: Compression rate vs reconstruction error
**R(D) = min I(Y;Ŷ)** subject to distortion ≤ D

## Key Theoretical Results

### Optimality Guarantees
- **PCA**: Optimal linear reconstruction (Eckart-Young theorem)
- **Johnson-Lindenstrauss**: Random projections preserve distances
- **Whitney Embedding**: m-dimensional manifold embeds in ℝ²ᵐ⁺¹

### Sample Complexity
Required samples scale with:
- Intrinsic dimensionality
- Manifold curvature
- Noise level
- Desired accuracy

### Convergence
- **Consistency**: Algorithms converge to true geometry as n → ∞
- **Stability**: Robustness to noise and perturbations

## Method Selection Guidelines

### Linear vs Non-Linear
- **Linear**: Data lies in linear subspace, need interpretability
- **Non-Linear**: Data has curved manifold structure

### Specific Choices
- **PCA**: Linear relationships, need fast computation
- **LDA**: Classification task, labeled data available  
- **t-SNE**: Visualization, cluster structure important
- **UMAP**: General purpose, balance of local/global structure
- **Autoencoders**: Complex non-linear relationships, large datasets

### Computational Complexity
- **PCA**: O(d³) for eigendecomposition
- **t-SNE**: O(n²) for pairwise distances
- **UMAP**: O(n^1.14) approximate
- **LLE/Isomap**: O(nk²) + O(n²) for eigendecomposition

## Evaluation Metrics

### Reconstruction Quality
- **Reconstruction Error**: **||X - X̂||²F**
- **Explained Variance**: Proportion of original variance retained

### Structure Preservation
- **Trustworthiness**: How well neighborhoods are preserved
- **Continuity**: Measure of local structure preservation
- **Silhouette Score**: Cluster quality in reduced space

### Information Theoretic
- **Mutual Information**: I(X;Y) between original and reduced data
- **Entropy**: Information content preservation

---

**Core Insight**: Dimensionality reduction exploits the fact that real-world high-dimensional data often has much lower intrinsic dimensionality due to correlations and underlying structure.
