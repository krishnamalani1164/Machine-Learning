# Kernel Support Vector Machine (SVM)

## Theory
Kernel SVM extends linear SVM to handle non-linearly separable data by mapping input features to a higher-dimensional space where linear separation becomes possible. The kernel trick allows this transformation without explicitly computing the high-dimensional coordinates.

## Key Formulas

### Decision Function
```
f(x) = Σ(i=1 to n) αᵢyᵢK(xᵢ, x) + b
```

### Classification Rule
```
ŷ = sign(f(x)) = sign(Σ(i=1 to n) αᵢyᵢK(xᵢ, x) + b)
```

### Dual Optimization Problem
```
Maximize: Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ, xⱼ)

Subject to:
0 ≤ αᵢ ≤ C
Σαᵢyᵢ = 0
```

## Common Kernel Functions

### Linear Kernel
```
K(xᵢ, xⱼ) = xᵢᵀxⱼ
```

### Polynomial Kernel
```
K(xᵢ, xⱼ) = (γxᵢᵀxⱼ + r)^d

Parameters:
- d: degree (default: 3)
- γ: gamma coefficient
- r: independent term (default: 0)
```

### Radial Basis Function (RBF/Gaussian) Kernel
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)

Parameter:
- γ: gamma coefficient (default: 1/n_features)
```

### Sigmoid Kernel
```
K(xᵢ, xⱼ) = tanh(γxᵢᵀxⱼ + r)

Parameters:
- γ: gamma coefficient
- r: independent term
```

## Kernel Trick
Instead of explicitly mapping φ(x), we use:
```
φ(xᵢ)ᵀφ(xⱼ) = K(xᵢ, xⱼ)
```
This allows computation in high-dimensional space without storing transformed features.

## Key Parameters

### C (Regularization)
- **High C**: Hard margin, less tolerance for misclassification
- **Low C**: Soft margin, more regularization

### γ (Gamma) - for RBF, Polynomial, Sigmoid
- **High γ**: Close fit to training data (potential overfitting)
- **Low γ**: Smoother decision boundary
- **Auto**: γ = 1/n_features

### Kernel-Specific Parameters
- **degree**: Polynomial degree
- **coef0**: Independent term in polynomial/sigmoid

## Algorithm Steps
1. **Choose Kernel**: Select appropriate kernel function
2. **Solve Dual**: Find optimal α values using SMO algorithm
3. **Support Vectors**: Identify points with αᵢ > 0
4. **Compute b**: Calculate bias term from support vectors
5. **Predict**: Use kernel decision function

## Kernel Selection Guidelines

### RBF Kernel (Most Popular)
- **Good default choice**
- Works well for most datasets
- Handles non-linear relationships
- Parameters: C and γ

### Polynomial Kernel
- **Suitable for**: Image processing, text classification
- Can model interactions between features
- Parameters: C, γ, degree, coef0

### Linear Kernel
- **Use when**: Large number of features, text data
- Equivalent to linear SVM
- Fastest option

### Sigmoid Kernel
- **Less common**
- Can behave like neural networks
- May not satisfy Mercer's condition

## Advantages
- Handles non-linear classification problems
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Versatile (different kernels for different data types)

## Disadvantages
- Computationally expensive for large datasets
- Sensitive to feature scaling
- Choice of kernel and parameters crucial
- No probabilistic output
- Difficult to interpret

## Use Cases
- **Image Classification**: Complex visual patterns
- **Text Classification**: Non-linear text relationships
- **Bioinformatics**: Gene expression analysis
- **Handwriting Recognition**: Complex character patterns
- **Face Recognition**: Non-linear facial features
