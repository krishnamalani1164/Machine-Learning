# Support Vector Regression (SVR)

## Theory
Support Vector Regression extends SVM to regression problems. Instead of finding a decision boundary, SVR finds a function that deviates from actual targets by at most ε (epsilon) while being as flat as possible.

## Key Formulas

### Linear SVR
```
f(x) = wᵀx + b
```

### Kernel SVR (Non-linear)
```
f(x) = Σ(αᵢ - αᵢ*)K(xᵢ, x) + b
```

### Optimization Problem
```
Minimize: (1/2)||w||² + C Σ(ξᵢ + ξᵢ*)

Subject to:
yᵢ - wᵀxᵢ - b ≤ ε + ξᵢ
wᵀxᵢ + b - yᵢ ≤ ε + ξᵢ*
ξᵢ, ξᵢ* ≥ 0
```

### ε-insensitive Loss Function
```
L_ε(y, f(x)) = {
  0,           if |y - f(x)| ≤ ε
  |y - f(x)| - ε,  otherwise
}
```

### Common Kernels
```
Linear: K(xᵢ, xⱼ) = xᵢᵀxⱼ
Polynomial: K(xᵢ, xⱼ) = (γxᵢᵀxⱼ + r)^d
RBF/Gaussian: K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
Sigmoid: K(xᵢ, xⱼ) = tanh(γxᵢᵀxⱼ + r)
```

## Key Concepts

### ε-tube
Creates a margin of tolerance around the regression line. Points within ε-tube have zero loss.

### Support Vectors
Data points that lie on or outside the ε-tube boundaries. Only these points influence the model.

### Dual Variables
αᵢ and αᵢ* are Lagrange multipliers from optimization that determine support vectors.

## Key Parameters
- **C**: Regularization parameter (higher C = less regularization)
- **ε (epsilon)**: Width of ε-tube (tolerance for errors)
- **kernel**: Kernel function type (linear, rbf, poly, sigmoid)
- **γ (gamma)**: Kernel coefficient for RBF/poly/sigmoid
- **degree**: Degree for polynomial kernel

## Algorithm Steps
1. **Setup**: Define ε-tube around target values
2. **Optimize**: Solve quadratic programming problem
3. **Support Vectors**: Identify points outside ε-tube
4. **Model**: Use support vectors and dual variables for prediction
5. **Kernel Trick**: Apply kernel function for non-linear relationships

## Advantages
- Effective for high-dimensional data
- Memory efficient (uses only support vectors)
- Versatile (different kernels for different relationships)
- Robust to outliers (ε-insensitive loss)

## Disadvantages
- Sensitive to feature scaling
- No probabilistic output
- Choice of kernel and parameters crucial
- Computationally expensive for large datasets

## Use Cases
- Financial time series prediction
- Image processing and computer vision
- Bioinformatics (gene expression analysis)
- Engineering applications (system modeling)
- Any regression with complex non-linear patterns
