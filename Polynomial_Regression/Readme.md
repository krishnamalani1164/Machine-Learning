# Polynomial Regression

## Theory
Polynomial regression extends linear regression by fitting a polynomial equation to data. It captures non-linear relationships by using polynomial terms (x², x³, etc.) while still being linear in parameters.

## Key Formula

### Single Variable Polynomial
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
```

### General Form (degree d)
```
y = Σ(i=0 to d) βᵢxⁱ + ε
```

### Multiple Variables with Polynomial Terms
```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ + ... + ε
```

### Matrix Form
```
Y = Xβ + ε
```
Where X contains polynomial features: [1, x, x², x³, ...]

### Parameter Estimation
```
β = (XᵀX)⁻¹XᵀY
```

### Cost Function
```
MSE = (1/m) Σ(yᵢ - ŷᵢ)²
```

## How It Works
1. **Feature Engineering**: Transform input x → [1, x, x², x³, ..., xⁿ]
2. **Linear Model**: Apply linear regression on polynomial features
3. **Prediction**: ŷ = β₀ + β₁x + β₂x² + ... + βₙxⁿ
4. **Training**: Minimize squared errors using standard linear regression methods

## Key Considerations
- **Degree Selection**: Higher degrees = more complexity
- **Overfitting**: High-degree polynomials can overfit easily
- **Underfitting**: Low degrees may miss important patterns
- **Feature Scaling**: Polynomial terms can have vastly different scales
- **Regularization**: Ridge/Lasso often needed for stability

## Model Selection
- **Cross-validation**: Choose optimal polynomial degree
- **Bias-Variance Tradeoff**: Balance model complexity
- **Learning Curves**: Plot training vs validation error

## Use Cases
- Growth curves (population, revenue)
- Physical phenomena (projectile motion)
- Time series with non-linear trends
- Any relationship with curved patterns
- Image processing (surface fitting)
