# Linear Regression

## Theory
Linear regression models the relationship between a dependent variable and one independent variable by fitting a linear equation. It assumes a straight-line relationship and finds the best-fitting line through the data points.

## Key Formulas

### Simple Linear Regression
```
y = β₀ + β₁x + ε
```
Where:
- y = dependent variable (target)
- x = independent variable (feature)
- β₀ = y-intercept
- β₁ = slope
- ε = error term

### Parameter Estimation (Least Squares)
```
β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)
β₀ = ȳ - β₁x̄
```

### Alternative Form
```
β₁ = (n*Σ(xᵢyᵢ) - Σ(xᵢ)Σ(yᵢ)) / (n*Σ(xᵢ²) - (Σ(xᵢ))²)
```

### Cost Function (Mean Squared Error)
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
MSE = (1/n) Σ(yᵢ - (β₀ + β₁xᵢ))²
```

### Coefficient of Determination (R²)
```
R² = 1 - (SSres/SStot)
SSres = Σ(yᵢ - ŷᵢ)²    (Residual Sum of Squares)
SStot = Σ(yᵢ - ȳ)²     (Total Sum of Squares)
```

## How It Works
1. **Input**: Single feature x and target y
2. **Fit Line**: Find β₀ and β₁ that minimize squared errors
3. **Prediction**: ŷ = β₀ + β₁x
4. **Evaluation**: Measure fit using R², RMSE, MAE

## Key Assumptions
- **Linearity**: Relationship between x and y is linear
- **Independence**: Observations are independent
- **Homoscedasticity**: Constant variance of residuals
- **Normality**: Residuals are normally distributed

## Model Evaluation
- **R²**: Proportion of variance explained (0-1, higher better)
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Residual Analysis**: Check assumption violations

## Use Cases
- Sales vs advertising spend
- Height vs weight relationships
- Temperature vs energy consumption
- Any simple linear relationship between two variables
