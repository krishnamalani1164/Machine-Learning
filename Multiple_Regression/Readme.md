# Multiple Linear Regression

## Theory
Multiple linear regression extends simple linear regression to model the relationship between a dependent variable and multiple independent variables. It assumes a linear relationship between predictors and the target variable.

## Key Formula

### Prediction Equation
```
y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ + ε
```

### Matrix Form
```
Y = Xβ + ε
```
Where:
- Y = dependent variable vector
- X = design matrix (features)
- β = coefficient vector
- ε = error term

### Parameter Estimation (Normal Equation)
```
β = (XᵀX)⁻¹XᵀY
```

### Cost Function (Mean Squared Error)
```
MSE = (1/m) Σ(yᵢ - ŷᵢ)²
```

## How It Works
1. **Input**: Multiple features x₁, x₂, ..., xₙ
2. **Linear Combination**: Weighted sum of features
3. **Prediction**: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
4. **Training**: Minimize squared errors using gradient descent or normal equation

## Key Assumptions
- **Linearity**: Linear relationship between variables
- **Independence**: Observations are independent
- **Homoscedasticity**: Constant variance of residuals
- **Normality**: Residuals are normally distributed
- **No Multicollinearity**: Features aren't highly correlated

## Model Evaluation
- **R²**: Coefficient of determination (0-1, higher is better)
- **Adjusted R²**: R² adjusted for number of predictors
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

## Use Cases
- House price prediction (size, location, rooms, etc.)
- Sales forecasting (advertising spend, seasonality, competition)
- Stock price modeling
- Any continuous target with multiple predictors
