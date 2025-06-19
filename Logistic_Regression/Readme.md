# Logistic Regression

## Theory
Logistic regression is a statistical method for binary classification that models the probability of an event occurring. Unlike linear regression, it uses the logistic (sigmoid) function to constrain outputs between 0 and 1.

## Key Formula

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

### Prediction
```
h(x) = σ(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)
h(x) = 1 / (1 + e^(-(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)))
```

### Cost Function
```
J(θ) = -(1/m) Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

## How It Works
1. **Input**: Features are combined linearly: `z = θ₀ + θ₁x₁ + θ₂x₂ + ...`
2. **Transform**: Apply sigmoid function to get probability: `p = σ(z)`
3. **Classify**: If p ≥ 0.5 → Class 1, else Class 0
4. **Train**: Minimize cost function using gradient descent

## Key Properties
- Output range: [0, 1] (probabilities)
- S-shaped (sigmoid) curve
- No assumptions about feature distributions
- Robust to outliers compared to linear regression

## Use Cases
- Email spam detection
- Medical diagnosis
- Marketing response prediction
- Any binary classification problem
