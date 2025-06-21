# Naive Bayes Classification

## Theory
Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this strong assumption, it often performs well in practice, especially for text classification.

## Key Formulas

### Bayes' Theorem
```
P(y|x) = P(x|y) × P(y) / P(x)
```

### Naive Bayes Classifier
```
ŷ = argmax_y P(y|x₁, x₂, ..., xₙ)
ŷ = argmax_y P(y) × ∏(i=1 to n) P(xᵢ|y)
```

### Posterior Probability
```
P(y|x₁, x₂, ..., xₙ) = P(y) × ∏(i=1 to n) P(xᵢ|y) / P(x₁, x₂, ..., xₙ)
```

### Prior Probability
```
P(y) = count(y) / total_samples
```

### Likelihood Estimation

#### Gaussian Naive Bayes (Continuous Features)
```
P(xᵢ|y) = (1/√(2πσᵧ²)) × exp(-(xᵢ - μᵧ)²/(2σᵧ²))

μᵧ = mean of feature i for class y
σᵧ² = variance of feature i for class y
```

#### Multinomial Naive Bayes (Discrete Features)
```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × n_features)
```
Where α is smoothing parameter (Laplace smoothing)

#### Bernoulli Naive Bayes (Binary Features)
```
P(xᵢ|y) = P(xᵢ|y)^xᵢ × (1 - P(xᵢ|y))^(1-xᵢ)
```

## Algorithm Steps
1. **Training Phase**:
   - Calculate prior probabilities P(y) for each class
   - Calculate likelihood P(xᵢ|y) for each feature given each class
   - Store these probabilities

2. **Prediction Phase**:
   - For new instance, calculate posterior probability for each class
   - Apply independence assumption: multiply individual likelihoods
   - Choose class with highest posterior probability

## Types of Naive Bayes

### Gaussian Naive Bayes
- **Use**: Continuous features
- **Assumption**: Features follow normal distribution
- **Formula**: Uses mean and variance

### Multinomial Naive Bayes
- **Use**: Discrete features (word counts, frequencies)
- **Assumption**: Features follow multinomial distribution
- **Common**: Text classification, document analysis

### Bernoulli Naive Bayes
- **Use**: Binary features (presence/absence)
- **Assumption**: Features are binary
- **Common**: Text classification with binary word occurrence

## Key Parameters
- **alpha**: Smoothing parameter (default: 1.0)
  - Prevents zero probabilities
  - Higher alpha = more smoothing
- **var_smoothing**: Gaussian NB variance smoothing
- **class_prior**: Prior probabilities (if not uniform)

## Advantages
- Simple and fast
- Works well with small training datasets
- Handles multi-class classification naturally
- Not sensitive to irrelevant features
- Good baseline for text classification
- Provides probabilistic predictions

## Disadvantages
- Strong independence assumption (rarely true in practice)
- Can be outperformed by more sophisticated methods
- Requires smoothing for unseen feature values
- Poor performance when features are highly correlated

## Use Cases
- **Text Classification**: Spam detection, sentiment analysis, document categorization
- **Medical Diagnosis**: Symptom-based diagnosis
- **Recommendation Systems**: User preference prediction
- **Real-time Predictions**: Fast classification needed
- **Categorical Data**: When features are discrete/categorical
