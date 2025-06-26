# Artificial Neural Networks (ANN)

## Overview
Artificial Neural Networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information through weighted connections.

## Biological Inspiration

### Neuron Components
```
Dendrites → Cell Body → Axon → Synapses
(Input)   → (Process) → (Output) → (Connection)
```

### Artificial Neuron Model
```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b
Net Input: z = Σ(wᵢxᵢ) + b
Output: y = f(z)
```

## Perceptron Model

### Single Perceptron
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
y = f(z)

Where f(z) is the activation function
```

### Perceptron Learning Rule
```
For each training example (x, target):
1. Calculate output: y = f(Σwᵢxᵢ + b)
2. Update weights: wᵢ = wᵢ + η(target - y)xᵢ
3. Update bias: b = b + η(target - y)

Where η is the learning rate
```

### Perceptron Limitations
- Only linearly separable problems
- Cannot solve XOR problem
- Single layer limitation

## Multi-Layer Perceptron (MLP)

### Architecture
```
Input Layer → Hidden Layer(s) → Output Layer

Layer l: aˡ = f(Wˡaˡ⁻¹ + bˡ)

Where:
aˡ = activation vector at layer l
Wˡ = weight matrix for layer l
bˡ = bias vector for layer l
f = activation function
```

### Universal Approximation Theorem
Any continuous function can be approximated by a neural network with at least one hidden layer and sufficient neurons.

## Activation Functions

### Linear Activation
```
f(x) = x
f'(x) = 1
Use: Output layer for regression
```

### Step Function
```
f(x) = 1 if x ≥ 0, else 0
f'(x) = 0 (undefined at x=0)
Use: Binary classification (historical)
```

### Sigmoid
```
f(x) = 1 / (1 + e⁻ˣ)
f'(x) = f(x)(1 - f(x))
Range: (0, 1)
Use: Binary classification, gate functions
```

### Hyperbolic Tangent (Tanh)
```
f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
f'(x) = 1 - f(x)²
Range: (-1, 1)
Use: Hidden layers (zero-centered)
```

### Rectified Linear Unit (ReLU)
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
Use: Most popular for hidden layers
```

### Softmax
```
f(xᵢ) = eˣⁱ / Σⱼeˣʲ
Use: Multi-class classification output
```

## Forward Propagation

### Step-by-Step Process
```
1. Input Layer: a⁰ = x (input features)

2. Hidden Layer 1: 
   z¹ = W¹a⁰ + b¹
   a¹ = f(z¹)

3. Hidden Layer 2:
   z² = W²a¹ + b²  
   a² = f(z²)

4. Output Layer:
   zᴸ = Wᴸaᴸ⁻¹ + bᴸ
   aᴸ = f(zᴸ) = ŷ (prediction)
```

### Matrix Form
```
For layer l:
Z[l] = W[l] · A[l-1] + B[l]
A[l] = g(Z[l])

Where:
Z[l] = weighted input matrix
A[l] = activation matrix
W[l] = weight matrix
B[l] = bias matrix
g() = activation function
```

## Backpropagation Algorithm

### Cost Function
```
Mean Squared Error: J = (1/m)Σ(yᵢ - ŷᵢ)²
Cross-Entropy: J = -(1/m)Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

### Gradient Computation
```
Output Layer Error:
δᴸ = ∇ₐJ ⊙ f'(zᴸ)

Hidden Layer Error:
δˡ = ((Wˡ⁺¹)ᵀδˡ⁺¹) ⊙ f'(zˡ)

Weight Gradients:
∂J/∂Wˡ = δˡ(aˡ⁻¹)ᵀ

Bias Gradients:
∂J/∂bˡ = δˡ
```

### Weight Update
```
Wˡ = Wˡ - η(∂J/∂Wˡ)
bˡ = bˡ - η(∂J/∂bˡ)

Where η is the learning rate
```

## Training Process

### Batch Training
```
1. Initialize weights randomly
2. For each epoch:
   a. Forward propagation on entire dataset
   b. Calculate total loss
   c. Backpropagation to compute gradients
   d. Update all weights simultaneously
3. Repeat until convergence
```

### Stochastic Gradient Descent (SGD)
```
1. Initialize weights randomly
2. For each epoch:
   a. Shuffle training data
   b. For each sample:
      - Forward propagation
      - Calculate loss
      - Backpropagation
      - Update weights
3. Repeat until convergence
```

### Mini-Batch Training
```
Compromise between batch and SGD:
- Process small batches (32, 64, 128 samples)
- More stable than SGD
- More efficient than full batch
```

## Weight Initialization

### Random Initialization
```
Small random values: W ~ N(0, 0.01)
Problem: Symmetry breaking
```

### Xavier/Glorot Initialization
```
W ~ N(0, √(2/(nᵢₙ + nₒᵤₜ)))
Good for sigmoid/tanh activations
```

### He Initialization
```
W ~ N(0, √(2/nᵢₙ))
Good for ReLU activations
```

## Common Problems & Solutions

### Vanishing Gradient Problem
```
Problem: Gradients become very small in deep networks
Solutions:
- Better activation functions (ReLU)
- Proper weight initialization
- Batch normalization
- Skip connections
```

### Exploding Gradient Problem
```
Problem: Gradients become very large
Solutions:
- Gradient clipping
- Proper weight initialization
- Lower learning rates
```

### Overfitting
```
Problem: Model memorizes training data
Solutions:
- Regularization (L1, L2)
- Dropout
- Early stopping
- More training data
- Cross-validation
```

## Regularization Techniques

### L1 Regularization (Lasso)
```
J = Loss + λΣ|wᵢ|
Promotes sparsity
```

### L2 Regularization (Ridge)
```
J = Loss + λΣwᵢ²
Prevents large weights
```

### Dropout
```
Training: Randomly set neurons to 0 with probability p
Testing: Scale outputs by (1-p)
```

## Network Architectures

### Feedforward Networks
```
Information flows in one direction
Input → Hidden → Output
```

### Fully Connected Networks
```
Every neuron connected to every neuron in next layer
Dense connectivity
```

### Deep Networks
```
Multiple hidden layers
Hierarchical feature learning
```

## Hyperparameter Tuning

### Key Hyperparameters
```
- Learning rate (η)
- Number of hidden layers
- Number of neurons per layer
- Batch size
- Activation functions
- Regularization parameters
```

### Learning Rate Selection
```
Too small: Slow convergence
Too large: Overshooting, instability
Adaptive methods: Adam, AdaGrad, RMSprop
```

## Performance Metrics

### Classification
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Regression
```
MAE = (1/n)Σ|yᵢ - ŷᵢ|
MSE = (1/n)Σ(yᵢ - ŷᵢ)²
RMSE = √MSE
R² = 1 - (SS_res / SS_tot)
```

## Implementation Example (Pseudocode)
```python
class NeuralNetwork:
    def __init__(self, layers):
        self.weights = initialize_weights(layers)
        self.biases = initialize_biases(layers)
    
    def forward(self, X):
        activation = X
        for W, b in zip(self.weights, self.biases):
            z = np.dot(activation, W) + b
            activation = sigmoid(z)
        return activation
    
    def backward(self, X, y, learning_rate):
        # Compute gradients using backpropagation
        # Update weights and biases
        pass
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = compute_loss(y, predictions)
            self.backward(X, y, learning_rate)
```

## Applications
- **Pattern Recognition**: Image, speech, handwriting
- **Classification**: Spam detection, medical diagnosis
- **Regression**: Price prediction, forecasting
- **Function Approximation**: Complex mathematical functions
- **Data Mining**: Customer segmentation, recommendation
- **Control Systems**: Robotics, autonomous systems
- **Game Playing**: Strategic decision making
- **Signal Processing**: Noise reduction, feature extraction

## Advantages
- **Universal approximators**: Can learn complex patterns
- **Parallel processing**: Distributed computation
- **Fault tolerance**: Graceful degradation
- **Adaptability**: Learn from experience
- **Non-linear modeling**: Handle complex relationships

## Limitations
- **Black box**: Difficult to interpret
- **Local minima**: May not find global optimum
- **Overfitting**: Memorization vs generalization
- **Computational cost**: Resource intensive
- **Data requirements**: Need large datasets
