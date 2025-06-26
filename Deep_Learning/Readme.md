# Deep Learning Fundamentals

## Overview
Deep Learning uses artificial neural networks with multiple layers to learn hierarchical representations of data for complex pattern recognition tasks.

## Neural Network Basics

### Perceptron
```
Output = σ(Σ(wᵢxᵢ) + b)

Where:
w = weights
x = inputs  
b = bias
σ = activation function
```

### Multi-Layer Perceptron (MLP)
```
Hidden Layer: h = σ(Wx + b)
Output Layer: y = σ(Vh + c)
```

## Activation Functions

### Common Activations
```
Sigmoid: σ(x) = 1 / (1 + e^(-x))
Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
ReLU: f(x) = max(0, x)
Leaky ReLU: f(x) = max(αx, x), α = 0.01
Swish: f(x) = x × σ(x)
GELU: f(x) = x × Φ(x)
```

### Derivative Properties
```
σ'(x) = σ(x)(1 - σ(x))    # Sigmoid
tanh'(x) = 1 - tanh²(x)    # Tanh
ReLU'(x) = 1 if x > 0 else 0    # ReLU
```

## Loss Functions

### Regression
```
Mean Squared Error: MSE = (1/n)Σ(yᵢ - ŷᵢ)²
Mean Absolute Error: MAE = (1/n)Σ|yᵢ - ŷᵢ|
Huber Loss: L_δ = 0.5(y-ŷ)² if |y-ŷ| ≤ δ, else δ|y-ŷ| - 0.5δ²
```

### Classification
```
Binary Cross-Entropy: -Σ(y log(ŷ) + (1-y)log(1-ŷ))
Categorical Cross-Entropy: -Σy log(ŷ)
Focal Loss: -α(1-p)^γ log(p)    # For imbalanced data
```

## Backpropagation Algorithm

### Forward Pass
```
Layer l: z^(l) = W^(l)a^(l-1) + b^(l)
         a^(l) = σ(z^(l))
```

### Backward Pass
```
Output Error: δ^(L) = ∇_a C ⊙ σ'(z^(L))
Hidden Error: δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^(l))

Gradients:
∂C/∂W^(l) = δ^(l)(a^(l-1))^T
∂C/∂b^(l) = δ^(l)
```

## Optimization Algorithms

### Gradient Descent Variants
```
SGD: θₜ₊₁ = θₜ - η∇J(θₜ)
Momentum: vₜ₊₁ = γvₜ + η∇J(θₜ), θₜ₊₁ = θₜ - vₜ₊₁
```

### Adaptive Methods
```
AdaGrad: θₜ₊₁ = θₜ - η/√(Gₜ + ε) ⊙ ∇J(θₜ)
RMSprop: Eₜ = γEₜ₋₁ + (1-γ)(∇J(θₜ))²
Adam: mₜ = β₁mₜ₋₁ + (1-β₁)∇J(θₜ)
      vₜ = β₂vₜ₋₁ + (1-β₂)(∇J(θₜ))²
      θₜ₊₁ = θₜ - η(m̂ₜ/√v̂ₜ + ε)
```

## Convolutional Neural Networks (CNNs)

### Convolution Operation
```
(f * g)[n] = Σ f[m]g[n-m]

Feature Map: Y[i,j] = Σₘ Σₙ X[i+m, j+n] × W[m,n] + b
```

### CNN Components
```
Input → Conv → Activation → Pooling → Conv → ... → FC → Output

Conv Layer: Learns local features
Pooling: Reduces spatial dimensions
- Max Pooling: max(region)
- Average Pooling: mean(region)
```

### Key Parameters
```
Filter Size: Typically 3×3, 5×5, 7×7
Stride: Step size for convolution
Padding: Border handling (same/valid)
Receptive Field: Input region affecting one output
```

## Recurrent Neural Networks (RNNs)

### Vanilla RNN
```
hₜ = tanh(Wₕₕhₜ₋₁ + Wₓₕxₜ + bₕ)
yₜ = Wₕᵧhₜ + bᵧ
```

### Long Short-Term Memory (LSTM)
```
Forget Gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
Input Gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
Candidate: C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)
Cell State: Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
Output Gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
Hidden State: hₜ = oₜ * tanh(Cₜ)
```

### Gated Recurrent Unit (GRU)
```
Reset Gate: rₜ = σ(Wr·[hₜ₋₁, xₜ])
Update Gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])
Candidate: h̃ₜ = tanh(W·[rₜ * hₜ₋₁, xₜ])
Output: hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

## Attention & Transformers

### Attention Mechanism
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Self-Attention: Q, K, V from same input
Multi-Head: Attention(Q,K,V) = Concat(head₁,...,headₕ)W^O
```

### Transformer Architecture
```
Encoder: Multi-Head Attention → Add & Norm → FFN → Add & Norm
Decoder: Masked MHA → Add & Norm → Cross MHA → Add & Norm → FFN
```

### Positional Encoding
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

## Regularization Techniques

### Dropout
```
Training: yᵢ = xᵢ/p with probability p, else 0
Inference: y = x (no scaling needed)
```

### Batch Normalization
```
μ_B = (1/m)Σxᵢ
σ²_B = (1/m)Σ(xᵢ - μ_B)²
x̂ᵢ = (xᵢ - μ_B)/√(σ²_B + ε)
yᵢ = γx̂ᵢ + β
```

### Layer Normalization
```
Normalize across features instead of batch dimension
```

## Advanced Architectures

### Residual Networks (ResNet)
```
Skip Connection: F(x) + x
Identity Mapping: H(x) = F(x) + x
```

### Generative Adversarial Networks (GANs)
```
Generator: G(z) → fake data
Discriminator: D(x) → real/fake probability

Min-Max Game: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
```

### Variational Autoencoders (VAEs)
```
Encoder: q(z|x) ≈ N(μ(x), σ²(x))
Decoder: p(x|z)
Loss: -E[log p(x|z)] + KL(q(z|x)||p(z))
```

## Training Best Practices

### Learning Rate Scheduling
```
Step Decay: lr = lr₀ × γ^(epoch/step_size)
Exponential: lr = lr₀ × e^(-λt)
Cosine Annealing: lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
```

### Weight Initialization
```
Xavier/Glorot: W ~ N(0, 2/(n_in + n_out))
He: W ~ N(0, 2/n_in)    # For ReLU
```

### Data Augmentation
- **Images**: Rotation, scaling, cropping, flipping
- **Text**: Synonym replacement, back-translation
- **Audio**: Time stretching, pitch shifting

## Popular Frameworks
- **PyTorch**: Dynamic computation graphs
- **TensorFlow/Keras**: Production-ready ecosystem
- **JAX**: High-performance ML research
- **Hugging Face**: Pre-trained model hub

## Applications
- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Translation, summarization, chatbots
- **Speech Recognition**: Voice assistants, transcription
- **Recommendation Systems**: Content and collaborative filtering
- **Game Playing**: AlphaGo, OpenAI Five
- **Autonomous Vehicles**: Perception and decision making
- **Drug Discovery**: Molecular property prediction
- **Finance**: Algorithmic trading, fraud detection

## Model Evaluation
```
Accuracy: (TP + TN) / (TP + TN + FP + FN)
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
AUC-ROC: Area under ROC curve
```

## Common Issues & Solutions
- **Overfitting**: Regularization, dropout, early stopping
- **Vanishing Gradients**: Skip connections, better initialization
- **Exploding Gradients**: Gradient clipping
- **Mode Collapse** (GANs): Better training techniques, diverse losses
