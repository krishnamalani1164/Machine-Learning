# Convolutional Neural Networks (CNN)

## Overview
CNNs are specialized neural networks designed for processing grid-like data such as images, using convolution operations to detect local features through shared parameters and spatial hierarchies.

## Core Concepts

### Convolution Operation
```
(f * g)[m,n] = ΣΣ f[i,j] × g[m-i, n-j]

For 2D convolution:
Y[i,j] = ΣₘΣₙ X[i+m, j+n] × W[m,n] + b

Where:
X = Input feature map
W = Filter/Kernel weights  
Y = Output feature map
b = Bias term
```

### Mathematical Properties
```
Commutative: f * g = g * f
Associative: (f * g) * h = f * (g * h)
Distributive: f * (g + h) = f * g + f * h
```

## CNN Architecture Components

### 1. Convolutional Layer
```
Input: H × W × D (Height × Width × Depth)
Filter: F × F × D × K (Size × Size × Input_Depth × Num_Filters)
Output: H' × W' × K

Where:
H' = (H + 2P - F) / S + 1
W' = (W + 2P - F) / S + 1
P = Padding, S = Stride
```

### 2. Pooling Layer
```
Max Pooling: Output = max(input_region)
Average Pooling: Output = mean(input_region)
Global Average Pooling: Output = mean(entire_feature_map)

Reduces spatial dimensions while retaining important features
```

### 3. Fully Connected Layer
```
Flattened_input → Dense_layer → Output
Used for final classification/regression
```

## Key Parameters

### Filter/Kernel
```
Size: Typically 3×3, 5×5, 7×7, 11×11
Depth: Matches input channels
Number: Determines output channels
Weights: Learned during training
```

### Stride
```
S = 1: Dense sampling (overlapping)
S = 2: Skip every other position
S = F: Non-overlapping windows

Larger stride → Smaller output size
```

### Padding
```
Valid: No padding (output smaller than input)
Same: Zero-padding to maintain size
P = (F-1)/2 for odd filter sizes

Output_size = (Input_size + 2P - F) / S + 1
```

## Types of Convolutions

### Standard Convolution
```
Input: H × W × C_in
Filter: F × F × C_in × C_out
Operations: F × F × C_in × H' × W' × C_out
```

### Depthwise Convolution
```
Each input channel convolved separately
Filter: F × F × 1 (applied to each channel)
Operations: F × F × C_in × H' × W'
```

### Pointwise Convolution
```
1×1 convolution across channels
Filter: 1 × 1 × C_in × C_out
Operations: C_in × H' × W' × C_out
```

### Depthwise Separable Convolution
```
Depthwise + Pointwise convolution
Total operations: F×F×C_in×H'×W' + C_in×H'×W'×C_out
Efficiency: ~8-9x fewer operations than standard
```

### Dilated/Atrous Convolution
```
Introduces gaps in filter
Dilation rate r: Insert (r-1) zeros between filter elements
Effective filter size: F + (F-1)(r-1)
Larger receptive field without more parameters
```

## Receptive Field
```
Layer 1: r₁ = F₁
Layer l: rₗ = rₗ₋₁ + (Fₗ - 1) × ∏ᵢ₌₁ˡ⁻¹ Sᵢ

Where:
r = receptive field size
F = filter size  
S = stride
```

## CNN Architectures

### LeNet-5 (1998)
```
Input(32×32) → Conv(6@28×28) → Pool(6@14×14) → 
Conv(16@10×10) → Pool(16@5×5) → FC(120) → FC(84) → FC(10)

First successful CNN for digit recognition
```

### AlexNet (2012)
```
Input(224×224×3) → Conv1(96@55×55) → Pool1 → Conv2(256@27×27) → 
Pool2 → Conv3(384@13×13) → Conv4(384@13×13) → Conv5(256@13×13) → 
Pool3 → FC(4096) → FC(4096) → FC(1000)

Innovations: ReLU, Dropout, GPU training
```

### VGG (2014)
```
Key insight: Deeper networks with smaller filters
VGG-16: 13 conv layers + 3 FC layers
Filter size: 3×3 throughout
Two 3×3 filters = one 5×5 filter (fewer parameters)
```

### ResNet (2015)
```
Skip connections: H(x) = F(x) + x
Residual block: F(x) = W₂σ(W₁x + b₁) + b₂
Output: y = F(x) + x

Solves vanishing gradient problem
Enables very deep networks (50, 101, 152 layers)
```

### Inception/GoogLeNet (2014)
```
Inception module: Multiple filter sizes in parallel
1×1, 3×3, 5×5 convolutions + 3×3 max pooling
1×1 bottleneck for dimension reduction
```

### MobileNet (2017)
```
Depthwise separable convolutions
Designed for mobile/embedded devices
Width multiplier α and resolution multiplier ρ
```

## Activation Functions in CNNs

### ReLU (Most Common)
```
f(x) = max(0, x)
Advantages: Non-saturating, computationally efficient
Problem: Dead neurons
```

### Leaky ReLU
```
f(x) = max(αx, x), α = 0.01
Solves dead neuron problem
```

### ELU (Exponential Linear Unit)
```
f(x) = x if x > 0, else α(eˣ - 1)
Smooth, zero-centered output
```

## Batch Normalization
```
Input: Batch of activations {x₁, x₂, ..., xₘ}
μ_B = (1/m)Σxᵢ (batch mean)
σ²_B = (1/m)Σ(xᵢ - μ_B)² (batch variance)
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε) (normalize)
yᵢ = γx̂ᵢ + β (scale and shift)

Benefits: Faster training, higher learning rates, regularization
```

## Loss Functions

### Classification
```
Softmax + Cross-Entropy:
p(yᵢ = j) = e^(zⱼ) / Σₖe^(zₖ)
L = -Σᵢ yᵢ log(pᵢ)
```

### Object Detection
```
Classification Loss + Localization Loss
L = L_cls + λL_loc
Smooth L1 loss for bounding box regression
```

### Semantic Segmentation
```
Pixel-wise cross-entropy
L = -ΣᵢΣⱼ yᵢⱼ log(pᵢⱼ)
Where i,j are pixel coordinates
```

## Training Techniques

### Data Augmentation
```
Geometric: Rotation, scaling, flipping, cropping
Photometric: Brightness, contrast, saturation
Noise: Gaussian noise, dropout
Mixup: Linear interpolation between samples
CutMix: Replace patches between images
```

### Transfer Learning
```
1. Pre-trained model on large dataset (ImageNet)
2. Fine-tune on target dataset
3. Options:
   - Freeze early layers, train last layers
   - Fine-tune entire network with low learning rate
   - Use as feature extractor
```

### Learning Rate Scheduling
```
Step decay: lr = lr₀ × γ^(epoch/step_size)
Cosine annealing: lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
Warm restart: Periodic lr resets
```

## Advanced CNN Techniques

### Attention Mechanisms
```
Spatial Attention: Focus on important spatial locations
Channel Attention: Weight different feature channels
Self-Attention: Relate different positions in feature map
```

### Residual Connections
```
Dense connections: Each layer connects to all previous
Highway networks: Learnable gating mechanisms
Feature pyramid networks: Multi-scale features
```

### Neural Architecture Search (NAS)
```
Automated architecture design
Search space: Possible operations and connections
Search strategy: Reinforcement learning, evolutionary
Performance estimation: Training smaller models
```

## Object Detection CNNs

### R-CNN Family
```
R-CNN: Region proposals + CNN features
Fast R-CNN: RoI pooling for efficiency  
Faster R-CNN: RPN for proposal generation
```

### YOLO (You Only Look Once)
```
Single-stage detector
Grid-based detection
Real-time performance
Loss: Classification + Localization + Confidence
```

### SSD (Single Shot Detector)
```
Multi-scale feature maps
Default boxes at different scales
Combines YOLO speed with better accuracy
```

## Semantic Segmentation CNNs

### FCN (Fully Convolutional Network)
```
Replace FC layers with conv layers
Upsampling for pixel-wise prediction
Skip connections for fine details
```

### U-Net
```
Encoder-decoder architecture
Skip connections between encoder-decoder
Popular for medical image segmentation
```

### DeepLab
```
Atrous convolution for larger receptive field
Atrous Spatial Pyramid Pooling (ASPP)
Conditional Random Fields (CRF) post-processing
```

## Performance Optimization

### Model Compression
```
Pruning: Remove unimportant weights/neurons
Quantization: Reduce precision (FP32 → INT8)
Knowledge distillation: Teacher-student training
Low-rank approximation: Matrix factorization
```

### Hardware Optimization
```
GPU acceleration: Parallel convolution operations
Tensor cores: Mixed precision training
Model optimization: TensorRT, ONNX
Edge deployment: TensorFlow Lite, Core ML
```

## Evaluation Metrics

### Classification
```
Top-1 Accuracy: Correct prediction probability
Top-5 Accuracy: Target in top 5 predictions
Confusion Matrix: Class-wise performance
```

### Object Detection
```
mAP (mean Average Precision): Area under PR curve
IoU (Intersection over Union): Overlap measure
FPS (Frames Per Second): Speed metric
```

### Segmentation
```
Pixel Accuracy: Correctly classified pixels
mIoU (mean Intersection over Union): Per-class IoU
Dice Coefficient: 2×overlap / (pred + gt)
```

## Popular Frameworks & Libraries
```
PyTorch: torchvision.models
TensorFlow/Keras: tf.keras.applications
Detectron2: Facebook's detection library
MMDetection: OpenMMLab's toolbox
YOLO: Ultralytics implementation
```

## Applications
- **Image Classification**: Object recognition, medical diagnosis
- **Object Detection**: Autonomous driving, surveillance
- **Semantic Segmentation**: Medical imaging, satellite imagery
- **Face Recognition**: Security systems, photo tagging
- **Style Transfer**: Artistic applications, photo editing
- **Medical Imaging**: X-ray, MRI, CT scan analysis
- **Autonomous Vehicles**: Road sign detection, lane following
- **Manufacturing**: Quality control, defect detection
- **Agriculture**: Crop monitoring, disease detection
- **Retail**: Product recognition, inventory management

## Advantages
- **Translation invariance**: Shared parameters across spatial locations
- **Local connectivity**: Efficient feature detection
- **Hierarchical features**: Low to high-level representations
- **Parameter sharing**: Fewer parameters than fully connected
- **GPU friendly**: Parallel convolution operations

## Challenges
- **Large datasets**: Require substantial training data
- **Computational cost**: Memory and processing intensive
- **Hyperparameter sensitivity**: Architecture design choices
- **Interpretability**: Black box nature
- **Adversarial attacks**: Vulnerability to crafted inputs
