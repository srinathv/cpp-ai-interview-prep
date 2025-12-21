# AI/ML Concepts

Fundamental machine learning concepts implemented in C++.

## Topics Covered

### 1. Neural Network Basics
- **neural_network_basics.cpp** - Simple 2-layer neural network
- Forward propagation
- Backpropagation
- Activation functions (sigmoid, ReLU, tanh)
- Training on XOR problem

### 2. Linear Models
- **linear_regression.cpp** - Linear and logistic regression
- Gradient descent optimization
- Mean squared error (MSE) loss
- Binary cross-entropy loss
- Batch vs stochastic gradient descent

### 3. Matrix Operations
- **matrix_operations.cpp** - Essential matrix operations for ML
- Matrix multiplication, transpose
- Element-wise operations
- Softmax, activation functions
- Vector operations (dot product, L2 norm)

## Key ML Concepts

### Activation Functions
| Function | Formula | Use Case | Derivative |
|----------|---------|----------|------------|
| Sigmoid | 1/(1+e^-x) | Binary classification, gates | σ(x)(1-σ(x)) |
| ReLU | max(0,x) | Hidden layers (faster training) | 1 if x>0, 0 otherwise |
| Tanh | (e^x-e^-x)/(e^x+e^-x) | Hidden layers (centered) | 1-tanh²(x) |
| Softmax | e^xi/Σe^xj | Multi-class output | Complex |

### Loss Functions
| Loss | Use Case | Formula |
|------|----------|---------|
| MSE | Regression | (1/n)Σ(y-ŷ)² |
| Binary Cross-Entropy | Binary classification | -Σ(y·log(ŷ) + (1-y)·log(1-ŷ)) |
| Categorical Cross-Entropy | Multi-class | -Σy·log(ŷ) |

### Gradient Descent Variants
1. **Batch GD** - Use all training examples
   - More stable, slower per iteration
2. **Stochastic GD** - Use one example at a time
   - Faster, noisier updates
3. **Mini-batch GD** - Use small batches
   - Balance of both approaches

## Neural Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
     ↓              ↓                 ↓
  Features    Non-linearity      Predictions
```

### Forward Propagation
1. Compute weighted sum: z = Wx + b
2. Apply activation: a = σ(z)
3. Repeat for each layer

### Backpropagation
1. Compute output error: δL = (a - y) · σ'(z)
2. Propagate error backwards: δl = (W^T·δl+1) · σ'(z)
3. Update weights: W -= lr · δ · a^T
4. Update biases: b -= lr · δ

## Common Interview Questions

- Explain backpropagation algorithm
- Why use activation functions?
- Difference between sigmoid and ReLU?
- What is vanishing gradient problem?
- Explain overfitting and regularization
- How does gradient descent work?
- What is the purpose of softmax?
- Explain bias-variance tradeoff

## Implementation Notes

These examples are educational implementations to understand concepts. For production ML:
- Use optimized libraries (PyTorch, TensorFlow, etc.)
- These C++ implementations show the math behind the libraries
- Useful for embedded systems or custom hardware
- Understanding fundamentals helps debug and optimize

## Performance Considerations

- Matrix operations are the bottleneck
- Use BLAS libraries for production (Eigen, Armadillo)
- GPU acceleration for large models (CUDA, HIP)
- Vectorization improves CPU performance
- Memory layout matters (row-major vs column-major)
