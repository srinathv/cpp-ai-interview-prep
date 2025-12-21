#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Activation functions
namespace Activation {
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    float sigmoidDerivative(float x) {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }
    
    float relu(float x) {
        return std::max(0.0f, x);
    }
    
    float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }
    
    float tanh(float x) {
        return std::tanh(x);
    }
    
    float tanhDerivative(float x) {
        float t = std::tanh(x);
        return 1.0f - t * t;
    }
}

// Simple 2-layer neural network
class NeuralNetwork {
private:
    int inputSize, hiddenSize, outputSize;
    std::vector<std::vector<float>> weightsIH;  // Input to Hidden
    std::vector<std::vector<float>> weightsHO;  // Hidden to Output
    std::vector<float> biasH, biasO;
    float learningRate;
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        // Initialize input->hidden weights
        weightsIH.resize(inputSize, std::vector<float>(hiddenSize));
        for (auto& row : weightsIH) {
            for (auto& w : row) {
                w = dist(gen);
            }
        }
        
        // Initialize hidden->output weights
        weightsHO.resize(hiddenSize, std::vector<float>(outputSize));
        for (auto& row : weightsHO) {
            for (auto& w : row) {
                w = dist(gen);
            }
        }
        
        // Initialize biases
        biasH.resize(hiddenSize, 0.0f);
        biasO.resize(outputSize, 0.0f);
    }
    
public:
    NeuralNetwork(int input, int hidden, int output, float lr = 0.1f)
        : inputSize(input), hiddenSize(hidden), outputSize(output), learningRate(lr) {
        initializeWeights();
    }
    
    // Forward propagation
    std::vector<float> forward(const std::vector<float>& input,
                               std::vector<float>& hiddenLayer,
                               std::vector<float>& hiddenActivation) {
        // Hidden layer computation
        hiddenLayer.resize(hiddenSize);
        hiddenActivation.resize(hiddenSize);
        
        for (int h = 0; h < hiddenSize; ++h) {
            float sum = biasH[h];
            for (int i = 0; i < inputSize; ++i) {
                sum += input[i] * weightsIH[i][h];
            }
            hiddenLayer[h] = sum;
            hiddenActivation[h] = Activation::sigmoid(sum);
        }
        
        // Output layer computation
        std::vector<float> output(outputSize);
        for (int o = 0; o < outputSize; ++o) {
            float sum = biasO[o];
            for (int h = 0; h < hiddenSize; ++h) {
                sum += hiddenActivation[h] * weightsHO[h][o];
            }
            output[o] = Activation::sigmoid(sum);
        }
        
        return output;
    }
    
    // Backpropagation
    void backward(const std::vector<float>& input,
                  const std::vector<float>& target,
                  const std::vector<float>& output,
                  const std::vector<float>& hiddenLayer,
                  const std::vector<float>& hiddenActivation) {
        // Calculate output layer error
        std::vector<float> outputError(outputSize);
        for (int o = 0; o < outputSize; ++o) {
            outputError[o] = (target[o] - output[o]) * 
                            Activation::sigmoidDerivative(output[o]);
        }
        
        // Calculate hidden layer error
        std::vector<float> hiddenError(hiddenSize);
        for (int h = 0; h < hiddenSize; ++h) {
            float error = 0.0f;
            for (int o = 0; o < outputSize; ++o) {
                error += outputError[o] * weightsHO[h][o];
            }
            hiddenError[h] = error * Activation::sigmoidDerivative(hiddenLayer[h]);
        }
        
        // Update hidden->output weights
        for (int h = 0; h < hiddenSize; ++h) {
            for (int o = 0; o < outputSize; ++o) {
                weightsHO[h][o] += learningRate * outputError[o] * hiddenActivation[h];
            }
        }
        
        // Update output bias
        for (int o = 0; o < outputSize; ++o) {
            biasO[o] += learningRate * outputError[o];
        }
        
        // Update input->hidden weights
        for (int i = 0; i < inputSize; ++i) {
            for (int h = 0; h < hiddenSize; ++h) {
                weightsIH[i][h] += learningRate * hiddenError[h] * input[i];
            }
        }
        
        // Update hidden bias
        for (int h = 0; h < hiddenSize; ++h) {
            biasH[h] += learningRate * hiddenError[h];
        }
    }
    
    // Train on one sample
    float train(const std::vector<float>& input, const std::vector<float>& target) {
        std::vector<float> hiddenLayer, hiddenActivation;
        std::vector<float> output = forward(input, hiddenLayer, hiddenActivation);
        
        // Calculate loss (MSE)
        float loss = 0.0f;
        for (int o = 0; o < outputSize; ++o) {
            float diff = target[o] - output[o];
            loss += diff * diff;
        }
        loss /= outputSize;
        
        backward(input, target, output, hiddenLayer, hiddenActivation);
        
        return loss;
    }
    
    // Predict
    std::vector<float> predict(const std::vector<float>& input) {
        std::vector<float> hiddenLayer, hiddenActivation;
        return forward(input, hiddenLayer, hiddenActivation);
    }
};

int main() {
    std::cout << "=== Simple Neural Network ===" << std::endl;
    
    // XOR problem
    std::vector<std::vector<float>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<float>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    NeuralNetwork nn(2, 4, 1, 0.5f);
    
    // Training
    std::cout << "Training on XOR problem..." << std::endl;
    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            totalLoss += nn.train(inputs[i], targets[i]);
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / inputs.size() << std::endl;
        }
    }
    
    // Testing
    std::cout << "\n=== Testing ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = nn.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] 
                  << "] -> Output: " << output[0] 
                  << " (Expected: " << targets[i][0] << ")" << std::endl;
    }
    
    // Test activation functions
    std::cout << "\n=== Activation Functions ===" << std::endl;
    float x = 0.5f;
    std::cout << "sigmoid(" << x << ") = " << Activation::sigmoid(x) << std::endl;
    std::cout << "relu(" << x << ") = " << Activation::relu(x) << std::endl;
    std::cout << "tanh(" << x << ") = " << Activation::tanh(x) << std::endl;
    
    return 0;
}
