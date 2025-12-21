#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class LinearRegression {
private:
    std::vector<float> weights;
    float bias;
    float learningRate;
    
public:
    LinearRegression(int features, float lr = 0.01f) 
        : weights(features, 0.0f), bias(0.0f), learningRate(lr) {}
    
    // Predict: y = w^T * x + b
    float predict(const std::vector<float>& x) const {
        float y = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            y += weights[i] * x[i];
        }
        return y;
    }
    
    // Train using gradient descent
    void train(const std::vector<std::vector<float>>& X,
               const std::vector<float>& y,
               int epochs) {
        int n = X.size();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0.0f;
            
            // For each training example
            for (int i = 0; i < n; ++i) {
                float pred = predict(X[i]);
                float error = pred - y[i];
                
                // Update weights: w = w - lr * error * x
                for (size_t j = 0; j < weights.size(); ++j) {
                    weights[j] -= learningRate * error * X[i][j];
                }
                
                // Update bias: b = b - lr * error
                bias -= learningRate * error;
                
                totalLoss += error * error;
            }
            
            if (epoch % 100 == 0) {
                float mse = totalLoss / n;
                std::cout << "Epoch " << epoch << ", MSE: " << mse << std::endl;
            }
        }
    }
    
    // Batch gradient descent (more efficient)
    void trainBatch(const std::vector<std::vector<float>>& X,
                    const std::vector<float>& y,
                    int epochs) {
        int n = X.size();
        int features = weights.size();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<float> gradW(features, 0.0f);
            float gradB = 0.0f;
            float totalLoss = 0.0f;
            
            // Compute gradients
            for (int i = 0; i < n; ++i) {
                float pred = predict(X[i]);
                float error = pred - y[i];
                
                for (int j = 0; j < features; ++j) {
                    gradW[j] += error * X[i][j];
                }
                gradB += error;
                totalLoss += error * error;
            }
            
            // Update parameters
            for (int j = 0; j < features; ++j) {
                weights[j] -= learningRate * gradW[j] / n;
            }
            bias -= learningRate * gradB / n;
            
            if (epoch % 100 == 0) {
                float mse = totalLoss / n;
                std::cout << "Epoch " << epoch << ", MSE: " << mse << std::endl;
            }
        }
    }
    
    void printModel() const {
        std::cout << "Model: y = ";
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << weights[i] << "*x" << i;
            if (i < weights.size() - 1) std::cout << " + ";
        }
        std::cout << " + " << bias << std::endl;
    }
};

// Logistic Regression for binary classification
class LogisticRegression {
private:
    std::vector<float> weights;
    float bias;
    float learningRate;
    
    float sigmoid(float z) const {
        return 1.0f / (1.0f + std::exp(-z));
    }
    
public:
    LogisticRegression(int features, float lr = 0.1f)
        : weights(features, 0.0f), bias(0.0f), learningRate(lr) {}
    
    float predict(const std::vector<float>& x) const {
        float z = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            z += weights[i] * x[i];
        }
        return sigmoid(z);
    }
    
    int predictClass(const std::vector<float>& x) const {
        return predict(x) >= 0.5f ? 1 : 0;
    }
    
    void train(const std::vector<std::vector<float>>& X,
               const std::vector<int>& y,
               int epochs) {
        int n = X.size();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0.0f;
            
            for (int i = 0; i < n; ++i) {
                float pred = predict(X[i]);
                float error = pred - y[i];
                
                for (size_t j = 0; j < weights.size(); ++j) {
                    weights[j] -= learningRate * error * X[i][j];
                }
                bias -= learningRate * error;
                
                // Binary cross-entropy loss
                totalLoss += -(y[i] * std::log(pred + 1e-7f) + 
                              (1 - y[i]) * std::log(1 - pred + 1e-7f));
            }
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / n << std::endl;
            }
        }
    }
};

int main() {
    std::cout << "=== Linear Regression ===" << std::endl;
    
    // Generate synthetic data: y = 2x + 3 + noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.5f);
    
    std::vector<std::vector<float>> X;
    std::vector<float> y;
    
    for (int i = 0; i < 100; ++i) {
        float x = static_cast<float>(i) / 10.0f;
        X.push_back({x});
        y.push_back(2.0f * x + 3.0f + noise(gen));
    }
    
    LinearRegression lr(1, 0.01f);
    lr.trainBatch(X, y, 1000);
    lr.printModel();
    
    std::cout << "\nTest predictions:" << std::endl;
    for (float x : {0.0f, 5.0f, 10.0f}) {
        float pred = lr.predict({x});
        float expected = 2.0f * x + 3.0f;
        std::cout << "x=" << x << " -> pred=" << pred 
                  << " (expected ~" << expected << ")" << std::endl;
    }
    
    std::cout << "\n=== Logistic Regression ===" << std::endl;
    
    // Binary classification data
    std::vector<std::vector<float>> X_class = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
        {5.0, 1.0}, {6.0, 2.0}, {7.0, 3.0}, {8.0, 4.0}
    };
    std::vector<int> y_class = {0, 0, 0, 0, 1, 1, 1, 1};
    
    LogisticRegression logit(2, 0.1f);
    logit.train(X_class, y_class, 1000);
    
    std::cout << "\nTest predictions:" << std::endl;
    for (const auto& x : X_class) {
        int pred = logit.predictClass(x);
        std::cout << "x=[" << x[0] << "," << x[1] << "] -> class " << pred << std::endl;
    }
    
    return 0;
}
