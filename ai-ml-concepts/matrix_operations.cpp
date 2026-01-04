#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

class Matrix {
private:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
public:
    Matrix(int r, int c, float val = 0.0f) : rows(r), cols(c) {
        data.resize(rows, std::vector<float>(cols, val));
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    
    float& operator()(int i, int j) { return data[i][j]; }
    const float& operator()(int i, int j) const { return data[i][j]; }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match");
        }
        
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Element-wise operations
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] - other(i, j);
            }
        }
        return result;
    }
    
    // Scalar multiplication
    Matrix operator*(float scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    // Apply function element-wise
    Matrix apply(float (*func)(float)) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = func(data[i][j]);
            }
        }
        return result;
    }
    
    // Softmax (for classification)
    Matrix softmax() const {
        if (rows != 1 && cols != 1) {
            throw std::invalid_argument("Softmax requires 1D matrix");
        }
        
        Matrix result(rows, cols);
        float maxVal = data[0][0];
        float sum = 0.0f;
        
        // Find max for numerical stability
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                maxVal = std::max(maxVal, data[i][j]);
            }
        }
        
        // Compute exp and sum
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = std::exp(data[i][j] - maxVal);
                sum += result(i, j);
            }
        }
        
        // Normalize
        return result * (1.0f / sum);
    }
    
    // Random initialization
    void randomize(float min = -1.0f, float max = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dist(gen);
            }
        }
    }
    
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) 
                         << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Common ML operations
namespace MLOps {
    // ReLU activation
    float relu(float x) {
        return std::max(0.0f, x);
    }
    
    // Sigmoid activation
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    // Dot product
    float dot(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    // L2 norm
    float l2Norm(const std::vector<float>& vec) {
        float sum = 0.0f;
        for (float v : vec) {
            sum += v * v;
        }
        return std::sqrt(sum);
    }
}

int main() {
    std::cout << "=== Matrix Operations for ML ===" << std::endl;
    
    // Create matrices
    Matrix A(3, 2);
    A.randomize(0.0f, 1.0f);
    
    Matrix B(2, 3);
    B.randomize(0.0f, 1.0f);
    
    std::cout << "\nMatrix A (3x2):" << std::endl;
    A.print();
    
    std::cout << "\nMatrix B (2x3):" << std::endl;
    B.print();
    
    // Matrix multiplication
    Matrix C = A * B;
    std::cout << "\nA * B (3x3):" << std::endl;
    C.print();
    
    // Transpose
    Matrix At = A.transpose();
    std::cout << "\nA transpose (2x3):" << std::endl;
    At.print();
    
    // Element-wise operations
    Matrix D(3, 3, 1.0f);
    Matrix E = C + D;
    std::cout << "\nC + ones (3x3):" << std::endl;
    E.print();
    
    // Apply activation
    Matrix F = E.apply(MLOps::sigmoid);
    std::cout << "\nSigmoid(C + ones):" << std::endl;
    F.print();
    
    // Softmax
    Matrix logits(1, 5);
    logits.randomize(-2.0f, 2.0f);
    std::cout << "\nLogits:" << std::endl;
    logits.print();
    
    Matrix probs = logits.softmax();
    std::cout << "\nSoftmax probabilities (should sum to 1):" << std::endl;
    probs.print();
    
    // Vector operations
    std::cout << "\n=== Vector Operations ===" << std::endl;
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2 = {4.0f, 5.0f, 6.0f};
    
    float dotProd = MLOps::dot(v1, v2);
    std::cout << "Dot product: " << dotProd << std::endl;
    
    float norm = MLOps::l2Norm(v1);
    std::cout << "L2 norm of v1: " << norm << std::endl;
    
    return 0;
}
