#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// Custom vector implementation from scratch
template<typename T>
class MyVector {
private:
    T* data;
    size_t capacity;
    size_t size;
    
    void resize() {
        capacity = capacity == 0 ? 1 : capacity * 2;
        T* new_data = new T[capacity];
        for (size_t i = 0; i < size; ++i) {
            new_data[i] = data[i];
        }
        delete[] data;
        data = new_data;
    }
    
public:
    MyVector() : data(nullptr), capacity(0), size(0) {}
    
    ~MyVector() {
        delete[] data;
    }
    
    void push_back(const T& value) {
        if (size >= capacity) {
            resize();
        }
        data[size++] = value;
    }
    
    T& operator[](size_t index) {
        return data[index];
    }
    
    size_t get_size() const { return size; }
    size_t get_capacity() const { return capacity; }
};

// Common vector operations
void demonstrate_vector_operations() {
    std::cout << "=== Vector Operations ===" << std::endl;
    
    // 1. Initialization
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2(10, 0);  // 10 elements initialized to 0
    
    // 2. Adding elements
    v1.push_back(6);
    v1.insert(v1.begin() + 2, 99);  // Insert 99 at index 2
    
    // 3. Removing elements
    v1.pop_back();
    v1.erase(v1.begin() + 2);  // Remove element at index 2
    
    // 4. Accessing elements
    std::cout << "First element: " << v1.front() << std::endl;
    std::cout << "Last element: " << v1.back() << std::endl;
    std::cout << "Element at index 2: " << v1[2] << std::endl;
    
    // 5. Size and capacity
    std::cout << "Size: " << v1.size() << ", Capacity: " << v1.capacity() << std::endl;
    
    // 6. Iterating
    std::cout << "Elements: ";
    for (const auto& elem : v1) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    // 7. Algorithms
    std::sort(v1.begin(), v1.end());
    std::reverse(v1.begin(), v1.end());
    
    int sum = std::accumulate(v1.begin(), v1.end(), 0);
    std::cout << "Sum: " << sum << std::endl;
    
    auto it = std::find(v1.begin(), v1.end(), 3);
    if (it != v1.end()) {
        std::cout << "Found 3 at index: " << std::distance(v1.begin(), it) << std::endl;
    }
}

// Interview question: Remove duplicates from sorted vector
int removeDuplicates(std::vector<int>& nums) {
    if (nums.empty()) return 0;
    
    int write_idx = 1;
    for (size_t i = 1; i < nums.size(); ++i) {
        if (nums[i] != nums[i-1]) {
            nums[write_idx++] = nums[i];
        }
    }
    return write_idx;
}

// Interview question: Rotate array right by k positions
void rotateArray(std::vector<int>& nums, int k) {
    int n = nums.size();
    k = k % n;
    std::reverse(nums.begin(), nums.end());
    std::reverse(nums.begin(), nums.begin() + k);
    std::reverse(nums.begin() + k, nums.end());
}

int main() {
    demonstrate_vector_operations();
    
    std::cout << "\n=== Custom Vector Implementation ===" << std::endl;
    MyVector<int> myVec;
    for (int i = 0; i < 10; ++i) {
        myVec.push_back(i);
        std::cout << "Size: " << myVec.get_size() 
                  << ", Capacity: " << myVec.get_capacity() << std::endl;
    }
    
    std::cout << "\n=== Interview Questions ===" << std::endl;
    
    // Remove duplicates
    std::vector<int> v1 = {1, 1, 2, 2, 2, 3, 4, 4, 5};
    int newLen = removeDuplicates(v1);
    std::cout << "After removing duplicates: ";
    for (int i = 0; i < newLen; ++i) {
        std::cout << v1[i] << " ";
    }
    std::cout << std::endl;
    
    // Rotate array
    std::vector<int> v2 = {1, 2, 3, 4, 5, 6, 7};
    rotateArray(v2, 3);
    std::cout << "After rotating by 3: ";
    for (int num : v2) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
