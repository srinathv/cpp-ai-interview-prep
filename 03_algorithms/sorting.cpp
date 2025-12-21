#include <iostream>
#include <vector>
#include <algorithm>

// Bubble Sort - O(n^2)
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Quick Sort - O(n log n) average
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// Merge Sort - O(n log n)
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> leftArr(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> rightArr(arr.begin() + mid + 1, arr.begin() + right + 1);
    
    int i = 0, j = 0, k = left;
    while (i < leftArr.size() && j < rightArr.size()) {
        arr[k++] = (leftArr[i] <= rightArr[j]) ? leftArr[i++] : rightArr[j++];
    }
    while (i < leftArr.size()) arr[k++] = leftArr[i++];
    while (j < rightArr.size()) arr[k++] = rightArr[j++];
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Heap Sort - O(n log n)
void heapify(std::vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = n / 2 - 1; i >= 0; --i)
        heapify(arr, n, i);
    for (int i = n - 1; i > 0; --i) {
        std::swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

void printArray(const std::vector<int>& arr) {
    for (int num : arr) std::cout << num << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> arr1 = {64, 34, 25, 12, 22, 11, 90};
    std::cout << "Original: "; printArray(arr1);
    bubbleSort(arr1);
    std::cout << "Bubble Sort: "; printArray(arr1);
    
    std::vector<int> arr2 = {64, 34, 25, 12, 22, 11, 90};
    quickSort(arr2, 0, arr2.size() - 1);
    std::cout << "Quick Sort: "; printArray(arr2);
    
    std::vector<int> arr3 = {64, 34, 25, 12, 22, 11, 90};
    mergeSort(arr3, 0, arr3.size() - 1);
    std::cout << "Merge Sort: "; printArray(arr3);
    
    std::vector<int> arr4 = {64, 34, 25, 12, 22, 11, 90};
    heapSort(arr4);
    std::cout << "Heap Sort: "; printArray(arr4);
    
    return 0;
}
