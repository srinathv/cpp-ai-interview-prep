CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

TARGET = comparison_omp
SRC = comparison_omp.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
