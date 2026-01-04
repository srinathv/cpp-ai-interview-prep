CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

TARGET = comparison
SRC = comparison.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
