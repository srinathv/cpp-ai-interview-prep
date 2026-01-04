NVCC = nvcc
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70

TARGET = comparison
SRC = comparison.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
