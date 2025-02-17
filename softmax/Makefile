NVCC = nvcc
CFLAGS = -arch=sm_70 -O3
SRC_DIR = kernels
BIN_DIR = bin

TARGET = $(BIN_DIR)/benchmark.out
SRC = benchmark.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)
