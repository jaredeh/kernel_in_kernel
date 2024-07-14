# Define the compiler
NVCC = nvcc
GCC = gcc

# Specify the target executables
TARGET1 = fakelinux
TARGET2 = cudalinux

# Specify the source files
SOURCE1 = fakelinux.c
SOURCE2 = cudalinux.cu

# Specify the compiler flags
NVCC_FLAGS = -O2 -arch=sm_75
GCC_FLAGS  =
# -O2 is too fast for emulating the kernel
# behavior likely because my fake thing isn't
# as complex as the real thing in terms of what checks
# it does.

# Build rules
all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SOURCE1)
	$(GCC) $(GCC_FLAGS) -o $(TARGET1) $(SOURCE1)

$(TARGET2): $(SOURCE2)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET2) $(SOURCE2) -lcuda

# Clean rule
clean:
	rm -f $(TARGET1) $(TARGET2)
