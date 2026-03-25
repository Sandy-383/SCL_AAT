# ─── CUDA GWO Shared Library Build ───────────────────────────────────────────
#
# Usage:
#   make            — build shared library
#   make clean      — remove build artifacts
#   make info       — print detected GPU compute capability
#
# Prerequisites:
#   CUDA Toolkit 11.x or 12.x, GCC or MSVC
#
# Detect GPU compute capability automatically via deviceQuery or nvidia-smi:
#   SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
#                | head -1 | tr -d '.')

SM        ?= 75          # Default: RTX 20xx / GTX 16xx
NVCC      := nvcc
CXX       := g++
CXXFLAGS  := -O3 -std=c++17
NVCCFLAGS := -O3 -arch=sm_$(SM) -std=c++17 \
             --compiler-options "-fPIC" \
             -Xcompiler -Wall

SRCS_CU  := gwo_kernel.cu fitness_kernel.cu memory_manager.cu
SRCS_CPP := gwo_host.cpp
LIB_OUT  := libcudagwo.so

# Windows target
ifeq ($(OS),Windows_NT)
    LIB_OUT  := cudagwo.dll
    NVCCFLAGS += --shared
else
    NVCCFLAGS += --shared
endif

OBJS := $(SRCS_CU:.cu=.o) $(SRCS_CPP:.cpp=.o)

.PHONY: all clean info

all: $(LIB_OUT)
	@echo "Build complete: $(LIB_OUT)"

$(LIB_OUT): $(SRCS_CU) $(SRCS_CPP)
	$(NVCC) $(NVCCFLAGS) -lcurand -o $@ $^
	@echo "Shared library ready: $(LIB_OUT)"

clean:
	rm -f $(LIB_OUT) *.o *.exp *.lib

info:
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null \
	    || echo "nvidia-smi not found — set SM manually"
