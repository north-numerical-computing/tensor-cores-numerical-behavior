ARCH ?= sm_70

all: tc_test_numerics.cu
	nvcc -o tc_test_numerics -arch=$(ARCH) -std=c++11 tc_test_numerics.cu

clean:
	rm -f tc_test_numerics
