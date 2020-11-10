all: tets-V100 test-T4 test-A100

test-V100: tc_test_numerics-V100.cu
	nvcc -o $@ -arch=sm_70 -std=c++11 $<

test-T4: tc_test_numerics-T4-A100-binary16.cu
	nvcc -o $@ -arch=sm_75 -std=c++11 $<

test-A100: test-A100-binary16 test-A100-bf16 test-A100-binary64 test-A100-tf32

test-A100-binary16: tc_test_numerics-T4-A100-binary16.cu
	nvcc -o $@ -arch=sm_80 -std=c++11 $<

test-A100-%: tc_test_numerics-A100-%.cu
	nvcc -o $@ -arch=sm_80 -std=c++11 $<

clean: clean-V100 clean-T4 clean-A100

clean-V100:
	rm -f test-V100

clean-T4:
	rm -f test-T4

clean-A100:
	rm -f test-A100-%
