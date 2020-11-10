# Numerical Behavior of the NVIDIA Tensor Cores

The aim of this test suite is to probe the numerical behavior of the tensor cores that equip some recent NVIDIA graphic cards. The tests in the suite are divided into 4 sections, and show the following features.

A. Support for subnormal numbers
* Tensor cores accept binary16 subnormals in input in binary16 mode
* Tensor cores accept binary16 subnormals in input in binary32 mode
* Tensor cores accept binary32 subnormals in input and return them
* Tensor cores can return binary16 subnormals in binary16 mode
* Tensor cores can return binary16 subnormals in binary32 mode

B. Accuracy of the dot products
* Tensor cores compute the products of two binary16 numbers exactly
* Tensor cores compute the products of two binary16 numbers exactly (binary16 mode)
* Tensor cores accumulate sums in binary32 arithemetic
* Tensor cores accumulate partial sums on the largest element in absolute value

C. Rounding modes in tensor core computations
* Tensor cores use round-down for postive values
* Tensor cores use round-up for negative values
* Tensor cores round the accumulator using round-to-nearest (binary16 mode)

D. Features of the accumulator
1) Tensor cores do not implement guard digits (extra bits on the right)
2) Tensor cores do not normalize by shifting right (in sums of elements with same sign)
3) Tensor cores do not normllize by shifting left (in sums of elements with opposite sign)
4) Tensor cores implement two carry-out digits (extra bits on the left)
5) The product of tensor cores is not monotonic

### Compiling and running the suite
The experiments can be compiled by issuing `make all`, which generates several executable files:
* `test-V100`, for testing Volta GPUs (requires version 9 or newer of the CUDA platform);
* `test-T4`, for testing Turing GPUs (requires version 10 or newer of the CUDA platform);
* `test-A100-binary16`, `test-A100-bf16`, `test-A100-tf32`, `test-A100-binary64`, for testing the four precision configurations available on Ampere GPUs (requires version 11 or newer of the CUDA platform).

### Reference
If you use the code in this repository, please reference the preprint:

Massimiliano Fasi,  Nicholas J. Higham, Mantas Mikaitis, and Srikara Pranesh. [_Numerical Behavior of the NVIDIA Tensor Cores._](http://eprints.maths.manchester.ac.uk/2784/)Technical Report 2020.10, Manchester Institute for Mathematical Sciences, The University of Manchester, UK, April 2020. *Revised October 2020*.

### License
This software is distributed under the terms of the GNU GPL v.2 software license (see [LICENSE.md](./LICENSE.md)).
