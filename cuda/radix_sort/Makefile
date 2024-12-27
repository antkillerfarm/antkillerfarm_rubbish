all: sort_cpu sort_cpu_mt sort_gpu

sort_gpu: sort_gpu_test.cu sort_gpu.cu sort_gpu_test_case.cu
	/usr/local/cuda/bin/nvcc -g -G -o $@ $^

sort_cpu_mt: sort_cpu_mt.cpp
	clang -g -O0 -o $@ $^

sort_cpu: sort_cpu.cpp
	clang -g -O0 -o $@ $^

clean:
	rm sort_cpu
