all: sort_cpu sort_cpu_mt sort_gpu

sort_gpu: sort_gpu_test.cu sort_gpu.cu sort_gpu_test_case.cu
	nvcc -g -G -o $@ $^

sort_cpu_mt: sort_cpu_mt.cpp
	g++ -g -O0 -o $@ $^

sort_cpu: sort_cpu.cpp
	g++ -g -O0 -o $@ $^

clean:
	rm sort_cpu sort_cpu_mt sort_gpu
