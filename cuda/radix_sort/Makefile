all: sort_cpu sort_cpu_mt

sort_cpu_mt: sort_cpu_mt.cpp
	clang -g -O0 -o $@ $^

sort_cpu: sort_cpu.cpp
	clang -g -O0 -o $@ $^

clean:
	rm sort_cpu
