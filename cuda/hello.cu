#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "iostream"

__global__ void sayHelloWorld();

int main(){

    printf("HelloWorld! CPU \n");
    sayHelloWorld<<<1, 10>>>();   //调用GPU上执行的函数，调用10个GPU线程

    cudaDeviceReset();    //显式地释放和清空当前进程中与当前设备有关的所有资源，不加这句不会打印GPU中的输出语句"HelloWorld! GPU"
}

__global__ void sayHelloWorld(){
    printf("HelloWorld! GPU \n");
    //cout << "HelloWorld! GPU" << endl;     //不能使用cout, std命名不能使用到GPU上
}
