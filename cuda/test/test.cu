#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TEST_INPUT_NUM 80

float input[TEST_INPUT_NUM] = {
    -55.795, -42.349, 79.255,  5.941,   96.018,  31.294,  -96.905, 53.291,
    -90.021, -11.393, 57.446,  41.810,  25.299,  98.622,  37.640,  -11.657,
    55.496,  45.014,  97.440,  -65.244, 20.372,  -40.049, 57.645,  -16.184,
    -95.877, -94.809, 64.730,  -60.664, -44.394, 51.183,  48.071,  -84.458,
    30.944,  -42.142, 7.816,   -22.595, -89.318, -6.443,  42.550,  81.024,
    53.816,  80.649,  76.768,  71.688,  -25.017, -36.421, 37.996,  70.614,
    -73.219, 38.254,  -58.320, -4.735,  -54.836, -94.747, -36.710, 98.710,
    79.415,  -13.906, -61.225, -24.768, -96.095, -88.033, 12.208,  36.603,
    80.426,  -15.617, -28.778, 35.831,  42.338,  -19.250, 89.067,  -44.727,
    80.168,  -77.544, -60.959, -78.450, -74.932, 20.722,  -22.494, 55.845};

float output[TEST_INPUT_NUM];

void transform_dims(int32_t *input, int32_t *output, int32_t num_items,
                    int32_t outer_size, int32_t dim_size, int32_t inter_size) {
  for (int32_t idx = 0; idx < num_items; idx++) {
    int32_t k = idx % inter_size;
    int32_t rest = idx / inter_size;
    int32_t j = rest % dim_size;
    int32_t i = rest / dim_size;
    int32_t idx0 = i * (dim_size * inter_size) + k * dim_size + j;
    // printf("KK: %d : %d : %d : %d ; %d\n", idx, i, j, k, idx0);
    output[idx0] = input[idx];
  }
}

void reverse_transform_dims(int32_t *input, int32_t *output, int32_t num_items,
                            int32_t outer_size, int32_t dim_size,
                            int32_t inter_size) {
  for (int32_t idx = 0; idx < num_items; idx++) {
    int32_t k = idx % inter_size;
    int32_t rest = idx / inter_size;
    int32_t j = rest % dim_size;
    int32_t i = rest / dim_size;
    int32_t idx0 = i * (dim_size * inter_size) + k * dim_size + j;
    // printf("KK: %d : %d : %d : %d ; %d\n", idx, i, j, k, idx0);
    output[idx] = input[idx0];
  }
}

__global__ void test_inc_kernel(float *input_gpu, float *output_gpu,
                                int32_t num_items) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) {
    output_gpu[idx] = input_gpu[idx] + 1.0f;
  }
}

__global__ void test_inc_kernel2(float *input_gpu, float *output_gpu,
                                 int32_t num_items) {
  int32_t block_size =
      (num_items + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);
  for (int32_t i = 0; i < block_size; i++) {
    int32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      output_gpu[idx] = input_gpu[idx] + 1.0f;
    }
  }
}

void print_fvec(float *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%f, ", (float)(vec[i]));
    if (i % 10 == 9) {
      printf("\n");
    }
  }
}

void print_ivec(int32_t *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%d, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
  printf("\n");
}

void test_inc() {
  float *input_gpu;
  cudaMalloc(&input_gpu, sizeof(float) * TEST_INPUT_NUM);
  float *output_gpu;
  cudaMalloc(&output_gpu, sizeof(float) * TEST_INPUT_NUM);
  cudaMemcpy(input_gpu, input, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16);
  dim3 numBlocks((TEST_INPUT_NUM + threadsPerBlock.x - 1) / threadsPerBlock.x);
  printf("numBlocks: %d, threadsPerBlock: %d\n", numBlocks.x,
         threadsPerBlock.x);
  test_inc_kernel<<<numBlocks, threadsPerBlock>>>(input_gpu, output_gpu,
                                                  TEST_INPUT_NUM);

  cudaMemcpy(output, output_gpu, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  print_fvec(output, TEST_INPUT_NUM);

  cudaFree(input_gpu);
  cudaFree(output_gpu);
}

void test_inc2() {
  float *input_gpu;
  cudaMalloc(&input_gpu, sizeof(float) * TEST_INPUT_NUM);
  float *output_gpu;
  cudaMalloc(&output_gpu, sizeof(float) * TEST_INPUT_NUM);
  cudaMemcpy(input_gpu, input, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(2);
  dim3 numBlocks(4);
  printf("numBlocks: %d, threadsPerBlock: %d\n", numBlocks.x,
         threadsPerBlock.x);
  test_inc_kernel2<<<numBlocks, threadsPerBlock>>>(input_gpu, output_gpu, 79);

  cudaMemcpy(output, output_gpu, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  print_fvec(output, TEST_INPUT_NUM);

  cudaFree(input_gpu);
  cudaFree(output_gpu);
}

// tensor([[[17,  8,  1, 18],
//          [10,  3,  2, 18],
//          [ 3,  6, 11,  6]],

//         [[15, 15, 18,  4],
//          [10,  7,  9, 15],
//          [ 9, 14,  1, 13]]])

#define TEST_INPUT_NUM2 (2 * 3 * 4)

int32_t input_dims[TEST_INPUT_NUM2] = {17, 8, 1,  18, 10, 3,  2,  18,
                                       3,  6, 11, 6,  15, 15, 18, 4,
                                       10, 7, 9,  15, 9,  14, 1,  13};
int32_t output_dims[TEST_INPUT_NUM2];

void test_transform_dims() {
  transform_dims(input_dims, output_dims, TEST_INPUT_NUM2, 2, 3, 4);
  print_ivec(output_dims, TEST_INPUT_NUM2);
  // 17, 10, 3, 8, 3, 6, 1, 2, 11, 18,
  // 18, 6, 15, 10, 9, 15, 7, 14, 18, 9,
  // 1, 4, 15, 13,
  transform_dims(input_dims, output_dims, TEST_INPUT_NUM2, 1, 2, 12);
  print_ivec(output_dims, TEST_INPUT_NUM2);
  // 17, 15, 8, 15, 1, 18, 18, 4, 10, 10,
  // 3, 7, 2, 9, 18, 15, 3, 9, 6, 14,
  // 11, 1, 6, 13,
  reverse_transform_dims(output_dims, input_dims, TEST_INPUT_NUM2, 1, 2, 12);
  print_ivec(input_dims, TEST_INPUT_NUM2);
}

int main() {

  // test_inc();
  // test_inc2();
  test_transform_dims();
  cudaDeviceSynchronize();
}