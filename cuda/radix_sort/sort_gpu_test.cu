#include "sort_gpu.cuh"
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_ivec(int32_t *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%d, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
}

void print_ixvec(int32_t *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%x, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
}

void print_fvec(float *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%f, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
}

__global__ void test_key() {
  auto key = Float_Point_Number<float>::GetKeyForRadixSort<false>(20000.5);
  auto bfe = Unsigned_Bits<int32_t>::BitfieldExtract(key, 16, 8);
  printf("A 0x%x\n", key);
  printf("B 0x%x\n", bfe);
}

template <typename KeyT, typename ValueT, bool is_descend>
void prepare_keys_cpu(const ValueT *d_values_in, KeyT *d_keys_in,
                      int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    d_keys_in[i] =
        Float_Point_Number<ValueT>::template GetKeyForRadixSort<is_descend>(
            d_values_in[i]);
  }
}

void prepare_indices_cpu(int32_t *indices, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    indices[i] = i;
  }
}

void test_prepare_keys() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(DType) * TEST_INPUT_NUM);
  KType *keys;
  cudaMalloc(&keys, sizeof(KType) * TEST_INPUT_NUM);
  KType *keys_cpu = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  KType *keys_cpu2 = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);

  cudaMemcpy(input_gpu, input, sizeof(DType) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(2);
  prepare_keys<KType, DType, false>
      <<<numBlocks, threadsPerBlock>>>(input_gpu, keys, TEST_INPUT_NUM);

  cudaMemcpy(keys_cpu, keys, sizeof(KType) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  print_ivec(keys_cpu, TEST_INPUT_NUM);

  prepare_keys_cpu<KType, DType, false>(input, keys_cpu2, TEST_INPUT_NUM);
  printf("cpu:\n");
  print_ivec(keys_cpu2, TEST_INPUT_NUM);

  cudaFree(input_gpu);
  cudaFree(keys);
  free(keys_cpu);
}

extern __global__ void prepare_indices(int32_t *indices, int32_t num_items);

void test_prepare_indices() {
  int32_t *input_gpu;
  cudaMalloc(&input_gpu, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *input_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(2);
  prepare_indices<<<numBlocks, threadsPerBlock>>>(input_gpu, TEST_INPUT_NUM);

  cudaMemcpy(input_cpu, input_gpu, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  print_ivec(input_cpu, TEST_INPUT_NUM);

  cudaFree(input_gpu);
  free(input_cpu);
}