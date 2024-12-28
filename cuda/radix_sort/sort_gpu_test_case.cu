#include "sort_gpu.cuh"
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void prepare_indices(int32_t *indices, int32_t num_items) {
  int32_t block_size =
      (num_items + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);
  for (int32_t i = 0; i < block_size; i++) {
    int32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      indices[idx] = idx;
    }
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

void test_extract_keys() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(DType) * TEST_INPUT_NUM);
  KType *keys;
  cudaMalloc(&keys, sizeof(KType) * TEST_INPUT_NUM);
  KType *bfe_keys;
  cudaMalloc(&bfe_keys, sizeof(KType) * TEST_INPUT_NUM);
  int32_t *indices;
  cudaMalloc(&indices, sizeof(int32_t) * TEST_INPUT_NUM);
  KType *keys_cpu = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  KType *keys_cpu2 = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  int32_t *bfe_keys_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *bfe_keys_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *indices_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);

  cudaMemcpy(input_gpu, input, sizeof(DType) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(2);
  prepare_keys<KType, DType, false>
      <<<numBlocks, threadsPerBlock>>>(input_gpu, keys, TEST_INPUT_NUM);
  prepare_indices<<<numBlocks, threadsPerBlock>>>(indices, TEST_INPUT_NUM);
  extract_keys<KType><<<numBlocks, threadsPerBlock>>>(keys, bfe_keys, indices,
                                                      TEST_INPUT_NUM, 0, 8);

  cudaMemcpy(bfe_keys_cpu, bfe_keys, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  print_ixvec(bfe_keys_cpu, TEST_INPUT_NUM);

  prepare_keys_cpu<KType, DType, false>(input, keys_cpu2, TEST_INPUT_NUM);
  prepare_indices_cpu(indices_cpu, TEST_INPUT_NUM);
  extract_keys_cpu<KType>(keys_cpu2, bfe_keys_cpu2, indices_cpu, TEST_INPUT_NUM,
                          0, 8);

  printf("cpu:\n");
  print_ixvec(bfe_keys_cpu2, TEST_INPUT_NUM);

  cudaFree(input_gpu);
  cudaFree(keys);
  cudaFree(bfe_keys);
  cudaFree(indices);

  free(keys_cpu);
  free(keys_cpu2);
  free(bfe_keys_cpu);
  free(bfe_keys_cpu2);
  free(indices_cpu);
}

void test_put_numbers_into_bucket() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(DType) * TEST_INPUT_NUM);
  KType *keys;
  cudaMalloc(&keys, sizeof(KType) * TEST_INPUT_NUM);
  KType *bfe_keys;
  cudaMalloc(&bfe_keys, sizeof(KType) * TEST_INPUT_NUM);
  int32_t *indices;
  cudaMalloc(&indices, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset;
  cudaMalloc(&offset, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *bucket_offset;
  cudaMalloc(&bucket_offset,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // int32_t *exclusive_cumsum;
  // cudaMalloc(&exclusive_cumsum, sizeof(int32_t) * THREAD_NUM * BLOCK_NUM);

  // KType *keys_cpu = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  KType *keys_cpu2 = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  // int32_t *bfe_keys_cpu = (int32_t *)malloc(sizeof(int32_t) *
  // TEST_INPUT_NUM);
  int32_t *bfe_keys_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *indices_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *bucket_offset_cpu =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *bucket_offset_cpu2 =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // int32_t *exclusive_cumsum_cpu = (int32_t *)malloc(sizeof(int32_t) *
  // THREAD_NUM * BLOCK_NUM);

  cudaMemcpy(input_gpu, input, sizeof(DType) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(BLOCK_NUM);
  prepare_keys<KType, DType, false>
      <<<numBlocks, threadsPerBlock>>>(input_gpu, keys, TEST_INPUT_NUM);
  prepare_indices<<<numBlocks, threadsPerBlock>>>(indices, TEST_INPUT_NUM);
  extract_keys<KType><<<numBlocks, threadsPerBlock>>>(keys, bfe_keys, indices,
                                                      TEST_INPUT_NUM, 0, 8);
  cudaMemset(bucket_offset, 0,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  put_numbers_into_bucket<<<numBlocks, threadsPerBlock>>>(
      bfe_keys, offset, bucket_offset, TEST_INPUT_NUM);

  cudaMemcpy(offset_cpu, offset, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(bucket_offset_cpu, bucket_offset,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  print_ivec(offset_cpu, TEST_INPUT_NUM);
  print_ivec(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  print_ivec_sum(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);

  prepare_keys_cpu<KType, DType, false>(input, keys_cpu2, TEST_INPUT_NUM);
  prepare_indices_cpu(indices_cpu, TEST_INPUT_NUM);
  extract_keys_cpu<KType>(keys_cpu2, bfe_keys_cpu2, indices_cpu, TEST_INPUT_NUM,
                          0, 8);
  put_numbers_into_bucket_cpu(bfe_keys_cpu2, offset_cpu2, bucket_offset_cpu2,
                              TEST_INPUT_NUM);

  printf("cpu:\n");
  print_ivec(offset_cpu2, TEST_INPUT_NUM);
  print_ivec(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  print_ivec_sum(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);

  cudaFree(input_gpu);
  cudaFree(keys);
  cudaFree(bfe_keys);
  cudaFree(indices);
  cudaFree(offset);
  cudaFree(bucket_offset);

  // free(keys_cpu);
  free(keys_cpu2);
  // free(bfe_keys_cpu);
  free(bfe_keys_cpu2);
  free(indices_cpu);
  free(offset_cpu);
  free(offset_cpu2);
  free(bucket_offset_cpu);
  free(bucket_offset_cpu2);
}

#define TEST_INPUT_NUM2 8
void test_calc_exclusive_cumsum() {
  int32_t *input_gpu;
  cudaMalloc(&input_gpu, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *output_gpu;
  cudaMalloc(&output_gpu, sizeof(int32_t) * TEST_INPUT_NUM);

  int32_t *output_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *output_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *output_cpu3 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);

  cudaMemcpy(input_gpu, input_i, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(BLOCK_NUM);

  calc_exclusive_cumsum<<<numBlocks, threadsPerBlock>>>(input_gpu, output_gpu,
                                                        TEST_INPUT_NUM2);

  cudaMemcpy(output_cpu3, output_gpu, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  calc_exclusive_cumsum_cpu(input_i, output_cpu, TEST_INPUT_NUM2);
  calc_exclusive_cumsum_cpu2(input_i, output_cpu2, TEST_INPUT_NUM2);

  printf("input:\n");
  print_ivec(input_i, TEST_INPUT_NUM2);
  printf("gpu:\n");
  print_ivec(output_cpu3, TEST_INPUT_NUM2);
  printf("cpu:\n");
  print_ivec(output_cpu, TEST_INPUT_NUM2);
  printf("cpu2:\n");
  print_ivec(output_cpu2, TEST_INPUT_NUM2);

  free(output_cpu);
  free(output_cpu2);
  free(output_cpu3);
}

void test_update_indices_ptr() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(DType) * TEST_INPUT_NUM);
  KType *keys;
  cudaMalloc(&keys, sizeof(KType) * TEST_INPUT_NUM);
  KType *bfe_keys;
  cudaMalloc(&bfe_keys, sizeof(KType) * TEST_INPUT_NUM);
  int32_t *indices;
  cudaMalloc(&indices, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset;
  cudaMalloc(&offset, sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *bucket_offset;
  cudaMalloc(&bucket_offset,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *exclusive_cumsum;
  cudaMalloc(&exclusive_cumsum,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *indices1;
  cudaMalloc(&indices1, sizeof(int32_t) * TEST_INPUT_NUM);

  // KType *keys_cpu = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  KType *keys_cpu2 = (KType *)malloc(sizeof(KType) * TEST_INPUT_NUM);
  // int32_t *bfe_keys_cpu = (int32_t *)malloc(sizeof(int32_t) *
  // TEST_INPUT_NUM);
  int32_t *bfe_keys_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *indices_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *indices1_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *indices1_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset_cpu = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *offset_cpu2 = (int32_t *)malloc(sizeof(int32_t) * TEST_INPUT_NUM);
  int32_t *bucket_offset_cpu =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *bucket_offset_cpu2 =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *exclusive_cumsum_cpu =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  int32_t *exclusive_cumsum_cpu2 =
      (int32_t *)malloc(sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);

  cudaMemcpy(input_gpu, input, sizeof(DType) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREAD_NUM);
  dim3 numBlocks(BLOCK_NUM);
  prepare_keys<KType, DType, false>
      <<<numBlocks, threadsPerBlock>>>(input_gpu, keys, TEST_INPUT_NUM2);
  prepare_indices<<<numBlocks, threadsPerBlock>>>(indices, TEST_INPUT_NUM2);
  extract_keys<KType><<<numBlocks, threadsPerBlock>>>(keys, bfe_keys, indices,
                                                      TEST_INPUT_NUM2, 0, 8);
  cudaMemset(bucket_offset, 0,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  put_numbers_into_bucket<<<numBlocks, threadsPerBlock>>>(
      bfe_keys, offset, bucket_offset, TEST_INPUT_NUM2);

  calc_exclusive_cumsum<<<numBlocks, threadsPerBlock>>>(
      bucket_offset, exclusive_cumsum, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);

  update_indices_ptr<<<numBlocks, threadsPerBlock>>>(
      bfe_keys, indices, offset, exclusive_cumsum, indices1, TEST_INPUT_NUM2);

  cudaMemcpy(offset_cpu, offset, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(bucket_offset_cpu, bucket_offset,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(exclusive_cumsum_cpu, exclusive_cumsum,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(indices1_cpu, indices1, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  // print_ivec(offset_cpu, TEST_INPUT_NUM2);
  // print_ivec(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec_sum(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec(exclusive_cumsum_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  printf("indices1:\n");
  print_ivec(indices1_cpu, TEST_INPUT_NUM2);

  prepare_keys_cpu<KType, DType, false>(input, keys_cpu2, TEST_INPUT_NUM2);
  prepare_indices_cpu(indices_cpu, TEST_INPUT_NUM2);
  extract_keys_cpu<KType>(keys_cpu2, bfe_keys_cpu2, indices_cpu,
                          TEST_INPUT_NUM2, 0, 8);
  memset(bucket_offset_cpu2, 0,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  put_numbers_into_bucket_cpu(bfe_keys_cpu2, offset_cpu2, bucket_offset_cpu2,
                              TEST_INPUT_NUM2);
  calc_exclusive_cumsum_cpu(bucket_offset_cpu2, exclusive_cumsum_cpu2,
                            BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  update_indices_ptr_cpu2(bfe_keys_cpu2, indices_cpu, offset_cpu2,
                          exclusive_cumsum_cpu2, indices1_cpu2,
                          TEST_INPUT_NUM2);
  printf("cpu:\n");
  // print_ivec(offset_cpu2, TEST_INPUT_NUM2);
  // print_ivec(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec_sum(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec(exclusive_cumsum_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  printf("indices:\n");
  print_ivec(indices_cpu, TEST_INPUT_NUM2);
  print_ivec(bfe_keys_cpu2, TEST_INPUT_NUM2);
  printf("indices1:\n");
  print_ivec(indices1_cpu2, TEST_INPUT_NUM2);

  cudaFree(input_gpu);
  cudaFree(keys);
  cudaFree(bfe_keys);
  cudaFree(indices);
  cudaFree(offset);
  cudaFree(bucket_offset);
  cudaFree(exclusive_cumsum);
  cudaFree(indices1);

  // free(keys_cpu);
  free(keys_cpu2);
  // free(bfe_keys_cpu);
  free(bfe_keys_cpu2);
  free(indices_cpu);
  free(offset_cpu);
  free(offset_cpu2);
  free(bucket_offset_cpu);
  free(bucket_offset_cpu2);
  free(exclusive_cumsum_cpu);
  free(exclusive_cumsum_cpu2);
}
