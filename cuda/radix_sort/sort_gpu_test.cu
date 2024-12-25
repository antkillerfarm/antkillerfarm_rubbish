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
  printf("\n");
}

void print_ivec_sum(int32_t *vec, int32_t num_items) {
  int32_t sum = 0;
  for (int32_t i = 0; i < num_items; i++) {
    sum += vec[i];
  }
  printf("sum: %d\n", sum);
}

void print_ixvec(int32_t *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%x, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
  printf("\n");
}

void print_fvec(float *vec, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    printf("%f, ", vec[i]);
    if (i % 10 == 9) {
      printf("\n");
    }
  }
  printf("\n");
}

__global__ void test_key() {
  auto key = Float_Point_Number<float>::GetKeyForRadixSort<false>(20000.5);
  auto bfe = Unsigned_Bits<int32_t>::BitfieldExtract(key, 16, 8);
  printf("A 0x%x\n", key);
  printf("B 0x%x\n", bfe);
}

template <typename KeyT, typename ValueT, bool is_descend>
__global__ void prepare_keys(const ValueT *d_values_in, KeyT *d_keys_in,
                             int32_t num_items) {
  int block_size =
      (num_items + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);

  for (int32_t i = 0; i < block_size; i++) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      d_keys_in[idx] =
          Float_Point_Number<ValueT>::template GetKeyForRadixSort<is_descend>(
              d_values_in[idx]);
    }
  }
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

__global__ void prepare_indices(int32_t *indices, int32_t num_items) {
  int block_size =
      (num_items + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);
  for (int32_t i = 0; i < block_size; i++) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      indices[idx] = idx;
    }
  }
}

void prepare_indices_cpu(int32_t *indices, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    indices[i] = i;
  }
}

template <typename KeyT>
void extract_keys_cpu(KeyT *d_keys_in, KeyT *d_keys_out, int32_t *indices,
                      int32_t num_items, int32_t bit_start, int32_t num_bits) {
  for (int32_t i = 0; i < num_items; i++) {
    d_keys_out[i] = Unsigned_Bits<KeyT>::BitfieldExtract(d_keys_in[indices[i]],
                                                         bit_start, num_bits);
  }
}

#if 0
void sort_pairs_loop(const int32_t *d_keys_in, int32_t *indices_ptr_in,
                     int32_t *indices_ptr_out, int32_t num_items) {
  int32_t num_items_per_thread = num_items / THREAD_NUM;
  for (int32_t i = 0; i < THREAD_NUM; i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx = j + i * num_items_per_thread;
      offset[idx] = bucket_offset[d_keys_in[idx]][i];
      bucket_offset[d_keys_in[idx]][i]++;
    }
  }
  calc_exclusive_cumsum((int32_t *)bucket_offset, (int32_t *)exclusive_cumsum,
                        BUCKET_SIZE * THREAD_NUM);
  update_indices_ptr(d_keys_in, indices_ptr_in, offset, (int32_t *)exclusive_cumsum,
                     indices_ptr_out, num_items);
}
#endif

void put_numbers_into_bucket_cpu(const int32_t *d_keys_in, int32_t *offset,
                                 int32_t *bucket_offset, int32_t num_items) {
  int32_t num_items_per_thread = num_items / (BLOCK_NUM * THREAD_NUM);
  for (int32_t i = 0; i < (BLOCK_NUM * THREAD_NUM); i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx = j + i * num_items_per_thread;
      if (idx < num_items) {
        offset[idx] =
            bucket_offset[d_keys_in[idx] * (BLOCK_NUM * THREAD_NUM) + i];
        bucket_offset[d_keys_in[idx] * (BLOCK_NUM * THREAD_NUM) + i]++;
      }
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
