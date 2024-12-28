#include "sort_gpu.cuh"
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int32_t input_i[TEST_INPUT_NUM] = {
    -27, -78, 74,  -3,  -63, -70, -89, 96,  -8,  -71, 33,  1,   -44, -75,
    67,  -28, -22, -84, -40, -34, -47, 83,  -49, -5,  60,  -11, -8,  -48,
    -52, 58,  49,  76,  -24, -47, -31, 47,  -84, -86, 97,  30,  60,  41,
    -46, 66,  53,  -37, -28, 99,  -21, 78,  -30, -57, -88, 4,   55,  85,
    41,  79,  97,  -43, 60,  27,  90,  88,  -57, 33,  -60, 90,  44,  7,
    -18, -75, -72, 7,   -25, 34,  59,  -35, -13, 58};

// float output[TEST_INPUT_NUM];
// int32_t indices_ptr_out[TEST_INPUT_NUM];

#if 0
int32_t keys[TEST_INPUT_NUM] = {0};
int32_t bfe_keys[TEST_INPUT_NUM] = {0};
int32_t bfe_keys_out[TEST_INPUT_NUM] = {0};
int32_t offset[TEST_INPUT_NUM] = {0};
// int32_t bucket_offset[BUCKET_SIZE] = {0};
// int32_t curr_count[THREAD_NUM] = {0};
int32_t bucket_offset[BUCKET_SIZE][THREAD_NUM] = {0};
int32_t exclusive_cumsum[BUCKET_SIZE][THREAD_NUM] = {0};
// int32_t exclusive_cumsum[BUCKET_SIZE] = {0};
int32_t indices[2][TEST_INPUT_NUM];
#endif

__global__ void put_numbers_into_bucket(const int32_t *d_keys_in,
                                        int32_t *offset, int32_t *bucket_offset,
                                        int32_t num_items) {
  int32_t block_size =
      (num_items + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);

  for (int32_t i = 0; i < block_size; i++) {
    int32_t idx = i + (blockIdx.x * blockDim.x + threadIdx.x) * block_size;
    if (idx < num_items) {
      int32_t idx0 = d_keys_in[idx] * (blockDim.x * gridDim.x) +
                     blockIdx.x * blockDim.x + threadIdx.x;
      offset[idx] = bucket_offset[idx0];
      bucket_offset[idx0]++;
    }
  }
}

#if 0
void sort_pairs_loop(const int32_t *d_keys_in, int32_t *indices_ptr_in,
                     int32_t *indices_ptr_out, int32_t num_items) {
  put_numbers_into_bucket(d_keys_in, offset, bucket_offset, num_items);
  calc_exclusive_cumsum((int32_t *)bucket_offset, (int32_t *)exclusive_cumsum,
                        BUCKET_SIZE * THREAD_NUM);
  update_indices_ptr(d_keys_in, indices_ptr_in, offset, (int32_t *)exclusive_cumsum,
                     indices_ptr_out, num_items);
}

template <typename KeyT, typename ValueT, bool is_descend>
void sort_pairs(const ValueT *d_values_in, ValueT *d_values_out,
                int32_t *indices_ptr, int32_t num_items) {
  int32_t loop_count = sizeof(KeyT) * 8 / BUCKET_WIDTH;
  prepare_keys<KeyT, ValueT, is_descend>(d_values_in, keys, num_items);
  prepare_indices(indices[0], num_items);
  for (int32_t i = 0; i < loop_count; i++) {
    int32_t begin_bit = (i)*BUCKET_WIDTH;
    extract_keys(keys, bfe_keys, indices[i % 2], num_items, begin_bit,
                 BUCKET_WIDTH);
    sort_pairs_loop(bfe_keys, indices[i % 2], indices[(i + 1) % 2], num_items);
    memset(bucket_offset, 0, sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM);
  }
  post_process(d_values_in, d_values_out, indices[0], indices_ptr, num_items);
}
#endif

#define TEST_INPUT_NUM2 8
void test_sort() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(DType) * TEST_INPUT_NUM);
  DType *output;
  cudaMalloc(&output, sizeof(DType) * TEST_INPUT_NUM);
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

  DType *output_cpu = (DType *)malloc(sizeof(DType) * TEST_INPUT_NUM);
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
  int32_t loop_count = sizeof(KType) * 8 / BUCKET_WIDTH;
  int32_t *indices_ptr[2] = {indices, indices1};
  for (int32_t i = 0; i < loop_count; i++) {
    int32_t begin_bit = (i)*BUCKET_WIDTH;
    cudaMemset(bucket_offset, 0,
               sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
    extract_keys<KType><<<numBlocks, threadsPerBlock>>>(
        keys, bfe_keys, indices_ptr[i % 2], TEST_INPUT_NUM2, begin_bit, BUCKET_WIDTH);
    put_numbers_into_bucket<<<numBlocks, threadsPerBlock>>>(
        bfe_keys, offset, bucket_offset, TEST_INPUT_NUM2);
    calc_exclusive_cumsum<<<numBlocks, threadsPerBlock>>>(
        bucket_offset, exclusive_cumsum, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
    update_indices_ptr<<<numBlocks, threadsPerBlock>>>(
        bfe_keys, indices_ptr[i % 2], offset, exclusive_cumsum,
        indices_ptr[(i + 1) % 2], TEST_INPUT_NUM2);
  }


  cudaMemcpy(offset_cpu, offset, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(bucket_offset_cpu, bucket_offset,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(exclusive_cumsum_cpu, exclusive_cumsum,
             sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM,
             cudaMemcpyDeviceToHost);

  cudaMemcpy(indices1_cpu, indices, sizeof(int32_t) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);

  printf("gpu:\n");
  // print_ivec(offset_cpu, TEST_INPUT_NUM2);
  // print_ivec(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec_sum(bucket_offset_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec(exclusive_cumsum_cpu, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  printf("indices1:\n");
  print_ivec(indices1_cpu, TEST_INPUT_NUM2);

  prepare_keys_cpu<KType, DType, false>(input, keys_cpu2, TEST_INPUT_NUM2);
  printf("keys_cpu2:\n");
  print_ixvec(keys_cpu2, TEST_INPUT_NUM2);
  prepare_indices_cpu(indices_cpu, TEST_INPUT_NUM2);
  int32_t *indices_ptr_cpu[2] = {indices_cpu, indices1_cpu2};
  for (int32_t i = 0; i < loop_count; i++) {
    int32_t begin_bit = (i)*BUCKET_WIDTH;
    memset(bucket_offset_cpu2, 0,
           sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
    extract_keys_cpu<KType>(keys_cpu2, bfe_keys_cpu2, indices_ptr_cpu[i % 2],
                            TEST_INPUT_NUM2, begin_bit, BUCKET_WIDTH);
    put_numbers_into_bucket_cpu(bfe_keys_cpu2, offset_cpu2, bucket_offset_cpu2,
                                TEST_INPUT_NUM2);
    calc_exclusive_cumsum_cpu(bucket_offset_cpu2, exclusive_cumsum_cpu2,
                              BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
    update_indices_ptr_cpu2(bfe_keys_cpu2, indices_ptr_cpu[i % 2], offset_cpu2,
                            exclusive_cumsum_cpu2, indices_ptr_cpu[(i + 1) % 2],
                            TEST_INPUT_NUM2);
    printf("indices:\n");
    print_ivec(indices_ptr_cpu[i % 2], TEST_INPUT_NUM2);
    print_ixvec(bfe_keys_cpu2, TEST_INPUT_NUM2);
    printf("indices1:\n");
    print_ivec(indices_ptr_cpu[(i + 1) % 2], TEST_INPUT_NUM2);
  }
  post_process_cpu(input, output_cpu, indices_ptr_cpu[0], indices_ptr_cpu[0],
                   TEST_INPUT_NUM2);

  printf("cpu:\n");
  print_fvec(input, TEST_INPUT_NUM2);
  // print_ivec(offset_cpu2, TEST_INPUT_NUM2);
  // print_ivec(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec_sum(bucket_offset_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  // print_ivec(exclusive_cumsum_cpu2, BUCKET_SIZE * THREAD_NUM * BLOCK_NUM);
  print_fvec(output_cpu, TEST_INPUT_NUM2);

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

int main() {
  // test_key<<<1, 2>>>();
  test_sort();
  // test_prepare_keys();
  // test_prepare_indices();
  // test_extract_keys();
  // test_put_numbers_into_bucket();
  // test_calc_exclusive_cumsum();
  // test_update_indices_ptr();
  cudaDeviceSynchronize();
  return 0;
}
