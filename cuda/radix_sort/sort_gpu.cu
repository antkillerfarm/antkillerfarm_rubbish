#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sort_gpu.cuh"

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
int32_t indices_ptr_out[TEST_INPUT_NUM];

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

__global__ void prepare_indices(int32_t *indices, int32_t num_items) {
  int block_size =
      num_items + (blockDim.x * gridDim.x) - 1 / (blockDim.x * gridDim.x);
  for (int32_t i = 0; i < block_size; i++) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      indices[idx] = idx;
    }
  }
}

#if 0
template <typename KeyT>
void extract_keys(KeyT *d_keys_in, KeyT *d_keys_out, int32_t *indices,
                  int32_t num_items, int32_t bit_start, int32_t num_bits) {
  for (int32_t i = 0; i < num_items; i++) {
    d_keys_out[i] = Unsigned_Bits<KeyT>::BitfieldExtract(d_keys_in[indices[i]],
                                                         bit_start, num_bits);
  }
}

void calc_exclusive_cumsum(const int32_t *value_in, int32_t *exclusive_cumsum,
                           int32_t num_items) {
  int32_t sum;
  for (int32_t i = 0; i < num_items; i++) {
    if (i == 0) {
      sum = 0;
    } else {
      sum += value_in[i - 1];
    }
    exclusive_cumsum[i] = sum;
  }
}

void update_indices_ptr(const int32_t *d_keys_in, const int32_t *indices_ptr_in,
                        const int32_t *offset, const int32_t *exclusive_cumsum,
                        int32_t *indices_ptr_out, int32_t num_items) {
  int32_t num_items_per_thread = num_items / THREAD_NUM;
  for (int32_t i = 0; i < THREAD_NUM; i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx0 = j + i * num_items_per_thread;
      int32_t idx = offset[idx0] + exclusive_cumsum[d_keys_in[idx0] * THREAD_NUM + i];
      indices_ptr_out[idx] = indices_ptr_in[idx0];
      bfe_keys_out[idx] = bfe_keys[idx0];
    }
  }
}

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

template <typename ValueT>
void post_process(const ValueT *d_values_in, ValueT *d_values_out,
                  int32_t *indices_ptr, int32_t *indices_ptr_out,
                  int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    indices_ptr_out[i] = indices_ptr[i];
    d_values_out[i] = d_values_in[indices_ptr[i]];
  }
}

template <typename KeyT, typename ValueT, bool is_descend>
void sort_pairs(const ValueT *d_values_in, ValueT *d_values_out,
                int32_t *indices_ptr, int32_t num_items) {
  int32_t loop_count = sizeof(KeyT) * 8 / BUCKET_WIDTH;
  KeyT *keys;
  cudaMalloc(&keys, sizeof(KeyT) * num_items);
  KeyT *bfe_keys;
  cudaMalloc(&bfe_keys, sizeof(KeyT) * num_items);
  int32_t *offset;
  cudaMalloc(&offset, sizeof(int32_t) * num_items);
  int32_t *bucket_offset;
  cudaMalloc(&offset, sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM);
  int32_t *exclusive_cumsum;
  cudaMalloc(&exclusive_cumsum, sizeof(int32_t) * BUCKET_SIZE * THREAD_NUM);
  int32_t* indices_ptr_backup;
  cudaMalloc(&indices_ptr_backup, sizeof(int32_t) * TEST_INPUT_NUM);

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

  cudaFree(keys);
  cudaFree(bfe_keys);
  cudaFree(offset);
  cudaFree(bucket_offset);
  cudaFree(exclusive_cumsum);
  cudaFree(indices_ptr_backup);
}
#endif

void test_sort() {
  DType *input_gpu;
  cudaMalloc(&input_gpu, sizeof(float) * TEST_INPUT_NUM);
  DType *output_gpu;
  cudaMalloc(&output_gpu, sizeof(float) * TEST_INPUT_NUM);
  int32_t *indices_ptr_out_gpu;
  cudaMalloc(&indices_ptr_out_gpu, sizeof(int32_t) * TEST_INPUT_NUM);

  cudaMemcpy(input_gpu, input, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyHostToDevice);

  // dim3 threadsPerBlock(THREAD_NUM);
  // dim3 numBlocks(2);
  // sort_pairs<int32_t, float, false>(input, output, indices_ptr_out,
  //                                   TEST_INPUT_NUM);

  cudaMemcpy(output, output_gpu, sizeof(float) * TEST_INPUT_NUM,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(indices_ptr_out, indices_ptr_out_gpu,
             sizeof(int32_t) * TEST_INPUT_NUM, cudaMemcpyDeviceToHost);

  cudaFree(input_gpu);
  cudaFree(output_gpu);
  cudaFree(indices_ptr_out);
}

int main() {
  // test_key<<<1, 2>>>();
  // test_sort();
  // test_prepare_keys();
  test_prepare_indices();
  cudaDeviceSynchronize();
  return 0;
}
