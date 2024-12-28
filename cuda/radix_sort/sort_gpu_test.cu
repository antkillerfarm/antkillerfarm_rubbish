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

void prepare_indices_cpu(int32_t *indices, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    indices[i] = i;
  }
}

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

void calc_exclusive_cumsum_cpu(const int32_t *value_in,
                               int32_t *exclusive_cumsum, int32_t num_items) {
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

// https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
// Sklansky method
void calc_exclusive_cumsum_cpu2(const int32_t *value_in,
                                int32_t *exclusive_cumsum, int32_t num_items) {
  int32_t num_threads = BLOCK_NUM * THREAD_NUM;
  int32_t block_size = (num_items + (2 * num_threads) - 1) / (2 * num_threads);
  memcpy(exclusive_cumsum, value_in, sizeof(int32_t) * num_items);
  for (uint32_t i = 0; i < 2 * num_threads; i++) {
    for (uint32_t j = 1; j < block_size; j++) {
      uint32_t idx = i * block_size + j;
      if (idx < num_items) {
        exclusive_cumsum[idx] =
            exclusive_cumsum[idx] + exclusive_cumsum[idx - 1];
      }
    }
  }

  // calc inclusive cumsum
  for (uint32_t s = 1; s <= num_threads; s <<= 1) {
    for (uint32_t i = 0; i < num_threads; i++) {
      uint32_t a = (i / s) * (2 * s) + s;
      uint32_t ti = a + (i % s);
      uint32_t si = a - 1;
      // printf("A: %d : %d : %d : %d : %d\n", i, s, a, ti, si);
      if (ti * block_size < num_items) {
        uint32_t idx0 = (si + 1) * block_size - 1;
        for (uint32_t j = 0; j < block_size; j++) {
          uint32_t idx1 = ti * block_size + j;
          // printf("B: %d : %d\n", idx0, idx1);
          if (idx1 < num_items) {
            exclusive_cumsum[idx1] =
                exclusive_cumsum[idx1] + exclusive_cumsum[idx0];
          }
        }
      }
    }
  }

  // calc exclusive cumsum
  for (uint32_t i = 0; i < num_threads; i++) {
    for (uint32_t j = 0; j < 2 * block_size; j++) {
      uint32_t idx = i * 2 * block_size + j;
      if (idx < num_items) {
        exclusive_cumsum[idx] = exclusive_cumsum[idx] - value_in[idx];
      }
    }
  }
}

__global__ void calc_exclusive_cumsum(const int32_t *value_in,
                                      int32_t *exclusive_cumsum,
                                      int32_t num_items) {
  int32_t num_threads = blockDim.x * gridDim.x;
  int32_t block_size = (num_items + (2 * num_threads) - 1) / (2 * num_threads);
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < block_size; j++) {
      uint32_t idx = i * block_size +
                     (blockIdx.x * blockDim.x + threadIdx.x) * 2 * block_size +
                     j;
      if (idx < num_items) {
        exclusive_cumsum[idx] = value_in[idx];
      }
    }
  }
  __syncthreads();
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 1; j < block_size; j++) {
      uint32_t idx = i * block_size +
                     (blockIdx.x * blockDim.x + threadIdx.x) * 2 * block_size +
                     j;
      if (idx < num_items) {
        exclusive_cumsum[idx] += exclusive_cumsum[idx - 1];
      }
    }
  }
  __syncthreads();

  // calc inclusive cumsum
  for (uint32_t s = 1; s <= num_threads; s <<= 1) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t a = (thread_id / s) * (2 * s) + s;
    uint32_t ti = a + (thread_id % s);
    uint32_t si = a - 1;
    if (ti * block_size < num_items) {
      uint32_t idx0 = (si + 1) * block_size - 1;
      for (uint32_t j = 0; j < block_size; j++) {
        uint32_t idx1 = ti * block_size + j;
        if (idx1 < num_items) {
          exclusive_cumsum[idx1] += exclusive_cumsum[idx0];
        }
      }
    }
    __syncthreads();
    // __threadfence();
  }

  // calc exclusive cumsum
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < block_size; j++) {
      uint32_t idx = i * block_size +
                     (blockIdx.x * blockDim.x + threadIdx.x) * 2 * block_size +
                     j;
      if (idx < num_items) {
        exclusive_cumsum[idx] -= value_in[idx];
      }
    }
  }
}

void update_indices_ptr_cpu(const int32_t *d_keys_in,
                            const int32_t *indices_ptr_in,
                            const int32_t *offset,
                            const int32_t *exclusive_cumsum,
                            int32_t *indices_ptr_out, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    int32_t idx = offset[i] + exclusive_cumsum[d_keys_in[i]];
    // printf("XX: %d : %d : %d : %d\n", i, offset[i], d_keys_in[i],
    //        exclusive_cumsum[d_keys_in[i]]);
    indices_ptr_out[idx] = indices_ptr_in[i];
    // bfe_keys_out[idx] = bfe_keys[i];
  }
}

void update_indices_ptr_cpu2(const int32_t *d_keys_in,
                             const int32_t *indices_ptr_in,
                             const int32_t *offset,
                             const int32_t *exclusive_cumsum,
                             int32_t *indices_ptr_out, int32_t num_items) {
  int32_t num_threads = BLOCK_NUM * THREAD_NUM;
  int32_t num_items_per_thread = (num_items + num_threads - 1) / num_threads;
  for (int32_t i = 0; i < num_threads; i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx0 = j + i * num_items_per_thread;
      if (idx0 < num_items) {
        int32_t idx =
            offset[idx0] + exclusive_cumsum[d_keys_in[idx0] * num_threads + i];

        indices_ptr_out[idx] = indices_ptr_in[idx0];
        // bfe_keys_out[idx] = bfe_keys[idx0];
      }
    }
  }
}

__global__ void
update_indices_ptr(const int32_t *d_keys_in, const int32_t *indices_ptr_in,
                   const int32_t *offset, const int32_t *exclusive_cumsum,
                   int32_t *indices_ptr_out, int32_t num_items) {
  int32_t num_threads = blockDim.x * gridDim.x;
  int32_t num_items_per_thread = (num_items + num_threads - 1) / num_threads;
  for (int32_t j = 0; j < num_items_per_thread; j++) {
    int32_t idx0 =
        j + (blockIdx.x * blockDim.x + threadIdx.x) * num_items_per_thread;
    if (idx0 < num_items) {
      int32_t idx = offset[idx0] +
                    exclusive_cumsum[d_keys_in[idx0] * num_threads +
                                     (blockIdx.x * blockDim.x + threadIdx.x)];
      indices_ptr_out[idx] = indices_ptr_in[idx0];
      // bfe_keys_out[idx] = bfe_keys[idx0];
    }
  }
}
