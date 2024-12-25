#include "half.hpp"
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_INPUT_NUM 80
#define THREAD_NUM 2
#define BUCKET_WIDTH 8
#define BUCKET_SIZE (1 << BUCKET_WIDTH)

using half_float::half;
using namespace half_float::literal;

// using DType = half;
// using KType = int16_t;
using DType = float;
using KType = int32_t;

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

// half input[TEST_INPUT_NUM] = {
//     73.561_h,  43.725_h,  -19.391_h, -35.444_h, 84.709_h,  46.380_h,
//     -45.529_h, 83.674_h,  -91.422_h, 65.276_h,  91.310_h,  58.979_h,
//     -20.822_h, 45.184_h, 13.943_h,  -40.147_h, -95.098_h, -9.922_h,
//     -88.527_h, -36.335_h, -9.820_h,
//     78.846_h,  18.892_h,  23.346_h,  59.521_h,
//     -79.604_h, 82.738_h,  71.176_h, 64.105_h,  -38.500_h, -33.920_h,
//     -71.161_h, -75.104_h, -24.361_h, -32.468_h, 41.793_h,  85.418_h,
//     -38.965_h, 80.583_h,  92.184_h,  20.046_h,  -19.807_h, -16.370_h,
//     -63.294_h, -14.729_h, -70.273_h, -11.686_h, -59.074_h, 74.355_h,
//     20.759_h,  -48.987_h, 82.671_h,  31.011_h,  56.022_h, -13.305_h, 1.438_h,
//     34.484_h,  14.673_h,
//     -64.752_h, 5.972_h,   60.900_h,  33.005_h,  46.021_h,
//     -54.705_h, 69.876_h,  21.417_h,  -99.893_h, 8.749_h, -23.831_h, 50.995_h,
//     -66.454_h, -31.685_h, 96.555_h,  24.891_h,
//     -78.219_h, 44.638_h,  17.146_h, 60.752_h,  79.542_h,  53.528_h};

DType output[TEST_INPUT_NUM];
int32_t indices_ptr_out[TEST_INPUT_NUM];

KType keys[TEST_INPUT_NUM] = {0};
KType bfe_keys[TEST_INPUT_NUM] = {0};
KType bfe_keys_out[TEST_INPUT_NUM] = {0};
int32_t offset[TEST_INPUT_NUM] = {0};
// int32_t bucket_offset[BUCKET_SIZE] = {0};
// int32_t curr_count[THREAD_NUM] = {0};
int32_t bucket_offset[BUCKET_SIZE * THREAD_NUM] = {0};
int32_t exclusive_cumsum[BUCKET_SIZE * THREAD_NUM] = {0};
// int32_t exclusive_cumsum[BUCKET_SIZE] = {0};
int32_t indices[2][TEST_INPUT_NUM];

template <typename _T> struct Unsigned_Bits {
  static constexpr _T HIGH_BIT = _T(1) << ((sizeof(_T) * 8) - 1);
  static constexpr _T LOWEST_KEY = _T(-1);
  static constexpr _T MAX_KEY = _T(-1) ^ HIGH_BIT;

  template <bool is_descend> static _T GetKeyForRadixSortBase(_T key) {
    _T mask = (key & HIGH_BIT) ? _T(-1) : HIGH_BIT;
    if (is_descend) {
      return ~(key ^ mask);
    }
    return key ^ mask;
  };

  static int32_t BitfieldExtract(_T source, int32_t bit_start,
                                 int32_t num_bits) {
    const int32_t MASK = (1u << num_bits) - 1;
    return MASK & (source >> bit_start);
  }
};

template <typename _T, typename UB_T> struct Float_Point_NumberBase {
  template <bool is_descend> static UB_T GetKeyForRadixSort(_T key) {
    UB_T key_in = *((UB_T *)&key);
    return Unsigned_Bits<UB_T>::template GetKeyForRadixSortBase<is_descend>(
        key_in);
  };
};

template <typename _T> struct Float_Point_Number;

template <>
struct Float_Point_Number<float> : Float_Point_NumberBase<float, int32_t> {};

template <>
struct Float_Point_Number<half> : Float_Point_NumberBase<half, int16_t> {};

template <typename KeyT, typename ValueT, bool is_descend>
void prepare_keys(const ValueT *d_values_in, KeyT *d_keys_in,
                  int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    d_keys_in[i] =
        Float_Point_Number<ValueT>::template GetKeyForRadixSort<is_descend>(
            d_values_in[i]);
  }
}

void prepare_indices(int32_t *indices, int32_t num_items) {
  for (int32_t i = 0; i < num_items; i++) {
    indices[i] = i;
  }
}

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

void update_indices_ptr(const KType *d_keys_in, const int32_t *indices_ptr_in,
                        const int32_t *offset, const int32_t *exclusive_cumsum,
                        int32_t *indices_ptr_out, int32_t num_items) {
  int32_t num_items_per_thread = num_items / THREAD_NUM;
  for (int32_t i = 0; i < THREAD_NUM; i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx0 = j + i * num_items_per_thread;
      if (idx0 < num_items) {
        int32_t idx =
            offset[idx0] + exclusive_cumsum[d_keys_in[idx0] * THREAD_NUM + i];
        indices_ptr_out[idx] = indices_ptr_in[idx0];
        bfe_keys_out[idx] = bfe_keys[idx0];
      }
    }
  }
}

void put_numbers_into_bucket(const int32_t *d_keys_in, int32_t *offset,
                             int32_t *bucket_offset, int32_t num_items) {
  int32_t num_items_per_thread = num_items / THREAD_NUM;
  for (int32_t i = 0; i < THREAD_NUM; i++) {
    for (int32_t j = 0; j < num_items_per_thread; j++) {
      int32_t idx = j + i * num_items_per_thread;
      if (idx < num_items) {
        offset[idx] = bucket_offset[d_keys_in[idx] * THREAD_NUM + i];
        bucket_offset[d_keys_in[idx] * THREAD_NUM + i]++;
      }
    }
  }
}

void sort_pairs_loop(const int32_t *d_keys_in, int32_t *indices_ptr_in,
                     int32_t *indices_ptr_out, int32_t num_items) {
  put_numbers_into_bucket(d_keys_in, offset, bucket_offset, num_items);

  calc_exclusive_cumsum((int32_t *)bucket_offset, (int32_t *)exclusive_cumsum,
                        BUCKET_SIZE * THREAD_NUM);
  update_indices_ptr(d_keys_in, indices_ptr_in, offset,
                     (int32_t *)exclusive_cumsum, indices_ptr_out, num_items);
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

void test_key() {
  auto key = Float_Point_Number<float>::GetKeyForRadixSort<false>(20000.5);
  auto bfe = Unsigned_Bits<int32_t>::BitfieldExtract(key, 16, 8);
  printf("A 0x%x\n", key);
  printf("B 0x%x\n", bfe);
}

void test_sort() {
  sort_pairs<KType, DType, false>(input, output, indices_ptr_out,
                                  TEST_INPUT_NUM);
  for (int32_t i = 0; i < TEST_INPUT_NUM; i++) {
    printf("%f, ", (float)(output[i]));
    if (i % 10 == 9) {
      printf("\n");
    }
  }
}

int main() {
  //   test_key();
  test_sort();
  return 0;
}
