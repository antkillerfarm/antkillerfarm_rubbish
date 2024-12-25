#define TEST_INPUT_NUM 80
#define THREAD_NUM 2
#define BUCKET_WIDTH 8
#define BUCKET_SIZE (1 << BUCKET_WIDTH)

using DType = float;
using KType = int32_t;

template <typename _T> struct Unsigned_Bits {
  static constexpr _T HIGH_BIT = _T(1) << ((sizeof(_T) * 8) - 1);
  static constexpr _T LOWEST_KEY = _T(-1);
  static constexpr _T MAX_KEY = _T(-1) ^ HIGH_BIT;

  template <bool is_descend>
  static __host__ __device__ _T GetKeyForRadixSortBase(_T key) {
    _T mask = (key & HIGH_BIT) ? _T(-1) : HIGH_BIT;
    if (is_descend) {
      return ~(key ^ mask);
    }
    return key ^ mask;
  };

  static __host__ __device__ int32_t BitfieldExtract(_T source,
                                                     int32_t bit_start,
                                                     int32_t num_bits) {
    const int32_t MASK = (1u << num_bits) - 1;
    return MASK & (source >> bit_start);
  }
};

template <typename _T, typename UB_T> struct Float_Point_NumberBase {
  template <bool is_descend>
  static __host__ __device__ UB_T GetKeyForRadixSort(_T key) {
    UB_T key_in = *((UB_T *)&key);
    return Unsigned_Bits<UB_T>::template GetKeyForRadixSortBase<is_descend>(
        key_in);
  };
};

template <typename _T> struct Float_Point_Number;

template <>
struct Float_Point_Number<float> : Float_Point_NumberBase<float, int32_t> {};

template <typename KeyT, typename ValueT, bool is_descend>
__global__ void prepare_keys(const ValueT *d_values_in, KeyT *d_keys_in,
                             int32_t num_items) {
  int block_size =
      num_items + (blockDim.x * gridDim.x) - 1 / (blockDim.x * gridDim.x);

  for (int32_t i = 0; i < block_size; i++) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * block_size + i;
    if (idx < num_items) {
      d_keys_in[idx] =
          Float_Point_Number<ValueT>::template GetKeyForRadixSort<is_descend>(
              d_values_in[idx]);
    }
  }
}

extern float input[TEST_INPUT_NUM];

void print_ivec(int32_t *vec, int32_t num_items);
void print_ixvec(int32_t *vec, int32_t num_items);
void print_fvec(float *vec, int32_t num_items);

template <typename KeyT, typename ValueT, bool is_descend>
void prepare_keys_cpu(const ValueT *d_values_in, KeyT *d_keys_in,
                      int32_t num_items);

void test_prepare_keys();
void test_prepare_indices();
