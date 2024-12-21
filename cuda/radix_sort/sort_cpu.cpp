#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_INPUT_NUM 80
#define BUCKET_WIDTH 8
#define BUCKET_SIZE (1 << 8)

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

unsigned int index[TEST_INPUT_NUM] = {0};
unsigned int count[TEST_INPUT_NUM] = {0};
unsigned int exclusive_cumsum[TEST_INPUT_NUM] = {0};
unsigned int exclusive_cumsum[1 << ] = {0};


template <typename _T> struct Unsigned_Bits {
  static constexpr _T HIGH_BIT = _T(1) << ((sizeof(_T) * 8) - 1);
  static constexpr _T LOWEST_KEY = _T(-1);
  static constexpr _T MAX_KEY = _T(-1) ^ HIGH_BIT;

  static _T GetKeyForRadixSortBase(_T key) {
    _T mask = (key & HIGH_BIT) ? _T(-1) : HIGH_BIT;
    return key ^ mask;
  };
};

template <typename _T, typename UB_T>
struct Float_Point_NumberBase {
  static UB_T GetKeyForRadixSort(_T key) {
    UB_T key_in = *((UB_T *)&key);
    return Unsigned_Bits<UB_T>::GetKeyForRadixSortBase(key_in);
  };
  static unsigned int
    BitfieldExtract(UB_T source, unsigned int bit_start, unsigned int num_bits)
    {
    const unsigned int MASK = (1u << num_bits) - 1;
    return (source >> bit_start) & MASK;
    }
};

template <typename _T> struct Float_Point_Number;

template <>
struct Float_Point_Number<float> : Float_Point_NumberBase<float, unsigned int> {
};



void test_key() {
  auto key = Float_Point_Number<float>::GetKeyForRadixSort(20000.5);
  auto bfe = Float_Point_Number<float>::BitfieldExtract(key, 16, 8);
  printf("A 0x%x\n", key);
  printf("B 0x%x\n", bfe);
}

int main() {
  test_key();
  return 0;
}
