#pragma once
#include <cstdint>
#include <cuda.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL_DIV(a, b) ((a + b - 1) / (b))

#define STR(x) #x
#define PRAGMA_UNROLL _Pragma(STR(unroll))
#define PRAGMA_UNROLL_COUNT(n) _Pragma(STR(unroll n))
#define CUDA_INLINE __device__ __forceinline__


template <typename T>
CUDA_INLINE uint32_t cast_smem_ptr_to_uint(T *smem_ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
};

constexpr uint32_t static_next_power_of_2(uint32_t v) {
  if (v <= 1) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

constexpr uint32_t get_max_load_bytes(uint32_t bytes) {
  if (bytes % 16 == 0) return 16;
  if (bytes % 8 == 0) return 8;
  if (bytes % 4 == 0) return 4;
  if (bytes % 2 == 0) return 2;
  return 1;
}

template <int bytes>
struct LoadTypeChooser {
  using Type = typename LoadTypeChooser<get_max_load_bytes(bytes)>::Type;
};
template <>
struct LoadTypeChooser<1> {
  using Type = uint8_t;
};
template <>
struct LoadTypeChooser<2> {
  using Type = uint16_t;
};
template <>
struct LoadTypeChooser<4> {
  using Type = uint32_t;
};
template <>
struct LoadTypeChooser<8> {
  using Type = uint2;
};
template <>
struct LoadTypeChooser<16> {
  using Type = uint4;
};

// Zero-size array wrapper: empty struct when size is 0, zero overhead.
// NVCC supports zero-size arrays as a GCC extension; NVRTC does not.
template <typename T, uint32_t N>
struct ZArray {
  T _data[N];
  CUDA_INLINE T& operator[](uint32_t i) { return _data[i]; }
  CUDA_INLINE const T& operator[](uint32_t i) const { return _data[i]; }
  CUDA_INLINE operator T*() { return _data; }
  CUDA_INLINE operator const T*() const { return _data; }
};
template <typename T>
struct ZArray<T, 0> {
  // Dummy accessors for dead-code type-checking (never called at runtime).
  CUDA_INLINE T& operator[](uint32_t) { return *reinterpret_cast<T*>(this); }
  CUDA_INLINE const T& operator[](uint32_t) const { return *reinterpret_cast<const T*>(this); }
  CUDA_INLINE operator T*() { return reinterpret_cast<T*>(this); }
  CUDA_INLINE operator const T*() const { return reinterpret_cast<const T*>(this); }
};

// 2D version: empty when inner dimension is 0.
template <typename T, uint32_t N1, uint32_t N2>
struct ZArray2D {
  T _data[N1][N2];
  CUDA_INLINE T* operator[](uint32_t i) { return _data[i]; }
  CUDA_INLINE const T* operator[](uint32_t i) const { return _data[i]; }
  static constexpr uint32_t kRowBytes = N2 * sizeof(T);
};
template <typename T, uint32_t N1>
struct ZArray2D<T, N1, 0> {
  CUDA_INLINE T* operator[](uint32_t) { return reinterpret_cast<T*>(this); }
  CUDA_INLINE const T* operator[](uint32_t) const { return reinterpret_cast<const T*>(this); }
  static constexpr uint32_t kRowBytes = 0;
};

template <uint32_t M_, uint32_t N_, uint32_t K_>
struct Shape {
  static constexpr uint32_t M = M_;
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
};
