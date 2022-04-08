// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend --cuda-gpu-arch=sm_80
// RUN: %t.out

#include <CL/sycl.hpp>

#include <cmath>
#include <vector>

using namespace cl::sycl;
using sycl::ext::oneapi::experimental::bfloat16;

constexpr int N = 16 * 3; // divisible by all vector sizes
constexpr float bf16_eps = 0.00390625;

union conv {
  float f;
  vec<uint16_t, 2> u;
  uint32_t u2;
};

float from_bf16(uint16_t x) {
  conv c;
  c.u.y() = x;
  c.u.x() = 0;
  return c.f;
}

bool check(float a, float b) {
  return fabs(2 * (a - b) / (a + b)) > bf16_eps * 2;
}

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read_write>(cgh);                \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(from_bf16(NAME(bfloat16{A[index]}).raw()),                   \
                  NAME(A[index]))) {                                           \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_1(NAME) TEST_BUILTIN_1_SCAL_IMPL(NAME)

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(                                                             \
                from_bf16(NAME(bfloat16{A[index]}, bfloat16{B[index]}).raw()), \
                NAME(A[index], B[index]))) {                                   \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_2(NAME) TEST_BUILTIN_2_SCAL_IMPL(NAME)

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    buffer<float> a_buf(&a[0], N);                                             \
    buffer<float> b_buf(&b[0], N);                                             \
    buffer<float> c_buf(&c[0], N);                                             \
    buffer<int> err_buf(&err, 1);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto ERR = err_buf.get_access<access::mode::write>(cgh);                 \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        if (check(from_bf16(NAME(bfloat16{A[index]}, bfloat16{B[index]},       \
                                 bfloat16{C[index]})                           \
                                .raw()),                                       \
                  NAME(A[index], B[index], C[index]))) {                       \
          ERR[0] = 1;                                                          \
        }                                                                      \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  assert(err == 0);

#define TEST_BUILTIN_3(NAME) TEST_BUILTIN_3_SCAL_IMPL(NAME)

int main() {
  queue q;

  auto computeCapability =
      std::stof(q.get_device().get_info<sycl::info::device::backend_version>());
  // TODO check for "ext_oneapi_bfloat16" aspect instead once aspect is
  // supported. Since this test only covers CUDA the current check is
  // functionally equivalent to "ext_oneapi_bfloat16".
  if (computeCapability >= 8.0) {
    std::vector<float> a(N), b(N), c(N);
    int err = 0;

    for (int i = 0; i < N; i++) {
      a[i] = (i - N / 2) / (float)N;
      b[i] = (N / 2 - i) / (float)N;
      c[i] = (float)(3 * i);
    }

    TEST_BUILTIN_1(fabs);
    TEST_BUILTIN_2(fmin);
    TEST_BUILTIN_2(fmax);
    TEST_BUILTIN_3(fma);
  }
  return 0;
}
