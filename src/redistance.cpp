#include <math.h>
#include <vector>
#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit/tensor.h>
#include <nanobind/nanobind.h>

namespace dr = drjit;

template <typename T>
dr::Tensor<T> arange(Py_ssize_t size) {
  return dr::Tensor<T>(dr::arange<T>(size));
}

template dr::Tensor<dr::LLVMArray<float>> arange(Py_ssize_t);

int add(int a, int b) { return a + b; }

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(MODULE, m) {
  m.def("add", &add, "a"_a, "b"_a);
  m.def("arange", &arange<dr::LLVMArray<float>>);
}