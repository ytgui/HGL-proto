#include <iostream>

#include "common.h"

torch::Tensor _spmm_cuda(
    const torch::Tensor &values,
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features);

torch::Tensor spmm(const torch::Tensor &values,
                   const torch::Tensor &indptr,
                   const torch::Tensor &indices,
                   const torch::Tensor &features) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(values, torch::kFloat32);
    CHECK_INPUT(features, torch::kFloat32);

    return _spmm_cuda(values, indptr, indices, features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm", &spmm, "SPMM forward matmul (CUDA)");
}