#include <iostream>

#include "common.h"

torch::Tensor _gspmm_cuda(
    const torch::Tensor &indptr,
    const torch::Tensor &indices,
    const torch::Tensor &features);

torch::Tensor gspmm(const torch::Tensor &indptr,
                    const torch::Tensor &indices,
                    const torch::Tensor &features) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(features, torch::kFloat32);

    return _gspmm_cuda(indptr, indices, features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gspmm", &gspmm, "SPMM forward matmul (CUDA)");
}