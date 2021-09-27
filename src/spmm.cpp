#include <iostream>

#include "common.h"

torch::Tensor propogate_move_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &features);

torch::Tensor propogate_spmm_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &values,
                                  const torch::Tensor &features);

torch::Tensor propogate_move(const torch::Tensor &indptr,
                             const torch::Tensor &indices,
                             const torch::Tensor &features) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(features, torch::kFloat32);
    return propogate_move_cuda(indptr, indices, features);
}

torch::Tensor propogate_spmm(const torch::Tensor &indptr,
                             const torch::Tensor &indices,
                             const torch::Tensor &values,
                             const torch::Tensor &features) {
    CHECK_INPUT(indptr, torch::kInt32);
    CHECK_INPUT(indices, torch::kInt32);
    CHECK_INPUT(values, torch::kFloat32);
    CHECK_INPUT(features, torch::kFloat32);

    return propogate_spmm_cuda(indptr, indices, values, features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("propogate_move", &propogate_move, "SPMM forward move (CUDA)");
    m.def("propogate_spmm", &propogate_spmm, "SPMM forward matmul (CUDA)");
}