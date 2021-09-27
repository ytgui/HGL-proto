#include <torch/extension.h>

template <typename index_t, typename value_t>
__global__ void propogate_move_cuda_kernel(
    index_t n_nodes, index_t n_features,
    const index_t *indptr, const index_t *indices, const value_t *features,
    value_t *out_features) {
    for (index_t row = blockIdx.x; row < n_nodes; row += gridDim.x) {
        for (index_t k = threadIdx.x; k < n_features; k += blockDim.x) {
            value_t out = 0.0;
            for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                index_t col = indices[i];
                out += features[col * n_features + k];
            }
            out_features[row * n_features + k] = out;
        }
    }
}

torch::Tensor propogate_move_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &features) {
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(features.dim() == 2);
    int32_t n_nodes = indptr.size(0) - 1;
    int32_t n_features = features.size(1);

    auto new_features = torch::zeros({n_nodes, n_features}, features.options());
    propogate_move_cuda_kernel<int32_t, float><<<n_nodes, 32>>>(
        n_nodes, n_features, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        features.data_ptr<float>(), new_features.data_ptr<float>());
    return new_features;
}

template <typename index_t, typename value_t>
__global__ void propogate_spmm_cuda_kernel(
    index_t n_nodes, index_t n_features,
    const index_t *indptr, const index_t *indices,
    const value_t *values, const value_t *features,
    value_t *out_features) {
    for (index_t row = blockIdx.x; row < n_nodes; row += gridDim.x) {
        for (index_t k = threadIdx.x; k < n_features; k += blockDim.x) {
            value_t out = 0.0;
            for (index_t i = indptr[row]; i < indptr[row + 1]; i += 1) {
                index_t col = indices[i];
                out += values[i] * features[col * n_features + k];
            }
            out_features[row * n_features + k] = out;
        }
    }
}

torch::Tensor propogate_spmm_cuda(const torch::Tensor &indptr,
                                  const torch::Tensor &indices,
                                  const torch::Tensor &values,
                                  const torch::Tensor &features) {
    TORCH_CHECK(indptr.dim() == 1);
    TORCH_CHECK(features.dim() == 2);
    int32_t n_nodes = indptr.size(0) - 1;
    int32_t n_features = features.size(1);

    auto new_features = torch::zeros({n_nodes, n_features}, features.options());

    propogate_spmm_cuda_kernel<int32_t, float><<<n_nodes, min(32, n_features)>>>(
        n_nodes, n_features, indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
        values.data_ptr<float>(), features.data_ptr<float>(), new_features.data_ptr<float>());

    return new_features;
}
