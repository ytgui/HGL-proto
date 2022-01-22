#include "common.h"
#include <torch/extension.h>

void b2gemm_cuda(const torch::Tensor &x,
                 const torch::Tensor &w_a,
                 const torch::Tensor &w_b,
                 torch::Tensor &y_a,
                 torch::Tensor &y_b) {
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(w_a.dim() == 2);
    TORCH_CHECK(y_a.dim() == 2);
    TORCH_CHECK(w_a.sizes() == w_b.sizes());
    TORCH_CHECK(y_a.sizes() == y_b.sizes());
    TORCH_CHECK(w_a.strides() == w_b.strides());
    TORCH_CHECK(y_a.strides() == y_b.strides());

    auto n = x.size(0);
    auto k = x.size(1);
    auto m = w_a.size(0);
    // auto stream = at::cuda::getCurrentCUDAStream();
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    /*

    std::array<float *, 6> host_group = {
        w_a.data_ptr<float>(), w_b.data_ptr<float>(),
        x.data_ptr<float>(), x.data_ptr<float>(),
        y_a.data_ptr<float>(), y_b.data_ptr<float>()};
    float **device_group = nullptr;
    CUDA_CHECK(cudaMalloc(&device_group, 6 * sizeof(float *)));
    CUDA_CHECK(cudaMemcpyAsync((void *)device_group, host_group.data(),
                               6 * sizeof(float *), cudaMemcpyHostToDevice, stream));

    float alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasSgemmBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        &device_group[0], w_a.stride(0),
        &device_group[2], x.stride(0),
        &beta,
        &device_group[4], y_a.stride(0), 2));

    CUDA_CHECK(cudaFree(device_group));

    */

    float alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        w_a.data_ptr<float>(), w_a.stride(0),
        x.data_ptr<float>(), x.stride(0),
        &beta,
        y_a.data_ptr<float>(), y_a.stride(0)));
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        w_b.data_ptr<float>(), w_b.stride(0),
        x.data_ptr<float>(), x.stride(0),
        &beta,
        y_b.data_ptr<float>(), y_b.stride(0)));
}