#include "common.h"
#include <torch/extension.h>

std::vector<torch::Tensor> _b2gemm_cuda(const torch::Tensor &x,
                                        const torch::Tensor &w_a,
                                        const torch::Tensor &w_b) {
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(w_a.dim() == 2);
    TORCH_CHECK(w_a.sizes() == w_b.sizes());
    TORCH_CHECK(w_a.strides() == w_b.strides());

    //
    int n = x.size(0);
    int k = x.size(1);
    int m = w_a.size(0);
    auto y_a = torch::zeros({n, m}, w_a.options());
    auto y_b = torch::zeros({n, m}, w_b.options());
    TORCH_CHECK(y_a.strides() == y_b.strides());
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    //
    int ld_a = w_a.stride(0);
    int ld_b = x.stride(0);
    int ld_c = y_a.stride(0);
    auto A = w_a.data_ptr<float>();
    auto B = x.data_ptr<float>();
    auto C = y_a.data_ptr<float>();
    int offset_a = w_b.data_ptr<float>() - A;
    if (offset_a < 0) {
        offset_a = -offset_a;
        A = w_b.data_ptr<float>();
    }
    int offset_c = y_b.data_ptr<float>() - C;
    if (offset_c < 0) {
        offset_c = -offset_c;
        C = y_b.data_ptr<float>();
    }

    float alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, ld_a, offset_a,
        B, ld_b, 0,
        &beta,
        C, ld_c, offset_c, 2));

    return {y_a, y_b};
}

std::vector<torch::Tensor> _b2gemm_backward_cuda(const torch::Tensor &x,
                                                 const torch::Tensor &w_a,
                                                 const torch::Tensor &w_b,
                                                 const torch::Tensor &grad_a,
                                                 const torch::Tensor &grad_b) {
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(w_a.dim() == 2);
    TORCH_CHECK(grad_a.dim() == 2);
    TORCH_CHECK(w_a.sizes() == w_b.sizes());
    TORCH_CHECK(w_a.strides() == w_b.strides());
    TORCH_CHECK(grad_a.sizes() == grad_b.sizes());
    TORCH_CHECK(grad_a.strides() == grad_b.strides());
    auto handle = at::cuda::getCurrentCUDABlasHandle();

    //
    torch::Tensor dw_a, dw_b;
    {
        int n = grad_a.size(1);
        int k = x.size(0);
        int m = x.size(1);
        dw_a = torch::zeros({n, m}, w_a.options());
        dw_b = torch::zeros({n, m}, w_b.options());
        TORCH_CHECK(dw_a.strides() == dw_b.strides());

        //
        int ld_a = x.stride(0);
        int ld_b = grad_a.stride(0);
        int ld_c = dw_a.stride(0);
        auto A = x.data_ptr<float>();
        auto B = grad_a.data_ptr<float>();
        auto C = dw_a.data_ptr<float>();
        int offset_b = grad_b.data_ptr<float>() - B;
        if (offset_b < 0) {
            offset_b = -offset_b;
            B = grad_a.data_ptr<float>();
        }
        int offset_c = dw_b.data_ptr<float>() - C;
        if (offset_c < 0) {
            offset_c = -offset_c;
            C = dw_b.data_ptr<float>();
        }

        float alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, k,
            &alpha,
            A, ld_a, 0,
            B, ld_b, offset_b,
            &beta,
            C, ld_c, offset_c, 2));
    }

    //
    torch::Tensor dx;
    {
        int n = x.size(0);
        int k = grad_a.size(1);
        int m = x.size(1);
        dx = torch::zeros({2, n, m}, x.options());
        // TORCH_CHECK(x.strides() == dx.strides());

        //
        int ld_a = w_a.stride(0);
        int ld_b = grad_a.stride(0);
        int ld_c = dx.stride(1);
        auto A = w_a.data_ptr<float>();
        auto B = grad_a.data_ptr<float>();
        auto C = dx.data_ptr<float>();
        int offset_a = w_b.data_ptr<float>() - A;
        if (offset_a < 0) {
            offset_a = -offset_a;
            A = w_b.data_ptr<float>();
        }
        int offset_b = grad_b.data_ptr<float>() - B;
        if (offset_b < 0) {
            offset_b = -offset_b;
            B = grad_a.data_ptr<float>();
        }
        int offset_c = dx.stride(0);

        float alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            A, ld_a, offset_a,
            B, ld_b, offset_b,
            &beta,
            C, ld_c, offset_c, 2));
    }

    return {dx, dw_a, dw_b};
}