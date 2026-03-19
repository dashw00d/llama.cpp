//
// Pure SYCL matrix multiplication kernels for use inside SYCL command graph recording.
// These avoid oneMKL/oneDNN which call event.wait() internally.
//
// Copyright (C) 2026 - graph recording compatibility
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_GEMM_PURE_HPP
#define GGML_SYCL_GEMM_PURE_HPP

#include <sycl/sycl.hpp>

// Tiled GEMM: C = alpha * A^T * B + beta * C
// A is (K x M) column-major, transposed to get (M x K)
// B is (K x N) column-major
// C is (M x N) column-major
//
// This matches the oneMKL call:
//   gemm(trans, nontrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
// where A is transposed, B is not.

constexpr int SYCL_GEMM_TILE = 16;

template <typename T_AB, typename T_C>
struct sycl_gemm_kernel {
    const T_AB * __restrict__ A;
    const T_AB * __restrict__ B;
    T_C * __restrict__ C;
    int M, N, K;
    int lda, ldb, ldc;
    T_C alpha, beta;

    void operator()(sycl::nd_item<2> item) const {
        // Each work-item computes one element of C
        const int row = item.get_global_id(1); // M dimension
        const int col = item.get_global_id(0); // N dimension

        if (row >= M || col >= N) return;

        // Local memory tiles
        auto sg = item.get_sub_group();
        const int local_row = item.get_local_id(1);
        const int local_col = item.get_local_id(0);

        // Accumulate in float for precision
        float acc = 0.0f;

        // Tile over K dimension
        for (int k_base = 0; k_base < K; k_base += SYCL_GEMM_TILE) {
            // Each iteration processes SYCL_GEMM_TILE elements of K

            for (int kk = 0; kk < SYCL_GEMM_TILE && (k_base + kk) < K; kk++) {
                int k = k_base + kk;
                // A is transposed: A[k, row] in col-major = A[k + row * lda]
                // Actually: A is (lda x M) col-major, accessing A^T means A[k][row] = A[k + row * lda]
                // Wait - col-major A with dims stored as (lda rows, M cols):
                // A^T(row, k) = A(k, row) = A[k + row * lda]
                float a_val = static_cast<float>(A[k + row * lda]);
                // B is not transposed: B[k, col] in col-major = B[k + col * ldb]
                float b_val = static_cast<float>(B[k + col * ldb]);
                acc += a_val * b_val;
            }
        }

        // C[row + col * ldc]
        if (beta == static_cast<T_C>(0)) {
            C[row + col * ldc] = static_cast<T_C>(alpha * acc);
        } else {
            C[row + col * ldc] = static_cast<T_C>(alpha * acc + static_cast<float>(beta) * static_cast<float>(C[row + col * ldc]));
        }
    }
};

// Launch a pure SYCL GEMM: C = alpha * op(A) * B + beta * C
// op(A) = A^T (transpose)
// A: lda x M (col-major), B: ldb x N (col-major), C: ldc x N (col-major)
template <typename T_AB, typename T_C>
static void sycl_gemm_pure(
    sycl::queue & q,
    int M, int N, int K,
    T_C alpha,
    const T_AB * A, int lda,
    const T_AB * B, int ldb,
    T_C beta,
    T_C * C, int ldc)
{
    // Grid: (ceil(N/TILE), ceil(M/TILE)) work-groups, each (TILE, TILE) work-items
    sycl::range<2> global(
        ((N + SYCL_GEMM_TILE - 1) / SYCL_GEMM_TILE) * SYCL_GEMM_TILE,
        ((M + SYCL_GEMM_TILE - 1) / SYCL_GEMM_TILE) * SYCL_GEMM_TILE
    );
    sycl::range<2> local(SYCL_GEMM_TILE, SYCL_GEMM_TILE);

    q.parallel_for(
        sycl::nd_range<2>(global, local),
        sycl_gemm_kernel<T_AB, T_C>{A, B, C, M, N, K, lda, ldb, ldc, alpha, beta}
    );
}

// Overload for f16 inputs, f32 output (mixed precision)
// A and B are f16, C is f32
static void sycl_gemm_f16_f32(
    sycl::queue & q,
    int M, int N, int K,
    float alpha,
    const sycl::half * A, int lda,
    const sycl::half * B, int ldb,
    float beta,
    float * C, int ldc)
{
    sycl_gemm_pure<sycl::half, float>(q, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// f32 x f32 -> f32
static void sycl_gemm_f32(
    sycl::queue & q,
    int M, int N, int K,
    float alpha,
    const float * A, int lda,
    const float * B, int ldb,
    float beta,
    float * C, int ldc)
{
    sycl_gemm_pure<float, float>(q, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// f16 x f16 -> f16
static void sycl_gemm_f16(
    sycl::queue & q,
    int M, int N, int K,
    sycl::half alpha,
    const sycl::half * A, int lda,
    const sycl::half * B, int ldb,
    sycl::half beta,
    sycl::half * C, int ldc)
{
    sycl_gemm_pure<sycl::half, sycl::half>(q, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Strided batched GEMM: C[b] = A[b]^T * B[b] for b in [0, batch_count)
// A layout: column-major, transposed, stride between batches = strideA
// B layout: column-major, not transposed, stride between batches = strideB
// C layout: column-major, stride between batches = strideC
template <typename T_AB, typename T_C>
struct sycl_gemm_batch_kernel {
    const T_AB * __restrict__ A;
    const T_AB * __restrict__ B;
    T_C * __restrict__ C;
    int M, N, K;
    int lda, ldb, ldc;
    int64_t strideA, strideB, strideC;

    void operator()(sycl::nd_item<3> item) const {
        const int row   = item.get_global_id(2); // M dimension
        const int col   = item.get_global_id(1); // N dimension
        const int batch = item.get_global_id(0); // batch dimension

        if (row >= M || col >= N) return;

        const T_AB * A_b = A + batch * strideA;
        const T_AB * B_b = B + batch * strideB;
        T_C * C_b = C + batch * strideC;

        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = static_cast<float>(A_b[k + row * lda]);
            float b_val = static_cast<float>(B_b[k + col * ldb]);
            acc += a_val * b_val;
        }

        C_b[row + col * ldc] = static_cast<T_C>(acc);
    }
};

template <typename T_AB, typename T_C>
static void sycl_gemm_batch_pure(
    sycl::queue & q,
    int M, int N, int K,
    const T_AB * A, int lda, int64_t strideA,
    const T_AB * B, int ldb, int64_t strideB,
    T_C * C, int ldc, int64_t strideC,
    int batch_count)
{
    sycl::range<3> global(
        batch_count,
        ((N + SYCL_GEMM_TILE - 1) / SYCL_GEMM_TILE) * SYCL_GEMM_TILE,
        ((M + SYCL_GEMM_TILE - 1) / SYCL_GEMM_TILE) * SYCL_GEMM_TILE
    );
    sycl::range<3> local(1, SYCL_GEMM_TILE, SYCL_GEMM_TILE);

    q.parallel_for(
        sycl::nd_range<3>(global, local),
        sycl_gemm_batch_kernel<T_AB, T_C>{A, B, C, M, N, K, lda, ldb, ldc, strideA, strideB, strideC}
    );
}

// f16 batched -> f32 output
static void sycl_gemm_batch_f16_f32(
    sycl::queue & q,
    int M, int N, int K,
    const sycl::half * A, int lda, int64_t strideA,
    const sycl::half * B, int ldb, int64_t strideB,
    float * C, int ldc, int64_t strideC,
    int batch_count)
{
    sycl_gemm_batch_pure<sycl::half, float>(q, M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, batch_count);
}

#endif // GGML_SYCL_GEMM_PURE_HPP
