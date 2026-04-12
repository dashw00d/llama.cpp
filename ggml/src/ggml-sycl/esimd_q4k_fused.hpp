// iter18: ESIMD Q4_K fused multi-column matmul, ported from
// gemma4-ipex/src/esimd/linear_forward_q4k_fused_sycl.cpp.
//
// Contract (from PORTING-GUIDE.md):
//   The kernel does NOT read ggml's standard block_q4_K AoS format.
//   It requires a pre-built SoA pair:
//     payload[N*128] : N consecutive 128-byte qs[] arrays
//     meta   [N*16]  : N consecutive 16-byte [d, dmin, scales] tuples
//   The split is the FIRST 16 bytes of each 144-byte block_q4_K go to
//   meta and the LAST 128 bytes go to payload. No data transformation,
//   just rearrangement.
//
//   The kernel uses esimd::block_load<uint8_t, 128> which requires
//   128-byte alignment at every block boundary -- only true with the
//   SoA layout, NOT when reading vx[i].qs at offset +16 inside the
//   144-byte AoS stride. Reading AoS directly produces garbage (the
//   iter4 <pad>-spam bug).

#ifndef GGML_SYCL_ESIMD_Q4K_FUSED_HPP
#define GGML_SYCL_ESIMD_Q4K_FUSED_HPP

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

struct ggml_sycl_esimd_q4k_fused_args {
    const std::uint8_t * payload;     // SoA Q4K payload (128 bytes/block)
    const std::uint8_t * meta;        // SoA Q4K meta    (16 bytes/block)
    int                  n_blocks_per_row;
    const float *        x;           // input activations
    float *              y;           // output
    int                  n_rows;
    int                  n_cols;      // src1->ne[1]; <= 8 supported
    std::size_t          x_col_stride;
    std::size_t          y_col_stride;
};

// Dispatch the ESIMD fused Q4K matmul on the given queue.
// Caller is responsible for the payload+meta SoA buffers being live
// at queue execution time (use the cache helper below).
void ggml_sycl_esimd_q4k_fused_dispatch(
    sycl::queue & queue,
    const ggml_sycl_esimd_q4k_fused_args & args);

// Lookup or create the SoA (payload, meta) pair for a given AoS Q4K
// weight buffer on a given device. The cache key is (device, src0_data).
// `src0_data` is the device pointer to the AoS block_q4_K[N] buffer
// (i.e. tensor->data for the weight). `n_blocks` is the total number
// of Q4K blocks in the buffer. On first call, allocates two new device
// buffers and copies the AoS bytes into them in SoA order via a small
// SYCL kernel. On subsequent calls returns the cached pointers.
//
// Returns true on success and writes payload/meta out-pointers; false
// if allocation failed (caller should fall back to the non-ESIMD path).
bool ggml_sycl_esimd_q4k_get_or_build_soa(
    sycl::queue &       queue,
    int                 device,
    const void *        src0_data_aos,
    std::size_t         n_blocks,
    const std::uint8_t ** out_payload,
    const std::uint8_t ** out_meta);

#endif // GGML_SYCL_ESIMD_Q4K_FUSED_HPP
