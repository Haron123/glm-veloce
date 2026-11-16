#include <vector>
#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

// narrow rows (a few values each, e.g. one score per block expanded to its cells): one thread per value; the
// block-per-row launch below ran one mostly idle 256-thread block per gathered row (32K rows of one float: ~0.3 ms)
template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_narrow(
        const src0_t * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne10, const int64_t ne11, const int64_t n,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t i00 = i % ne00;
    const int64_t r   = i / ne00;
    const int64_t i10 = r % ne10;
    const int64_t i11 = (r / ne10) % ne11;
    const int64_t i12 = r / (ne10*ne11);
    const int32_t i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const src0_t * src0_row = (const src0_t *) ((const char *) src0 + (int64_t) i01*nb01 + i11*nb02 + i12*nb03);
    dst[i10*s1 + i11*s2 + i12*s3 + i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    if (ne00 <= 16) {
        const int64_t n = ne00*ne10*ne11*ne12;
        const int64_t nblocks = (n + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        if (nblocks <= INT_MAX) {
            k_get_rows_narrow<<<(int) nblocks, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>(src0_d, src1_d, dst_d,
                ne00, ne10, ne11, n, s1, s2, s3, nb01, nb02, nb03, s10, s11, s12);
            return;
        }
    }

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);

    // GGML_CUDA_DUMP_GR=1 debug: compare quantized get_rows outputs with a host dequantization (small tensors)
    static const bool dump_gr = [] { const char * e = getenv("GGML_CUDA_DUMP_GR"); return e && atoi(e) != 0; }();
    if (dump_gr && ggml_is_quantized(src0->type) && ggml_nelements(src0) <= (1 << 20) && dst->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && ne11 == 1 && ne12 == 1) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<char> w(ggml_nbytes(src0));
        std::vector<int32_t> ids(ne10);
        std::vector<float> out(ggml_nelements(dst)), ref(ne00);
        CUDA_CHECK(cudaMemcpy(w.data(), src0->data, w.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ids.data(), src1->data, ids.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out.data(), dst->data, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
        double maxd = 0.0, maxv = 0.0;
        for (int64_t i = 0; i < ne10; ++i) {
            ggml_get_type_traits(src0->type)->to_float(w.data() + (size_t) ids[i]*nb01, ref.data(), ne00);
            for (int64_t k = 0; k < ne00; ++k) {
                maxd = std::max(maxd, (double) fabsf(ref[k] - out[i*ne00 + k]));
                maxv = std::max(maxv, (double) fabsf(ref[k]));
            }
        }
        fprintf(stderr, "get_rows dev %d %s %s [%lld,%lld] ids0 %d: max |gpu - ref| %.3g (max |ref| %.3g)\n", ctx.device,
            src0->name, ggml_type_name(src0->type), (long long) ne00, (long long) ne10, ne10 > 0 ? ids[0] : -1, maxd, maxv);
    }
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}

// GGML_OP_GET_ROWS_MEAN: one block per output row, threads over the row; the n source rows are summed in registers
template <typename T>
static __global__ void k_get_rows_mean(const T * __restrict__ src0, const int32_t * __restrict__ src1, float * __restrict__ dst,
        const int n, const int64_t ne00, const int64_t ne01, const int64_t m, const int64_t ne11,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t j   = blockIdx.x;
    const int64_t i11 = blockIdx.y;
    const int64_t i12 = blockIdx.z;
    const float scale = 1.0f / (float) n;
    float * d = (float *) ((char *) dst + j*nb1 + i11*nb2 + i12*nb3);
    for (int64_t c = threadIdx.x; c < ne00; c += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) {
            const int64_t i10 = (int64_t) n*j + i;
            const int32_t i01 = *(const int32_t *) ((const char *) src1 + i10*nb10 + i11*nb11 + i12*nb12);
            const T * srow = (const T *) ((const char *) src0 + (int64_t) i01*nb01 + i11*nb02 + i12*nb03);
            acc += ggml_cuda_cast<float>(srow[c]);
        }
        d[c] = acc*scale;
    }
}

void ggml_cuda_op_get_rows_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const int     n = ggml_get_op_params_i32(dst, 0);
    const int64_t m = ne10/n;
    GGML_ASSERT(src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type) && nb0 == sizeof(float));
    GGML_ASSERT(m <= INT_MAX && ne11 <= 65535 && ne12 <= 65535);

    const dim3 grid((unsigned) m, (unsigned) ne11, (unsigned) ne12);
    const int  nt = (int) std::min<int64_t>(256, ((ne00 + 63)/64)*64);
    cudaStream_t stream = ctx.stream();
    if (src0->type == GGML_TYPE_F16) {
        k_get_rows_mean<half><<<grid, nt, 0, stream>>>((const half *) src0->data, (const int32_t *) src1->data, (float *) dst->data,
            n, ne00, ne01, m, ne11, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3);
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        k_get_rows_mean<float><<<grid, nt, 0, stream>>>((const float *) src0->data, (const int32_t *) src1->data, (float *) dst->data,
            n, ne00, ne01, m, ne11, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3);
    }
}

// qwen4exp indexer epilogue: RELU of the per-head block scores -> sum over the heads -> + per-block bias -> expand
// blocks to cells (GET_ROWS through two permutes) -> + attention mask (cast to F32) in one kernel, same arithmetic
// order as the unfused chain (bit-identical):
//   out[c,t] = (((relu(s[b,0,t]) + relu(s[b,1,t])) + ...) + bias[b,t]) + mask[c,t],  b = cell_blk[c]
// Replaces ~11 kernels per indexer layer (decode) and their passes over n_kv-sized tensors (long context).
template <typename T_mask>
static __global__ void __launch_bounds__(256) idx_score_expand(
        const float * __restrict__ score, const float * __restrict__ bias, const int32_t * __restrict__ cell_blk,
        const T_mask * __restrict__ mask, float * __restrict__ out, const int n_kv, const int n_h,
        const int64_t ss_h, const int64_t ss_t, const int64_t sb_t, const int64_t sm_t, const int64_t so_t) {
    const int c = blockIdx.x*256 + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (c >= n_kv) {
        return;
    }
    const int b = cell_blk[c];
    const float * sp = score + b + t*ss_t;
    float v = fmaxf(sp[0], 0.0f);
    for (int h = 1; h < n_h; ++h) {
        v += fmaxf(sp[h*ss_h], 0.0f);
    }
    v += bias[b + t*sb_t];
    v += ggml_cuda_cast<float>(mask[c + t*sm_t]);
    out[c + t*so_t] = v;
}

void ggml_cuda_idx_score_expand(ggml_backend_cuda_context & ctx, const ggml_tensor * score, const ggml_tensor * bias,
                                const ggml_tensor * cell_blk, const ggml_tensor * mask, ggml_tensor * out) {
    const int n_kv = (int) out->ne[0], n_t = (int) out->ne[1];
    const dim3 grid((n_kv + 255)/256, n_t);
    const int64_t ss_h = score->nb[1]/sizeof(float), ss_t = score->nb[2]/sizeof(float);
    const int64_t sb_t = bias->nb[1]/sizeof(float), so_t = out->nb[1]/sizeof(float);
    if (mask->type == GGML_TYPE_F16) {
        idx_score_expand<half><<<grid, 256, 0, ctx.stream()>>>((const float *) score->data, (const float *) bias->data,
            (const int32_t *) cell_blk->data, (const half *) mask->data, (float *) out->data, n_kv, (int) score->ne[1],
            ss_h, ss_t, sb_t, mask->nb[1]/sizeof(half), so_t);
    } else {
        idx_score_expand<float><<<grid, 256, 0, ctx.stream()>>>((const float *) score->data, (const float *) bias->data,
            (const int32_t *) cell_blk->data, (const float *) mask->data, (float *) out->data, n_kv, (int) score->ne[1],
            ss_h, ss_t, sb_t, mask->nb[1]/sizeof(float), so_t);
    }
    CUDA_CHECK(cudaGetLastError());
}

// qwen4exp pooled-key scores at decode: GET_ROWS(f16 pooled-key cache, block reps) -> MUL_MAT(., indexer q) in one pass
// over the cached keys: the f32 copy of every block's key (n_blocks x 512 B) is never written or read back. 16 lanes per
// row, 8 halves each (one 16-byte load), the N query columns in registers. The dot products are summed in another order
// than MUL_MAT's, so the scores differ at the rounding level.
template <int N>
static __global__ void __launch_bounds__(256) qsa_gather_score(const char * __restrict__ k, const int32_t * __restrict__ ids,
        const float * __restrict__ q, float * __restrict__ dst, const int n_rows, const int64_t k_nb1, const int64_t q_s1,
        const int64_t d_s1) {
    __shared__ float qs[N][128];
    for (int t = threadIdx.x; t < N*128; t += blockDim.x) {
        qs[t / 128][t % 128] = q[(t / 128)*q_s1 + t % 128];
    }
    __syncthreads();
    const int l     = threadIdx.x & 15;
    const int grp   = (blockIdx.x*blockDim.x + threadIdx.x) >> 4;
    const int n_grp = (gridDim.x*blockDim.x) >> 4;
    float qr[N][8];
#pragma unroll
    for (int n = 0; n < N; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qr[n][i] = qs[n][8*l + i];
        }
    }
    // two rows per step: the id load and the dependent row load of the second overlap the first's
    for (int r0 = grp; r0 < n_rows; r0 += 2*n_grp) {
        const int  r1  = r0 + n_grp;
        const bool ok1 = r1 < n_rows;
        const int  id0 = ids[r0];
        const int  id1 = ids[ok1 ? r1 : r0];
        const int4 raw[2] = { *(const int4 *) (k + (int64_t) id0*k_nb1 + 16*l), *(const int4 *) (k + (int64_t) id1*k_nb1 + 16*l) };
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const half2 * h = (const half2 *) &raw[u];
            float kf[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(h[i]);
                kf[2*i] = f.x;
                kf[2*i + 1] = f.y;
            }
            const int r = u == 0 ? r0 : r1;
#pragma unroll
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc += qr[n][i]*kf[i];
                }
                acc = warp_reduce_sum<16>(acc);
                if (l == 0 && (u == 0 || ok1)) {
                    dst[r + n*d_s1] = acc;
                }
            }
        }
    }
}

bool ggml_cuda_qsa_gather_score_supported(const ggml_tensor * gr, const ggml_tensor * mm) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_QSA_SCORE_FUSE"); return !e || atoi(e) != 0; }();
    // 12/16 columns: the MTP verify (4 heads x 3/4 tokens); GGML_CUDA_QSA_SCORE_FUSE_NT=0: those stay GET_ROWS + MUL_MAT
    static const bool nt = [] { const char * e = getenv("GGML_CUDA_QSA_SCORE_FUSE_NT"); return !e || atoi(e) != 0; }();
    if (!enabled || !gr || !mm || mm->op != GGML_OP_MUL_MAT || mm->src[0] != gr || gr->op != GGML_OP_GET_ROWS) {
        return false;
    }
    const ggml_tensor * k = gr->src[0], * ids = gr->src[1], * q = mm->src[1];
    return k->type == GGML_TYPE_F16 && k->ne[0] == 128 && k->nb[0] == sizeof(half) && k->nb[1] % 16 == 0 &&
        (uintptr_t) k->data % 16 == 0 && k->ne[2] == 1 && k->ne[3] == 1 &&
        ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        gr->type == GGML_TYPE_F32 && gr->ne[0] == 128 && gr->ne[1] == ids->ne[0] && gr->ne[2] == 1 && gr->ne[3] == 1 &&
        gr->ne[1] <= INT_MAX &&
        q->type == GGML_TYPE_F32 && q->ne[0] == 128 && q->ne[1] >= 1 && (q->ne[1] <= 8 || (nt && (q->ne[1] == 12 || q->ne[1] == 16))) &&
        q->ne[2] == 1 && q->ne[3] == 1 &&
        q->nb[0] == sizeof(float) && q->nb[1] % sizeof(float) == 0 &&
        mm->type == GGML_TYPE_F32 && mm->ne[0] == gr->ne[1] && mm->ne[1] == q->ne[1] && mm->ne[2] == 1 && mm->ne[3] == 1 &&
        mm->nb[0] == sizeof(float) && mm->nb[1] % sizeof(float) == 0;
}

void ggml_cuda_qsa_gather_score(ggml_backend_cuda_context & ctx, const ggml_tensor * gr, ggml_tensor * mm) {
    const ggml_tensor * k = gr->src[0], * ids = gr->src[1], * q = mm->src[1];
    const int n_rows = (int) gr->ne[1];
    {
        // GGML_CUDA_QSA_DEBUG=1 (debug, with GGML_CUDA_DISABLE_GRAPHS=1): check every launch on the host (index range vs
        // the key rows, output extent vs the score buffer) and abort with the details on a violation
        static const bool dbg = [] { const char * e = getenv("GGML_CUDA_QSA_DEBUG"); return e && atoi(e) != 0; }();
        if (dbg) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            std::vector<int32_t> h(n_rows);
            CUDA_CHECK(cudaMemcpy(h.data(), ids->data, n_rows*sizeof(int32_t), cudaMemcpyDeviceToHost));
            int32_t mx = INT32_MIN, mn = INT32_MAX;
            for (int32_t v : h) { mx = std::max(mx, v); mn = std::min(mn, v); }
            const ggml_tensor * kv = k->view_src ? k->view_src : k;
            const char * kb = (const char *) ggml_backend_buffer_get_base(kv->buffer);
            const size_t kbuf = ggml_backend_buffer_get_size(kv->buffer);
            const size_t k_end = (size_t) ((const char *) k->data - kb) + (size_t) std::max(mx, 0)*k->nb[1] + 256;
            const ggml_tensor * mv = mm->view_src ? mm->view_src : mm;
            const char * mb = (const char *) ggml_backend_buffer_get_base(mv->buffer);
            const size_t mbuf = ggml_backend_buffer_get_size(mv->buffer);
            const size_t m_end = (size_t) ((const char *) mm->data - mb) + ((size_t) (n_rows - 1) + (size_t) (q->ne[1] - 1)*(mm->nb[1]/sizeof(float)) + 1)*sizeof(float);
            const ggml_tensor * qv = q->view_src ? q->view_src : q;
            const size_t q_end = (size_t) ((const char *) q->data - (const char *) ggml_backend_buffer_get_base(qv->buffer)) + (size_t) (q->ne[1]-1)*q->nb[1] + 128*sizeof(float);
            const size_t qbuf = ggml_backend_buffer_get_size(qv->buffer);
            const bool bad = mn < 0 || k_end > kbuf || m_end > mbuf || q_end > qbuf || mx >= kv->ne[1];
            static int n_calls = 0;
            ++n_calls;
            if (bad || n_calls <= 2) {
                fprintf(stderr, "qsa_chk call %d dev %d %s: rows %d ids [%d,%d] | k %s ne1 %lld nb1 %zu end %zu / buf %zu | mm %s ne [%lld,%lld] nb1 %zu end %zu / buf %zu | q %s ne [%lld,%lld] nb1 %zu end %zu / buf %zu\n",
                    n_calls, ctx.device, bad ? "VIOLATION" : "ok", n_rows, mn, mx, kv->name, (long long) kv->ne[1], k->nb[1], k_end, kbuf,
                    mm->name, (long long) mm->ne[0], (long long) mm->ne[1], mm->nb[1], m_end, mbuf,
                    q->name, (long long) q->ne[0], (long long) q->ne[1], q->nb[1], q_end, qbuf);
            }
            if (bad) {
                GGML_ABORT("qsa_gather_score: out-of-bounds launch");
            }
        }
    }
    // ~4 rows per 16-lane group (2 rows in flight per step): 64K-cell context, 16384 rows: 18 us (16 rows: 26 us, 1: 24 us)
    const int nblk = (int) std::min<int64_t>(4096, std::max<int64_t>(1, ((int64_t) n_rows + 63)/64));
    const int64_t q_s1 = q->nb[1]/sizeof(float), d_s1 = mm->nb[1]/sizeof(float);
#define QSA_GS(N_) qsa_gather_score<N_><<<nblk, 256, 0, ctx.stream()>>>((const char *) k->data, (const int32_t *) ids->data, \
        (const float *) q->data, (float *) mm->data, n_rows, (int64_t) k->nb[1], q_s1, d_s1)
    switch (q->ne[1]) {
        case 1: QSA_GS(1); break;
        case 2: QSA_GS(2); break;
        case 3: QSA_GS(3); break;
        case 4: QSA_GS(4); break;
        case 5: QSA_GS(5); break;
        case 6: QSA_GS(6); break;
        case 7: QSA_GS(7); break;
        case 8: QSA_GS(8); break;
        // the MTP verify: 4 indexer heads x 3 (4) tokens; each column is summed exactly as with fewer columns
        case 12: QSA_GS(12); break;
        case 16: QSA_GS(16); break;
        default: GGML_ABORT("qsa_gather_score: %d columns", (int) q->ne[1]);
    }
#undef QSA_GS
    CUDA_CHECK(cudaGetLastError());
}
