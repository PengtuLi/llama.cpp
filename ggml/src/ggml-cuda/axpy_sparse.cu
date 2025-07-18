#include "ggml.h"
#include "common.cuh"
#include "mmv_sparse.cuh"

// GTODO: the two kernel are demo kernels for axpy, need to be optimized in the future...
// the powerinfer kernel: 
static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

static __global__ void dequantize_mul_mat_axpy_sparse(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, const int *lst, const float *idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y; // range from [0,nrows]
    int qk =1;
    int qr = 1;

    if (gpu_row >= nrows) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;
    const int tid = threadIdx.x; // range from [0,31]
    short *d = (short *)((char *)vx + ncols * gpu_row * 2);

    if (y[row] == 0)
        return;
    if (idx[row] < 0.5f) {
        return;
    }

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int iter_stride = 2*32;
    const int vals_per_iter = iter_stride / 32; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += 32) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (gpu_row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            convert_f16(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
            tmp = v.x * y[row];
            shared_dst[col] = tmp;  // share_dst[col] = tmp
            tmp = v.y * y[row];
            shared_dst[col+1] = tmp; // share_dst[col+1] = tmp
            
        }
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += 32) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

static __global__ void dequantize_mul_mat_axpy_sparse_batch(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, int src1_ne0, int src1_ncols, int *lst, float *idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y;
    int qk = 1;
    int qr = 1;

    if (gpu_row >= nrows) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;
    const int bid = blockIdx.y;

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int tid = threadIdx.x;

    const int iter_stride = 2*32;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;
    float * loop_idx = idx;
    dfloat * loop_y = (dfloat *)y;
    float * loop_dst = dst;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += 32) {
        shared_dst[i+tid] = 0;
    }
    // __syncthreads();
    for (int col_id = 0; col_id < src1_ncols; col_id++) {
        __syncthreads();
        if (loop_idx[row] < 0.5f) {
            loop_dst += ncols;
            loop_idx += src1_ne0;
            loop_y += src1_ne0;
            continue;
        }
        

        for (int i = 0; i < ncols; i += iter_stride)
        {
            const int col = i + vals_per_iter * tid;
            const int ib = (gpu_row * ncols + col) / qk; // x block index
            const int iqs = (col % qk) / qr;         // x quant index
            const int iybs = col - col % qk;         // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
            for (int j = 0; j < vals_per_iter; j += 2)
            {
                // process 2 vals per j iter

                // dequantize
                // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
                dfloat2 v;
                convert_f16(vx, ib, iqs + j / qr, v);

                // matrix multiplication
                // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
                tmp = v.x * loop_y[row];
                shared_dst[iybs + iqs + j / qr + 0] = tmp;
                tmp = v.y * loop_y[row];
                shared_dst[iybs + iqs + j / qr + y_offset] = tmp;
            }
        }
        /* __syncthreads(); */

        for (int i = 0; i < ncols; i += 32)
        {
            atomicAdd(&loop_dst[i + tid], shared_dst[i + tid]);
            shared_dst[i+tid] = 0;
        }
        loop_dst += ncols;
        loop_idx += src1_ne0;
        loop_y += src1_ne0;
    }
}


template <typename T, typename type_acc>
static void launch_mul_mat_axpy_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int32_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t src_ncols, cudaStream_t stream) {
    
    // vec_axpy
    if(src_ncols == 1){
        // the lanucher for powerinfer kernel: 
        const dim3 block_nums(1, nrows, 1);
        const dim3 block_dims(32, 1, 1);

        dequantize_mul_mat_axpy_sparse<<<block_nums, block_dims, ncols*sizeof(float), stream>>>(x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
    }
    else{ // batch_axpy
        const dim3 block_nums(1, nrows, 1);
        const dim3 block_dims(32, 1, 1);
        dequantize_mul_mat_axpy_sparse<<<block_nums, block_dims, ncols*sizeof(float), stream>>>(x, y, dst, ncols, nrows, gpu_neu_idx, sparse_idx);
        // GGML_ASSERT(false && "GTODO: batch axpy need to be done");
    }

}

template<typename T>
static void mul_mat_axpy_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int32_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t src_ncols,
        enum ggml_prec prec, cudaStream_t stream) {
    if constexpr(std::is_same<T, half>::value) {
        if (prec == GGML_PREC_DEFAULT) {
            launch_mul_mat_axpy_cuda_sparse<T, half>
                (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols, stream);
            return;
        }
    }
    launch_mul_mat_axpy_cuda_sparse<T, float>
        (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src_ncols, stream);
}

// GTODO: this is very hacky, we need to add more safety check later
// but more importantly, what's the diffence between tensor->data & tensor-extra->data_device[device]? which to load???
void * ggml_cuda_get_tensor_data_axpy(const ggml_tensor * tensor) {
    return tensor->data;
    // if (!tensor) {
    //     printf("no tensor, %s\n",tensor->name);
    //     GGML_ASSERT(false && "tensor is null");
    //     return nullptr;
    // }
    // if (!tensor->extra) {
    //     printf("no tensor-extra, %s\n",tensor->name); 
    //     GGML_ASSERT(false && "tensor->extra is null"); sparse_idx在这里会报错, saprse_idx is only at tensor->data 
    //     return nullptr;
    // }
    // int device = ggml_cuda_get_device();
    // auto extra = (ggml_tensor_extra_gpu *) tensor->extra;

    // if(tensor->data)
    // return extra->data_device[device];
}


void ggml_cuda_op_axpy_sparse(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, 
    const ggml_tensor * src1, 
    ggml_tensor *       dst, 

    const char *        src0_dd_i, 
    const float *       src1_ddf_i,
    const char *        src1_ddq_i, 
    float *             dst_dd_i, 

    const int64_t       row_low, 
    const int64_t       row_high, 
    const int64_t       src1_ncols,
    const int64_t       src1_padded_row_size, 
    
    cudaStream_t        stream) 
    {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = row_high - row_low;

    GGML_ASSERT(ggml_cuda_get_tensor_data_axpy(dst->src[2])!=nullptr  && "missing sparse_idx");

    float * sparse_idx = static_cast<float *>(ggml_cuda_get_tensor_data_axpy(dst->src[2]));
    int32_t * gpu_neu_idx = dst->src[3] != NULL ? static_cast<int32_t *>(ggml_cuda_get_tensor_data_axpy(dst->src[3])) : NULL;

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    void * src0_d = nullptr;
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    mul_mat_axpy_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ncols, nrows, src1_ncols, prec, stream);

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
