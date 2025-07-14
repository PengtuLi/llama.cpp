#include "ggml.h"
#include "common.cuh"
#include "mmb_sparse.cuh"

// GTODO: we have not tested the kernel so far, test it when batch-example could be run
template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_batch_sparse(
        const T * __restrict__ x, 
        const float * __restrict__ y, 
        const float *  __restrict__ sparse_idx, 
        const int32_t *  __restrict__ gpu_neu_idx,
        float * __restrict__ dst,

        const int64_t nrows,
        const int64_t ncols2, 
        const int64_t stride_row,  // ncols
        const int64_t src1_ncols  // token batch number
        ) {
            
    const int64_t row         = blockIdx.x;  // (0, nrows)
    const int64_t s1col_b     = blockIdx.y;   // (0, scr1_ncols) the block that responsible for the specific token in batch
    const int     tid         = threadIdx.x; // (0, 256)
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    int gpu_neu = gpu_neu_idx ? gpu_neu_idx[row] : row; // (one of the gpu_neurons index)

    x          += stride_row * row;
    y          += stride_row * s1col_b;
    dst        += nrows * s1col_b;
    sparse_idx += nrows * s1col_b;

    if(sparse_idx[gpu_neu] < 0.5){ // GTODO: do we need sparse_threshold?
        //printf("sparse: %d: %f\n", gpu_neu,sparse_idx[gpu_neu]);
        return;
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf = 0.0f;

    if constexpr (std::is_same<T, float>::value) {
        const float2 * x2 = (const float2 *) x;

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += tmpx.x*tmpy.x;
            sumf += tmpx.y*tmpy.y;
        }
    } else if constexpr (std::is_same<T, half>::value) {
        const half2 * x2 = (const half2 *) x;

        if (std::is_same<type_acc, float>::value) {
            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                const float2 tmpy = y2[col2];
                sumf += tmpx.x * tmpy.x;
                sumf += tmpx.y * tmpy.y;
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2 = make_half2(0.0f, 0.0f);

            for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmp = y2[col2];
                sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
            }

            sumf = __low2float(sumh2) + __high2float(sumh2);
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        const int * x2 = (const int *) x;
        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const int    tmpx = x2[col2];
            const float2 tmpy = y2[col2];
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
            sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

    sumf = warp_reduce_sum<warp_size>(sumf);

    if (block_size > warp_size) {
        buf_iw[tid/warp_size] = sumf;
        __syncthreads();
        if (tid >= warp_size) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<warp_size>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[gpu_neu] = sumf;
}

template <typename T, typename type_acc>
static void launch_mul_mat_batch_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int32_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t src1_ncols, cudaStream_t stream) {
    GGML_ASSERT(ncols      % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);

    int device;
    int warp_size;

    CUDA_CHECK(cudaGetDevice(&device));
    warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {   // GTODO: we need to optimize best blocksize in batch-compute?
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const int smem = warp_size*sizeof(float);
    const dim3 block_nums(nrows, src1_ncols, 1); // (neurons_num, src1_ncols, 1)
    const dim3 block_dims(block_size_best, 1, 1); // (256, 1 ,1)
    
    switch (block_size_best) { // 256
        case   32: {
            mul_mat_batch_sparse<T, type_acc,  32><<<block_nums, block_dims, smem, stream>>>
                (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case   64: {
            mul_mat_batch_sparse<T, type_acc,  64><<<block_nums, block_dims, smem, stream>>>
                (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case   96: {
            mul_mat_batch_sparse<T, type_acc,  96><<<block_nums, block_dims, smem, stream>>>
                (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case  128: {
            mul_mat_batch_sparse<T, type_acc, 128><<<block_nums, block_dims, smem, stream>>>
                (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case  160: {
            mul_mat_batch_sparse<T, type_acc, 160><<<block_nums, block_dims, smem, stream>>>
               (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case  192: {
            mul_mat_batch_sparse<T, type_acc, 192><<<block_nums, block_dims, smem, stream>>>
               (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case  224: {
            mul_mat_batch_sparse<T, type_acc, 224><<<block_nums, block_dims, smem, stream>>>
               (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        case  256: {
            mul_mat_batch_sparse<T, type_acc, 256><<<block_nums, block_dims, smem, stream>>>
                (x, y, sparse_idx, gpu_neu_idx, dst, nrows, ncols/2, stride_row, src1_ncols);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template<typename T>
static void mul_mat_batch_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int32_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t src1_ncols,
        enum ggml_prec prec, cudaStream_t stream) {
    if constexpr(std::is_same<T, half>::value) {
        if (prec == GGML_PREC_DEFAULT) {
            launch_mul_mat_batch_cuda_sparse<T, half>
                (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, stride_row, src1_ncols, stream);
            return;
        }
    }
    launch_mul_mat_batch_cuda_sparse<T, float>
        (x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, stride_row, src1_ncols, stream);
}

void * ggml_cuda_get_tensor_data_batch(const ggml_tensor * tensor) {
    return tensor->data;
    // if (!tensor) {
    //     printf("no tensor, %s\n",tensor->name);
    //     GGML_ASSERT(false && "tensor is null");
    //     return nullptr;
    // }
    // if (!tensor->extra) {
    //     printf("no tensor-extra, %s\n",tensor->name);
    //     GGML_ASSERT(false && "tensor->extra is null");
    //     return nullptr;
    // }
    // int device = ggml_cuda_get_device();
    // auto extra = (ggml_tensor_extra_gpu *) tensor->extra;

    // if(tensor->data)
    // return extra->data_device[device];
}


void ggml_cuda_op_mul_mat_batch_sparse(
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

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(ggml_cuda_get_tensor_data_batch(dst->src[2])!=nullptr  && "missing sparse_idx");

    float * sparse_idx = static_cast<float *>(ggml_cuda_get_tensor_data_batch(dst->src[2]));
    int32_t * gpu_neu_idx = dst->src[3] != NULL ? static_cast<int32_t *>(ggml_cuda_get_tensor_data_batch(dst->src[3])) : NULL;

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const int64_t stride_row  = ne00;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_batch_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ne00, row_diff, stride_row, src1_ncols, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_batch_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ne00, row_diff, stride_row, src1_ncols, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_batch_cuda_sparse(src0_d, src1_ddf_i, sparse_idx, gpu_neu_idx, dst_dd_i, ne00, row_diff, stride_row, src1_ncols, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
