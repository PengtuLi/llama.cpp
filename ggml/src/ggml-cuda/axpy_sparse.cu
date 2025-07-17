#include "ggml.h"
#include "common.cuh"
#include "mmv_sparse.cuh"

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

// template <typename T, typename type_acc, int block_size>
// static __global__ void mul_mat_axpy_sparse(
//         const T * __restrict__ x, 
//         const float * __restrict__ y, 
//         const float *  __restrict__ sparse_idx, 
//         const int32_t *  __restrict__ gpu_neu_idx,
//         float * __restrict__ dst,

//         const int64_t ncols2, 
//         const int64_t stride_row
//         ) {
            
//     const int64_t row         = blockIdx.x;  // (0, nrows)
//     const int     tid         = threadIdx.x; // (0, 256)

//     int gpu_neu = gpu_neu_idx ? gpu_neu_idx[row] : row; // (one of the gpu_neurons index)
    
//     if(sparse_idx[gpu_neu] < 0.5f){ // GTODO: do we need sparse_threshold?
//         if (tid == 0) dst[gpu_neu] = 0.0f; // GTODO: this should be done in initialization. ps: outputs are different if we dont set 0 before return, meaning dst was not initialized as 0 at the beginning?
//         return;
//     }

//     constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

//     x += row*stride_row;

//     const float2 * y2 = (const float2 *) y;

//     extern __shared__ char data_mmv[];
//     float * buf_iw = (float *) data_mmv;

//     if (block_size > warp_size) {
//         if (tid < warp_size) {
//             buf_iw[tid] = 0.0f;
//         }
//         __syncthreads();
//     }

//     float sumf = 0.0f;

//     if constexpr (std::is_same<T, float>::value) {
//         const float2 * x2 = (const float2 *) x;

//         for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//             const float2 tmpx = x2[col2];
//             const float2 tmpy = y2[col2];
//             sumf += tmpx.x*tmpy.x;
//             sumf += tmpx.y*tmpy.y;
//         }
//     } else if constexpr (std::is_same<T, half>::value) {
//         const half2 * x2 = (const half2 *) x;

//         if (std::is_same<type_acc, float>::value) {
//             for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//                 const float2 tmpx = __half22float2(x2[col2]);
//                 const float2 tmpy = y2[col2];
//                 sumf += tmpx.x * tmpy.x;
//                 sumf += tmpx.y * tmpy.y;
//             }
//         } else {
// #ifdef FP16_AVAILABLE
//             half2 sumh2 = make_half2(0.0f, 0.0f);

//             for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//                 const float2 tmp = y2[col2];
//                 sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
//             }

//             sumf = __low2float(sumh2) + __high2float(sumh2);
// #else
//             NO_DEVICE_CODE;
// #endif // FP16_AVAILABLE
//         }
//     } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
//         const int * x2 = (const int *) x;
//         for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
//             const int    tmpx = x2[col2];
//             const float2 tmpy = y2[col2];
//             sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]) * tmpy.x;
//             sumf += float(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]) * tmpy.y;
//         }
//     } else {
//         static_assert(std::is_same<T, void>::value, "unsupported type");
//     }

//     sumf = warp_reduce_sum<warp_size>(sumf);

//     if (block_size > warp_size) {
//         buf_iw[tid/warp_size] = sumf;
//         __syncthreads();
//         if (tid >= warp_size) {
//             return;
//         }
//         sumf = buf_iw[tid];
//         sumf = warp_reduce_sum<warp_size>(sumf);
//     }

//     if (tid != 0) {
//         return;
//     }

//     dst[gpu_neu] = sumf;
// }

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
        
        // GGML_ASSERT(ncols      % 2 == 0);
        // GGML_ASSERT(stride_row % 2 == 0);

        // int device;
        // int warp_size;

        // CUDA_CHECK(cudaGetDevice(&device));
        // warp_size = ggml_cuda_info().devices[device].warp_size;

        // int64_t block_size_best = warp_size;
        // int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
        // int64_t max_block_size  = 256;
        // if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        //     max_block_size = 128;
        // }

        // // GTODO: understand why we choose block_size like this, do we need to change this in sparse inference?
        // for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        //     const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        //     if (niter < niter_best) {
        //         niter_best      = niter;
        //         block_size_best = block_size;
        //     }
        // }

        // const int smem = warp_size*sizeof(float);
        // const dim3 block_nums(nrows, 1, 1); // (neurons_num, 1, 1)
        // const dim3 block_dims(block_size_best, 1, 1); // (256, 1 ,1)
        
        // switch (block_size_best) { // 256
        //     case   32: {
        //         mul_mat_axpy_sparse<T, type_acc,  32><<<block_nums, block_dims, smem, stream>>>
        //             (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case   64: {
        //         mul_mat_axpy_sparse<T, type_acc,  64><<<block_nums, block_dims, smem, stream>>>
        //             (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case   96: {
        //         mul_mat_axpy_sparse<T, type_acc,  96><<<block_nums, block_dims, smem, stream>>>
        //             (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case  128: {
        //         mul_mat_axpy_sparse<T, type_acc, 128><<<block_nums, block_dims, smem, stream>>>
        //             (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case  160: {
        //         mul_mat_axpy_sparse<T, type_acc, 160><<<block_nums, block_dims, smem, stream>>>
        //            (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case  192: {
        //         mul_mat_axpy_sparse<T, type_acc, 192><<<block_nums, block_dims, smem, stream>>>
        //            (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case  224: {
        //         mul_mat_axpy_sparse<T, type_acc, 224><<<block_nums, block_dims, smem, stream>>>
        //            (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     case  256: {
        //         mul_mat_axpy_sparse<T, type_acc, 256><<<block_nums, block_dims, smem, stream>>>
        //             (x, y, sparse_idx, gpu_neu_idx, dst, ncols/2, stride_row);
        //     } break;
        //     default: {
        //         GGML_ABORT("fatal error");
        //     } break;
        // }
    }
    else{ // batch_axpy
        GGML_ASSERT(false && "GTODO: batch axpy need to be done");
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

    GGML_ASSERT(src0->ne[0] == src1->ne[0] && "src0->ne[0] != src1->ne[0]");
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
