/*
    to test the correctness of mul_mat_sparse_kernels

nvcc -O3 -arch=sm_80 mul_mat_kernels_test.cu -o mul_mat_kernels_test
./mul_mat_kernels_test

*/
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

// vec
template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_vec_sparse(
        const T *       __restrict__ x, 
        const float *   __restrict__ y, 
        const float *   __restrict__ sparse_idx, 
        const int64_t * __restrict__ gpu_neu_idx,
        float *         __restrict__ dst,

        const int64_t   ncols2
) {
    const int64_t row         = blockIdx.x;  // (0, nrows)
    const int     tid         = threadIdx.x; // (0, 256)

    int neu = gpu_neu_idx ? gpu_neu_idx[row] : row; // (one of the neurons(on gpu) original index)
    
    if(sparse_idx[neu] < 0.5f){ // GTODO: do we need sparse_threshold?
        if (tid == 0) dst[neu] = 0.0f; // GTODO: this should be done in initialization. ps: outputs are different if we dont set 0 before return, meaning dst was not initialized as 0 at the beginning?
        return;
    }

    x += row*ncols2*2;

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;

    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
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
#endif // FP16_AVAILABLE
        }
    } else {
        static_assert(std::is_same<T, void>::value, "unsupported type");
    }

    sumf = warp_reduce_sum<WARP_SIZE>(sumf);

    if (block_size > WARP_SIZE) {
        buf_iw[tid/WARP_SIZE] = sumf;
        __syncthreads();
        if (tid >= WARP_SIZE) {
            return;
        }
        sumf = buf_iw[tid];
        sumf = warp_reduce_sum<WARP_SIZE>(sumf);
    }

    if (tid != 0) {
        return;
    }

    dst[neu] = sumf;
}

// batch
// GTODO: we have not tested the kernel so far, test it when batch-example could be run
template <typename T, typename type_acc, int block_size>
static __global__ void mul_mat_batch_sparse(
        const T *           __restrict__ x, 
        const float *       __restrict__ y, 
        const float *       __restrict__ sparse_idx, 
        const int64_t *     __restrict__ gpu_neu_idx,
        float *             __restrict__ dst,

        const int64_t ncols,
        const int64_t nrows,
        const int64_t src1_ncols  // token batch number
        ) {
    
    const int64_t ncols2      = ncols/2;

    const int64_t row         = blockIdx.x;  // (0, nrows)
    const int64_t s1col_b     = blockIdx.y;   // (0, scr1_ncols) the block that responsible for the specific token in batch
    const int     tid         = threadIdx.x; // (0, 256)

    constexpr int warp_size   = 32;

    int neu = gpu_neu_idx ? gpu_neu_idx[row] : row; // (one of the gpu_neurons index)

    x          += ncols * row;
    y          += ncols * s1col_b;
    dst        += nrows * s1col_b;
    sparse_idx += nrows * s1col_b;

    // we have ensure the cuda memory error will happen below

    // if(tid == 0) printf("row=%d ready for sparse_idx[%d]=%f\n",row, neu, sparse_idx[neu]);
    if(sparse_idx[neu] < 0.5f){ // GTODO: do we need sparse_threshold?
        // if(tid == 0) printf("row=%d in sparse_idx[%d]=%f\n",row, neu, sparse_idx[neu]);  
        if (tid == 0) dst[neu] = 0.0f;
        return;
    }

    // we have ensure the cuda memory error will happen above

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
#endif // FP16_AVAILABLE
        }
    }else {
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

    dst[neu] = sumf;
}

template <typename T, typename type_acc>
static void launch_mul_mat_cuda_sparse(
        const T * x, const float * y, const float * sparse_idx, const int64_t * gpu_neu_idx, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t src1_ncols, int64_t num_gpu_neurons, cudaStream_t stream) {

    // print<<<1, 32, 0, stream>>>(sparse_idx, gpu_neu_idx, ncols, nrows, src1_ncols);
    
    assert(ncols % 2 == 0);
    int device;
    cudaGetDevice(&device);

    int64_t block_size_best = WARP_SIZE;
    int64_t niter_best      = (ncols + 2*WARP_SIZE - 1) / (2*WARP_SIZE);
    int64_t max_block_size  = 256;

    // GTODO: understand why we choose block_size like this, do we need to change this in sparse inference?
    for (int64_t block_size = 2*WARP_SIZE; block_size <= max_block_size; block_size += WARP_SIZE) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    // Shared memory size
    const int smem = WARP_SIZE * sizeof(float);

    if (src1_ncols == 1) {
        // vector case
        dim3 grid(nrows, 1, 1);
        switch (block_size_best) {
            case 32:
                mul_mat_vec_sparse<T,type_acc,32><<<grid, dim3(32,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 64:
                mul_mat_vec_sparse<T,type_acc,64><<<grid, dim3(64,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 96:
                mul_mat_vec_sparse<T,type_acc,96><<<grid, dim3(96,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 128:
                mul_mat_vec_sparse<T,type_acc,128><<<grid, dim3(128,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 160:
                mul_mat_vec_sparse<T,type_acc,160><<<grid, dim3(160,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 192:
                mul_mat_vec_sparse<T,type_acc,192><<<grid, dim3(192,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 224:
                mul_mat_vec_sparse<T,type_acc,224><<<grid, dim3(224,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
            case 256:
            default:
                mul_mat_vec_sparse<T,type_acc,256><<<grid, dim3(256,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols/2);
                break;
        }
    } else {
        // Batch case
        dim3 grid(nrows, src1_ncols, 1);
        switch (block_size_best) {
            case 32:
                mul_mat_batch_sparse<T,type_acc,32><<<grid, dim3(32,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 64:
                mul_mat_batch_sparse<T,type_acc,64><<<grid, dim3(64,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 96:
                mul_mat_batch_sparse<T,type_acc,96><<<grid, dim3(96,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 128:
                mul_mat_batch_sparse<T,type_acc,128><<<grid, dim3(128,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 160:
                mul_mat_batch_sparse<T,type_acc,160><<<grid, dim3(160,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 192:
                mul_mat_batch_sparse<T,type_acc,192><<<grid, dim3(192,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 224:
                mul_mat_batch_sparse<T,type_acc,224><<<grid, dim3(224,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
            case 256:
            default:
                mul_mat_batch_sparse<T,type_acc,256><<<grid, dim3(256,1,1), smem, stream>>>(
                    x, y, sparse_idx, gpu_neu_idx, dst, ncols, nrows, src1_ncols);
                break;
        }
    }
}

// CPU reference for mul_mat sparse
void reference_mul_mat(
        const half  * x,           // [nrows][ncols]
        const float * y,           // [ntokens][ncols]
        const float * idx,        // [ntokens][nrows]
        const int64_t * lst,       // [nrows] or nullptr
        float * dst,               // [ntokens][nrows]
        int64_t nrows,
        int64_t ncols,
        int64_t num_gpu_neurons,
        int64_t n_tokens
) {
    for (int t = 0; t < n_tokens; ++t) {
        for (int r_gpu = 0; r_gpu < num_gpu_neurons; ++r_gpu) { // num_neurons
            int row = lst[r_gpu];
            if (idx[t * nrows + row] < 0.5f) {
                continue;
            }
            // simpified version of convert_f16:
            float acc = 0.0f;
            for (int c = 0; c < ncols; c += 2) {
                float v0 = __half2float(x[row * ncols + c + 0]);
                float v1 = __half2float(x[row * ncols + c + 1]);
                float y0 = y[t * ncols + c + 0];
                float y1 = y[t * ncols + c + 1];
                acc += v0 * y0 + v1 * y1;
            }
            dst[t * nrows + row] = acc;
        }
    }
}
void check() {
    // —— params ——
    const int ncols      = 4096;
    const int nrows      = 11008;
    const int n_tokens   = 4;
    const int dst_ne0    = nrows;
    const bool full_gpu  = false; // if true, offloaded all neurons onto gpu
    const float tol      = 1e-3f;
    const float hotcold_split = full_gpu ? 1.0f : 0.8f;
    const float hot_activation_ratio = 0.4f;
    const int num_gpu_neurons = static_cast<int>(nrows * hotcold_split);


    // initialize data on host
    std::vector<half>       h_vx(ncols * nrows);
    std::vector<float>      h_y(n_tokens * ncols);
    std::vector<int64_t>    h_lst(num_gpu_neurons);
    std::vector<float>      h_idx(n_tokens * nrows);
    std::vector<float>      h_dst(n_tokens * nrows, 0.0f);
    std::vector<float>      h_ref(n_tokens * nrows, 0.0f);

    // // simple initialize
    // for (int i = 0; i < nrows * ncols; ++i) {
    //     h_vx[i] = __float2half(float(i % 10 + 1));
    // }
    // for (int i = 0; i < src1_ncols * ncols; ++i) {
    //     h_y[i] = 1.0f;
    // }
    // for (int i = 0; i < num_gpu_neurons; ++i) {
    //     h_lst[i] = i;
    // }
    // for (int i = 0; i < src1_ncols * dst_ne0; ++i) {
    //     h_idx[i] = 1.0f;
    // }

    // more complex init
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : h_vx) v = __float2half(dist(rng));
    for (auto &v : h_y)  v = dist(rng);

    // init gpu_bucket ( lst)
    std::vector<int64_t> gpu_bucket(nrows);
    std::iota(gpu_bucket.begin(), gpu_bucket.end(), 0);  // [0, 1, 2, ..., nrows-1]
    if(!full_gpu) std::shuffle(gpu_bucket.begin(), gpu_bucket.end(), rng);
    for (int i = 0; i < num_gpu_neurons; ++i) {
        h_lst[i] = gpu_bucket[i];
    }

    // initialize sparse index
    for (int i = 0; i < n_tokens * nrows; ++i) {
        h_idx[i] = (dist(rng) < hot_activation_ratio) ? 1.0f : 0.0f;
    }

    // malloc & copy data to device
    half    *d_vx;   cudaMalloc(&d_vx,   num_gpu_neurons * ncols * sizeof(half));
    float   *d_y;    cudaMalloc(&d_y,    h_y.size()    * sizeof(float));
    int64_t *d_lst;  cudaMalloc(&d_lst,  h_lst.size()  * sizeof(int64_t));
    float   *d_idx;  cudaMalloc(&d_idx,  h_idx.size()  * sizeof(float));
    float   *d_dst;  cudaMalloc(&d_dst,  h_dst.size()* sizeof(float));

    // *** powerinfer type neurons loading
    const int single_nueron_size = ncols * sizeof(half);
    half* d_vx_base = d_vx;
    for(int i = 0; i < num_gpu_neurons; i++){
        int neuron_index = h_lst[i];
        const half* src = h_vx.data() + neuron_index * ncols;
        half* dst = d_vx_base + i * ncols;
        cudaMemcpy(dst, src, single_nueron_size, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_y,  h_y.data(),  h_y.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lst,h_lst.data(),h_lst.size()*sizeof(int64_t),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx,h_idx.data(),h_idx.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dst, 0,         h_dst.size()*sizeof(float));

    if(full_gpu) d_lst = nullptr;

    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    launch_mul_mat_cuda_sparse<half, float>
                (d_vx, d_y, d_idx, d_lst, d_dst, ncols, nrows, n_tokens, num_gpu_neurons, stream);

    cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size()*sizeof(float), cudaMemcpyDeviceToHost);

    reference_mul_mat(h_vx.data(), h_y.data(), h_idx.data(), h_lst.data(), h_ref.data(), nrows, ncols, num_gpu_neurons, n_tokens);

    // print the matrix if small
    if(nrows <= 64 && ncols <= 64){
        // input h_y
        printf("=== Matrix y (float) [%d x %d] ===\n", n_tokens, ncols);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < ncols; ++j) {
                printf("%6.1f ", h_y[i * ncols + j]);
            }
            printf("\n");
        }
        printf("\n");

        // gpu_bucket (lst)
        printf("index sequence of gpu_neurons: ");
        for(int i = 0;i < num_gpu_neurons;i++){
            printf("%d ",h_lst[i]);
        }
        printf("\n");

        // the weights h_vx
        printf("=== Matrix vx (half) [%d x %d] ===\n", num_gpu_neurons, ncols);
        for (int i = 0; i < num_gpu_neurons; ++i) {
            for (int j = 0; j < ncols; ++j) {
                printf("%6.1f ", __half2float(h_vx[i * ncols + j]));
            }
            printf("\n");
        }
        printf("\n");

        // sparse index
        printf("=== Matrix idx (float) [%d x %d] ===\n", n_tokens, dst_ne0);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < dst_ne0; ++j) {
                printf("%6.1f ", h_idx[i * dst_ne0 + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // compare the kernel output with the reference
    printf("=== GPU Output vs CPU Reference ===\n");
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < dst_ne0; ++j) {
            int idx = i * dst_ne0 + j;
            float gpu_val = h_dst[idx];
            float ref_val = h_ref[idx];
            printf("[%2d,%2d] GPU: %8.3f | REF: %8.3f | Diff: %6.3f  %s\n", i, j, gpu_val, ref_val, fabs(gpu_val - ref_val), fabs(gpu_val - ref_val) < tol ? " " : "x");
        }
    }

    // final result
    bool pass = true;
    for (int i = 0; i < n_tokens * dst_ne0; ++i) {
        float a = h_dst[i], b = h_ref[i];
        if (std::fabs(a - b) > tol) {
            printf(">>> FAILED <<<\n");
            pass = false;
            break;
        }
    }
    if (pass) {
        printf(">>> checkCorrectness PASSED <<<\n");
    }

    // free
    cudaFree(d_vx);
    cudaFree(d_y);
    cudaFree(d_lst);
    cudaFree(d_idx);
    cudaFree(d_dst);
}

int main() {
    // check_mat_vec();
    check();
    return 0;
}
