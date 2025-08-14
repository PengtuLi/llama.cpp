/* test_axpy_sparse.cu

nvcc -O3 -arch=sm_80 axpy_kernels_test.cu -o axpy_kernels_test
./axpy_kernels_test

*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
typedef float dfloat; 
typedef float2 dfloat2;
float thresh = 0.5f;

// kernels from powerinfer
static __device__ void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const half * x = (const half *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x = x[ib + iqs + 0];
    v.y = x[ib + iqs + 1];
}

extern "C" static __global__ void powerinfer_axpy_sparse(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows, int64_t *lst, float *idx) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int gpu_row = blockIdx.y*blockDim.y + threadIdx.y; // range from [0,nrows]
    int qk = 1;
    int qr = 1;
    float dev_sparse_threshold = 0.5f;
    int GGML_CUDA_DMMV_X = 32;
    int GGML_CUDA_DMMV_Y = 1;
    int WARP_SIZE = 32;

    if (gpu_row >= nrows) {
        return;
    }
    int row = lst ? lst[gpu_row] : gpu_row;
    const int tid = threadIdx.x; // range from [0,31]
    short *d = (short *)((char *)vx + ncols * gpu_row * 2);

    if (y[row] == 0)
        return;
    if (idx[row] < dev_sparse_threshold) {
        return;
    }

    const int bid = blockIdx.y; // unused, wtf is this

    extern __shared__ float shared_dst[]; // TODO:dynamic

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X) {
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
            shared_dst[iybs + iqs + j/qr + 0] = tmp;  // share_dst[col] = tmp
            tmp = v.y * y[row];
            shared_dst[iybs + iqs + j/qr + y_offset] = tmp; // share_dst[col+1] = tmp
            
        }
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += GGML_CUDA_DMMV_X) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

// sparkinfer kernels
extern "C" static __global__ void mul_mat_axpy_sparse(
    const void * __restrict__ vx, 
    const dfloat * __restrict__ y, 
    float * __restrict__ dst, 
    
    const int ncols, 
    const int nrows, 
    
    const int64_t   * gpu_neu_idx, 
    const float     * sparse_idx
    ) {

    const int blk_idx = blockIdx.x;          // block index, range from [0,nrows]
    const int thds_per_blk = blockDim.x;     // number of threads per block

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
    const int tid = threadIdx.x; // range from [0,31]

    float alpha = y[neu];

    if (fabsf(alpha) < 1e-6f || sparse_idx[neu] < 0.5f) {
        // if (tid == 0) dst[gpu_neu] = 0.0f;
        return;
    }

    extern __shared__ float shared_dst[]; // TODO:dynamic
    
    const int VALS_PER_ITER = 2;   // each iter compute 2 vals consequently, we should not modify this
    const int   iter_stride = VALS_PER_ITER * thds_per_blk;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += thds_per_blk) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col  = i + VALS_PER_ITER*tid;
        const int vx_i = blk_idx*ncols + col; // vx index, vx was store in "blk_idx way", so indice it with blk_idx

        dfloat2 v;
        const half * x = (const half *) vx;
        v.x = x[vx_i + 0];
        v.y = x[vx_i + 1];

        // matrix multiplication, process 2 vals per j iter
        tmp = v.x * alpha;
        shared_dst[col] = tmp;  // share_dst[col] = tmp
        tmp = v.y * alpha;
        shared_dst[col+1] = tmp; // share_dst[col+1] = tmp       
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += thds_per_blk) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

extern "C" static __global__ void mul_mat_axpy_sparse_batch(
    const void * __restrict__ vx, 
    const dfloat * __restrict__ y, 
    float * __restrict__ dst, 
    
    const int ncols, 
    const int nrows, 
    
    const int64_t   * gpu_neu_idx, 
    const float     * sparse_idx
    ) {

    const int blk_idx   = blockIdx.x;          // block index, range from [0, nrows]
    const int token_idx = blockIdx.y;         // parallel input index, range from [0, src1_ncols]
    const int thds_per_blk = blockDim.x;     // number of threads per block

    y           += token_idx * nrows;
    dst         += token_idx * ncols;
    sparse_idx  += token_idx * nrows;

    const int neu = gpu_neu_idx ? gpu_neu_idx[blk_idx] : blk_idx;
    const int tid = threadIdx.x; // range from [0,31]

    float alpha = y[neu];

    if (fabsf(alpha) < 1e-6f || sparse_idx[neu] < 0.5f) {
        // if (tid == 0) dst[gpu_neu] = 0.0f;
        return;
    }

    extern __shared__ float shared_dst[]; // TODO:dynamic
    
    const int VALS_PER_ITER = 2;   // each iter compute 2 vals consequently, we should not modify this
    const int   iter_stride = VALS_PER_ITER * thds_per_blk;

// partial sum for each thread
    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += thds_per_blk) {
        shared_dst[i+tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col  = i + VALS_PER_ITER*tid;
        const int vx_i = blk_idx*ncols + col; // vx index, vx was store in "blk_idx way", so indice it with blk_idx

        dfloat2 v;
        const half * x = (const half *) vx;
        v.x = x[vx_i + 0];
        v.y = x[vx_i + 1];

        // matrix multiplication, process 2 vals per j iter
        tmp = v.x * alpha;
        shared_dst[col] = tmp;  // share_dst[col] = tmp
        tmp = v.y * alpha;
        shared_dst[col+1] = tmp; // share_dst[col+1] = tmp       
    }
    __syncthreads();

    for (int i = 0; i < ncols; i += thds_per_blk) {
        atomicAdd(&dst[i+tid], shared_dst[i+tid]);
    }
}

// CPU 参考实现
void reference_axpy(
    const std::vector<half>&    h_x,        // size = num_gpu_neurons * ncols
    const std::vector<float>&   h_y,        // size = n_tokens * nrows
    const std::vector<int64_t>& h_lst,      // size = num_gpu_neurons
    const std::vector<float>&   h_idx,      // size = n_tokens * nrows
    std::vector<float>&         h_ref,      // size = n_tokens * ncols

    int                         ncols,
    int                         nrows,
    int                         n_tokens,
    float                       threshold
) {
    std::fill(h_ref.begin(), h_ref.end(), 0.0f);
    int num_gpu_neurons = (int)h_lst.size();

    for (int t = 0; t < n_tokens; ++t) {
        const float* mask_ptr = h_idx.data() + t * nrows;
        const float* act_ptr  = h_y.data()   + t * nrows;
        float*       out_ptr  = h_ref.data() + t * ncols;

        for (int i = 0; i < num_gpu_neurons; ++i) {
            int neu  = (int)h_lst[i];
            float mask = mask_ptr[neu];
            float val  = act_ptr[neu];
            if (mask < threshold || fabsf(val) < 1e-6f) continue;

            const half* wptr = h_x.data() + neu * ncols;
            for (int c = 0; c < ncols; ++c) {
                float w = __half2float(wptr[c]);
                out_ptr[c] += w * val;
            }
        }
    }
}


void checkCorrectness() {
    // —— params ——
    const int ncols      = 4096;
    const int nrows      = 11008;
    const int n_tokens   = 4;
    const int dst_ne0    = ncols;
    const bool full_gpu  = false; // if true, offloaded all neurons onto gpu
    const float tol      = 1e-3f;
    const float hotcold_split = full_gpu ? 1.0f : 0.8f;
    const float hot_activation_ratio = 0.4f;
    const int num_gpu_neurons = static_cast<int>(nrows * hotcold_split);


    // initialize data on host
    std::vector<half>       h_vx(ncols * nrows);
    std::vector<float>      h_y(n_tokens * nrows);
    std::vector<int64_t>    h_lst(num_gpu_neurons);
    std::vector<int64_t>    h_mask(nrows, 0);
    std::vector<float>      h_idx(n_tokens * nrows);
    std::vector<float>      h_dst(n_tokens * ncols, 0.0f);
    std::vector<float>      h_ref(n_tokens * ncols, 0.0f);

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

    // init gpu_neu_mask
    for(int i = 0; i < num_gpu_neurons;i++){
        h_mask[h_lst[i]] = 1;
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
    // kernel launch
    {
        const dim3 grid(num_gpu_neurons, n_tokens);
        const dim3 block(32, 1);
        size_t shared_bytes = ncols * sizeof(float);
        mul_mat_axpy_sparse_batch<<<grid, block, shared_bytes>>>(
            d_vx, d_y, d_dst,
            ncols, nrows,
            d_lst, d_idx
        );
    }
    cudaDeviceSynchronize();

    // // powerinfer kernel
    // {
    //     const dim3 block_nums(1, num_gpu_neurons, 1); // (1, nrows, 1)
    //     const dim3 block_dims(32, 1, 1); // (32, 1, 1)
    //     powerinfer_axpy_sparse
    //     <<<block_nums, block_dims, ncols*sizeof(float)>>>(d_vx, d_y, d_dst, ncols, nrows, d_lst, d_idx);
    // }
    // cudaDeviceSynchronize();

    // copy results back to host
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size()*sizeof(float), cudaMemcpyDeviceToHost);

    // the reference results (computing by cpu) surely correct i think
    reference_axpy(h_vx, h_y, h_lst, h_idx, h_ref, ncols, nrows, n_tokens, thresh);


    // print the matrix if small
    if(nrows <= 64 && ncols <= 64){
        // input h_y
        printf("=== Matrix y (float) [%d x %d] ===\n", n_tokens, nrows);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < nrows; ++j) {
                printf("%6.1f ", h_y[i * nrows + j]);
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
                printf("%f ", __half2float(h_vx[i * ncols + j]));
            }
            printf("\n");
        }
        printf("\n");

        // sparse index
        printf("=== Matrix idx (float) [%d x %d] ===\n", n_tokens, nrows);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < nrows; ++j) {
                printf("%6.1f ", h_idx[i * nrows + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // compare the kernel output with the reference
    bool pass = true;
    printf("=== GPU Output vs CPU Reference ===\n");
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < dst_ne0; ++j) {
            int idx = i * dst_ne0 + j;
            float gpu_val = h_dst[idx];
            float ref_val = h_ref[idx];
            pass = fabs(gpu_val - ref_val) < tol;
            printf("[%2d,%2d] GPU: %8.3f | REF: %8.3f | Diff: %6.3f  %s\n", i, j, gpu_val, ref_val, fabs(gpu_val - ref_val), fabs(gpu_val - ref_val) < tol ? " " : "x");
        }
    }

    if (pass) {
        printf(">>> checkCorrectness PASSED <<<\n");
    }else printf(">>> FAILED <<<\n");

    // free
    cudaFree(d_vx);
    cudaFree(d_y);
    cudaFree(d_lst);
    cudaFree(d_idx);
    cudaFree(d_dst);
}

int main() {
    checkCorrectness();
}
