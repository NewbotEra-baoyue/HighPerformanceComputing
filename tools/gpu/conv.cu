// CUDA 2D convolution benchmark: 15x15 kernel, 8192x8192 float image
// Optimized kernel: shared-memory tiling + 4x4 register blocking + float4 stores.
// Reports effective TFLOPS as a percentage of RTX 4070 FP32 peak (~29.2 TFLOPS).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define K 15
#define R (K / 2)
#define TILE_OUT 64
#define TILE_IN (TILE_OUT + K - 1)   // 78
#define BLOCK 16
#define THREADS (BLOCK * BLOCK)

static const int W = 8192;
static const int H = 8192;
static const int ITER = 20;
static const double PEAK_TFLOPS = 5888.0 * 2.0 * 2.475e-3;  // cores * FMA * boost clock

__constant__ float c_kernel[K * K];

#define CHECK_CUDA(x)                                                          \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(e), __FILE__, __LINE__);                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------- naive reference (one output per thread, direct global reads)
__global__ void conv_naive(const float *__restrict__ in, float *__restrict__ out,
                           int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.f;
    for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int gy = y - R + ky, gx = x - R + kx;
            if (gx >= 0 && gx < w && gy >= 0 && gy < h)
                sum += c_kernel[ky * K + kx] * in[gy * w + gx];
        }
    out[y * w + x] = sum;
}

// ---------------- optimized: 64x64 output tile per 256-thread block,
// each thread computes 4x4 outputs with a sliding register window over SMEM.
__global__ void __launch_bounds__(THREADS) conv_opt(const float *__restrict__ in,
                                                    float *__restrict__ out,
                                                    int w, int h) {
    __shared__ float tile[TILE_IN][TILE_IN + 1];  // +1 pad avoids bank conflicts
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BLOCK + tx;
    const int inOx = blockIdx.x * TILE_OUT - R;
    const int inOy = blockIdx.y * TILE_OUT - R;

    for (int i = tid; i < TILE_IN * TILE_IN; i += THREADS) {
        int r = i / TILE_IN, c = i % TILE_IN;
        int gy = inOy + r, gx = inOx + c;
        tile[r][c] = (gx >= 0 && gx < w && gy >= 0 && gy < h) ? in[gy * w + gx] : 0.f;
    }
    __syncthreads();

    const int ox = tx * 4;  // output column within tile
    const int oy = ty * 4;

    float acc[4][4] = {};
#pragma unroll
    for (int ky = 0; ky < K; ky++) {
        float v[4][4];
#pragma unroll
        for (int dy = 0; dy < 4; dy++)
#pragma unroll
            for (int dx = 0; dx < 4; dx++)
                v[dy][dx] = tile[oy + ky + dy][ox + dx];
#pragma unroll
        for (int kx = 0; kx < K; kx++) {
            float wgt = c_kernel[ky * K + kx];
#pragma unroll
            for (int dy = 0; dy < 4; dy++) {
                acc[dy][0] += wgt * v[dy][0];
                acc[dy][1] += wgt * v[dy][1];
                acc[dy][2] += wgt * v[dy][2];
                acc[dy][3] += wgt * v[dy][3];
            }
            if (kx < K - 1) {  // folds away after unrolling
#pragma unroll
                for (int dy = 0; dy < 4; dy++) {
                    v[dy][0] = v[dy][1];
                    v[dy][1] = v[dy][2];
                    v[dy][2] = v[dy][3];
                    v[dy][3] = tile[oy + ky + dy][ox + kx + 4];
                }
            }
        }
    }

    const int outOx = blockIdx.x * TILE_OUT + ox;
    const int outOy = blockIdx.y * TILE_OUT + oy;
#pragma unroll
    for (int dy = 0; dy < 4; dy++) {
        int gy = outOy + dy;
        if (gy < h) {
            float4 val = make_float4(acc[dy][0], acc[dy][1], acc[dy][2], acc[dy][3]);
            reinterpret_cast<float4 *>(out)[(gy * w + outOx) / 4] = val;
        }
    }
}

int main() {
    size_t n = (size_t)W * H;
    float *d_in, *d_out, *d_ref;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ref, n * sizeof(float)));

    float *h_in = (float *)malloc(n * sizeof(float));
    float h_ker[K * K];
    srand(42);
    for (size_t i = 0; i < n; i++) h_in[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * K; i++) h_ker[i] = (float)rand() / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_ker, sizeof(h_ker)));

    dim3 blkNaive(16, 16);
    dim3 grdNaive((W + 15) / 16, (H + 15) / 16);
    conv_naive<<<grdNaive, blkNaive>>>(d_in, d_ref, W, H);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 blkOpt(BLOCK, BLOCK);
    dim3 grdOpt(W / TILE_OUT, H / TILE_OUT);
    conv_opt<<<grdOpt, blkOpt>>>(d_in, d_out, W, H);  // warmup
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < ITER; i++)
        conv_opt<<<grdOpt, blkOpt>>>(d_in, d_out, W, H);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    ms /= ITER;

    // validation
    float *h_out = (float *)malloc(n * sizeof(float));
    float *h_ref = (float *)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ref, d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
    double maxErr = 0.0;
    for (size_t i = 0; i < n; i++)
        maxErr = fmax(maxErr, fabs((double)h_out[i] - h_ref[i]));

    double flops = 2.0 * (double)W * H * K * K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("image: %dx%d, kernel: %dx%d\n", W, H, K, K);
    printf("time:  %.3f ms/iter (%d iters)\n", ms, ITER);
    printf("perf:  %.2f TFLOPS  (FP32 peak %.2f TFLOPS)  ->  %.1f%% of peak\n",
           tflops, PEAK_TFLOPS, 100.0 * tflops / PEAK_TFLOPS);
    printf("check: max abs err = %.3e  %s\n", maxErr,
           maxErr < 1e-3 ? "PASS" : "FAIL");

    free(h_in); free(h_out); free(h_ref);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ref);
    return maxErr < 1e-3 ? 0 : 1;
}
