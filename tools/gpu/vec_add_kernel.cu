#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// 每个线程负责一个元素
__global__ void vecAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int N = 1 << 20;  // 约 104 万元素
    size_t bytes = N * sizeof(float);

    // 主机内存
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 设备内存
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 启动配置：block 256 线程，grid 覆盖所有元素
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 热身一次，摊掉首次调用的上下文初始化开销
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 用 CUDA Event 计时 kernel 执行，跑 100 次取平均
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float kernel_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start, stop));
    kernel_ms /= iters;
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    printf("GPU kernel avg : %8.3f ms  (vecAdd, %d iters, 不含数据传输)\n",
           kernel_ms, iters);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 主机端验证
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_c[i] - 3.0f) > 1e-5f) {
            if (errors < 5) {
                printf("Mismatch at %d: %f\n", i, h_c[i]);
            }
            ++errors;
        }
    }
    printf("%s (N = %d, errors = %d)\n",
           errors == 0 ? "Result is correct" : "Result is WRONG", N, errors);

    // 释放
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
