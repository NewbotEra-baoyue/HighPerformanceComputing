#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

int main()
{
    const int N = 1 << 20;  // 约 104 万元素，与 kernel 版本一致
    const int iters = 100;

    // 初始化主机数据
    thrust::host_vector<float> h_a(N, 1.0f);
    thrust::host_vector<float> h_b(N, 2.0f);

    // ---- CPU 串行版本（计时对比，取 100 次平均） ----
    thrust::host_vector<float> h_c_cpu(N);
    for (int i = 0; i < N; ++i) {  // 热身
        h_c_cpu[i] = h_a[i] + h_b[i];
    }
    double cpu_total = 0.0;
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            h_c_cpu[i] = h_a[i] + h_b[i];
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double cpu_ms = cpu_total / iters;

    // 拷贝到 GPU
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_c(N);

    // ---- GPU 版本：一行调用 ----
    // 先热身一次，摊掉首次调用的上下文初始化开销
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                      thrust::plus<float>());
    cudaDeviceSynchronize();

    double gpu_total = 0.0;
    for (int it = 0; it < iters; ++it) {
        auto t4 = std::chrono::high_resolution_clock::now();
        thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                          thrust::plus<float>());
        cudaDeviceSynchronize();
        auto t5 = std::chrono::high_resolution_clock::now();
        gpu_total += std::chrono::duration<double, std::milli>(t5 - t4).count();
    }
    double gpu_ms = gpu_total / iters;

    // 数据传输计时（热身后单独测一次拷贝）
    auto t2 = std::chrono::high_resolution_clock::now();
    thrust::copy(h_a.begin(), h_a.end(), d_a.begin());
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double h2d_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 拷回主机并验证
    thrust::host_vector<float> h_c = d_c;
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_c[i] - 3.0f) > 1e-5f) {
            ++errors;
        }
    }
    printf("%s (N = %d, errors = %d)\n",
           errors == 0 ? "Result is correct" : "Result is WRONG", N, errors);
    printf("CPU serial add : %8.3f ms\n", cpu_ms);
    printf("GPU add  : %8.3f ms  (不含数据传输)\n", gpu_ms);
    printf("Host->Device 拷贝: %8.3f ms\n", h2d_ms);
    printf("加速比 (CPU / GPU 计算) = %.1fx\n", cpu_ms / gpu_ms);

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
