#include <stdio.h>
#include <omp.h>

int main() {
    long long n = 500000000;
    double step = 1.0 / (double)n;
    double pi_serial, pi_parallel;
    double t0, t_serial, t_parallel;

    printf("=== 数值积分计算 π ===\n");
    printf("公式: π = ∫₀¹ 4/(1+x²) dx\n");
    printf("迭代次数: %lld\n\n", n);

    /* -------- 串行版本 -------- */
    double sum_serial = 0.0;
    t0 = omp_get_wtime();
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * step;
        sum_serial += 4.0 / (1.0 + x * x);
    }
    pi_serial = sum_serial * step;
    t_serial = omp_get_wtime() - t0;

    printf("[串行] π ≈ %.15f, 耗时: %.3f 秒\n", pi_serial, t_serial);

    /* -------- OpenMP 并行版本 -------- */
    double sum_parallel = 0.0;
    t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum_parallel) schedule(static)
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * step;
        sum_parallel += 4.0 / (1.0 + x * x);
    }
    pi_parallel = sum_parallel * step;
    t_parallel = omp_get_wtime() - t0;

    printf("[并行] π ≈ %.15f, 耗时: %.3f 秒\n", pi_parallel, t_parallel);
    printf("理论值:  3.141592653589793\n");
    printf("加速比:  %.2f x (使用 %d 线程)\n", 
           t_serial / t_parallel, omp_get_max_threads());

    return 0;
}