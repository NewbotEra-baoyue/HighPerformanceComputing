#include <stdio.h>
#include <omp.h>
#include <time.h>

// 正确做法：串行执行
double good_serial_small_loop() {
    double sum = 0.0;
    int n = 100;
    
    double t0 = omp_get_wtime();
    
    for (int i = 0; i < n; i++) {
        sum += i * 0.5;
    }
    
    return omp_get_wtime() - t0;
}

// 错误示范：迭代次数少，线程开销大
double bad_openmp_small_loop() {
    double sum = 0.0;
    int n = 100;  // 只有100次迭代
    
    double t0 = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:sum)  // 创建线程开销巨大！
    for (int i = 0; i < n; i++) {
        sum += i * 0.5;
    }
    
    return omp_get_wtime() - t0;
}

int main() {

    double t_serial, t_openmp;
    t_serial = good_serial_small_loop();
    t_openmp = bad_openmp_small_loop();
    
    printf("普通耗时：%.9f，错误加速耗时：%.9f",t_serial, t_openmp);

    return 0;
}