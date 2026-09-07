/* 并行归并排序（MPI）
 *
 * 流程：
 *   1. rank 0 生成全局数组，计算各进程分量的 counts/displs
 *   2. MPI_Scatterv 将数据均分给所有进程（处理不整除的余数）
 *   3. 每个进程对本地段做串行排序（qsort）
 *   4. 归并树：step = 1, 2, 4, ...
 *      - rank % (2*step) == step 的进程把整段有序数据发给 rank - step
 *      - rank % (2*step) == 0 且邻居存在的进程接收并二路归并
 *   5. rank 0 得到全局有序数组，校验有序性和校验和
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static void merge(const int *a, int na, const int *b, int nb, int *out) {
    int i = 0, j = 0, k = 0;
    while (i < na && j < nb)
        out[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = (argc > 1) ? atoll(argv[1]) : (1LL << 26);
    if (N < size) {
        if (rank == 0) fprintf(stderr, "N=%lld 小于进程数 %d\n", N, size);
        MPI_Finalize();
        return 1;
    }

    long long base = N / size, rem = N % size;
    int count = (int)(base + (rank < rem ? 1 : 0));

    int *buf = NULL;
    long long checksum = 0;

    double t_start = MPI_Wtime();

    if (rank == 0) {
        srand(42);
        int *data = malloc((size_t)N * sizeof(int));
        if (!data) { fprintf(stderr, "malloc 失败\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (long long i = 0; i < N; i++) {
            data[i] = rand();
            checksum += data[i];
        }
        int *counts = malloc(size * sizeof(int));
        int *displs = malloc(size * sizeof(int));
        if (!counts || !displs) { fprintf(stderr, "malloc 失败\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < size; i++) {
            counts[i] = (int)(base + (i < rem ? 1 : 0));
            displs[i] = (int)(i * base + (i < rem ? i : rem));
        }
        MPI_Scatterv(data, counts, displs, MPI_INT,
                     buf = malloc((size_t)count * sizeof(int)), count, MPI_INT,
                     0, MPI_COMM_WORLD);
        free(data); free(counts); free(displs);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     buf = malloc((size_t)count * sizeof(int)), count, MPI_INT,
                     0, MPI_COMM_WORLD);
    }

    double t_sort0 = MPI_Wtime();
    qsort(buf, count, sizeof(int), cmp_int);
    double t_sort1 = MPI_Wtime();

    double t_merge0 = MPI_Wtime();
    for (int step = 1; step < size; step <<= 1) {
        int r = rank % (2 * step);
        if (r == step) {
            MPI_Send(buf, count, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        } else if (r == 0 && rank + step < size) {
            int nbr = rank + step;
            long long nbase = N / size, nrem = N % size;
            int hi = (nbr + step < size) ? nbr + step : size;
            int ncount = 0;
            for (int i = nbr; i < hi; i++)
                ncount += (int)(nbase + (i < nrem ? 1 : 0));
            int *tmp = malloc((size_t)ncount * sizeof(int));
            int *merged = malloc((size_t)(count + ncount) * sizeof(int));
            if (!tmp || !merged) { fprintf(stderr, "rank %d malloc 失败\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
            MPI_Recv(tmp, ncount, MPI_INT, nbr, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merge(buf, count, tmp, ncount, merged);
            free(buf); free(tmp);
            buf = merged;
            count += ncount;
        }
    }
    double t_merge1 = MPI_Wtime();

    double local_sort = t_sort1 - t_sort0;
    double local_merge = t_merge1 - t_merge0;
    double max_sort = 0, max_merge = 0;
    MPI_Reduce(&local_sort, &max_sort, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_merge, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total1 = MPI_Wtime();

    if (rank == 0) {
        int ok = 1;
        for (long long i = 1; i < N; i++)
            if (buf[i - 1] > buf[i]) { ok = 0; break; }
        long long final_sum = 0;
        for (long long i = 0; i < N; i++) final_sum += buf[i];
        if (final_sum != checksum) ok = 0;

        printf("RESULT n=%lld np=%d sort=%.4f merge=%.4f total=%.4f valid=%d\n",
               N, size, max_sort, max_merge, t_total1 - t_start, ok);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
