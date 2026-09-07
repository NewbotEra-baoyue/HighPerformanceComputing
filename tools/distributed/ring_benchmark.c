#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int max_bytes = 1 << 20;
    char *sendbuf = malloc(max_bytes);
    char *recvbuf = malloc(max_bytes);
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(sendbuf, rank, max_bytes);

    const int right = (rank + 1) % size;
    const int left  = (rank - 1 + size) % size;

    if (rank == 0)
        printf("# ring over %d processes\n"
               "# bytes | per-hop us | per-hop us (avg) | bandwidth MB/s\n",
               size);

    for (int nbytes = 1; nbytes <= max_bytes; nbytes <<= 1) {
        int iters = nbytes <= 1024 ? 10000 : (nbytes <= 65536 ? 2000 : 500);

        /* warm-up lap */
        MPI_Sendrecv(sendbuf, nbytes, MPI_BYTE, right, 0,
                     recvbuf, nbytes, MPI_BYTE, left, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int k = 0; k < iters; k++) {
            MPI_Sendrecv(sendbuf, nbytes, MPI_BYTE, right, k,
                         recvbuf, nbytes, MPI_BYTE, left, k,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double elapsed = MPI_Wtime() - t0;
        double per_hop_us = elapsed / iters / size * 1e6;

        double avg_us;
        MPI_Reduce(&per_hop_us, &avg_us, 1, MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            avg_us /= size;
            double avg_bw = nbytes / (avg_us * 1e-6) / 1e6;
            printf("%8d | %10.3f | %14.3f | %12.2f\n",
                   nbytes, per_hop_us, avg_us, avg_bw);
        }
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
