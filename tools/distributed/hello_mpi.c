#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256];
    gethostname(host, sizeof(host));

    printf("Hello World from process %d of %d on host %s\n", rank, size, host);

    MPI_Finalize();
    return 0;
}
