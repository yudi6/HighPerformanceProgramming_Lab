#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#define random(a, b) (rand() % (b - a) + a)

int thread_count;

void FillMatrix(int *matrix, int row, int col);
void MatrixMul(int *A, int *B, int *C, int m, int n, int k);
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k);

int main(int argc, char *argv[])
{
    if (argc != 5)
        return 1;

    int *A, *B, *C;
    int m, n, k;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);

    A = new int[m * n];
    B = new int[n * k];
    C = new int[m * k];

    FillMatrix(A, m, n);
    FillMatrix(B, n, k);
    thread_count = strtol(argv[1], NULL, 10);
    float cal_time = 0;
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    MatrixMul(A, B, C, m, n, k);
    gettimeofday(&end, NULL);
    cal_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Calculation time is %.10f ms\n", cal_time);
    // PrintMatrix(A, B, C, m, n, k);
    return 0;
}
void FillMatrix(int *matrix, int row, int col)
{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            matrix[i * col + j] = random(0, 9);
}
// GEMM
void MatrixMul(int *A, int *B, int *C, int m, int n, int k)
{
    //动态调度
#   pragma omp parallel for num_threads(thread_count)\
    schedule(dynamic,1)
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int temp = 0;
            for (int z = 0; z < n; ++z)
                temp += A[i * n + z] * B[z * k + j];
            C[i * k + j] = temp;
        }
    }
}
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k)
{
    printf("Matrix A:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%d ", A[i * n + j]);
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
            printf("%d ", B[i * k + j]);
        printf("\n");
    }
    printf("Matrix C:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
            printf("%d ", C[i * k + j]);
        printf("\n");
    }
}