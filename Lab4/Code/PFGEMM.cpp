#include "ParallelForLib.h"
#define random(a, b) (rand() % (b - a) + a)

struct args
{
    int *A;
    int *B;
    int *C;
    int *m;
    int *n;
    int *k;
    args(int *tA, int *tB, int *tC, int *tm, int *tn, int *tk)
    {
        A = tA;
        B = tB;
        C = tC;
        m = tm;
        n = tn;
        k = tk;
    }
};

int thread_count;

void FillMatrix(int *matrix, int row, int col);
void *MatrixMul(void *arg);
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
    struct args *arg = new args(A, B, C, &m, &n, &k);
    gettimeofday(&start, NULL);
    parallel_for(0, m, 1, MatrixMul, arg, thread_count);
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
void *MatrixMul(void *arg)
{
    struct for_index_arg *index = (struct for_index_arg *)arg;
    struct args *true_arg = (struct args *)(index->args);
    // 按行进行划分
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        for (int j = 0; j < *true_arg->k; ++j)
        {
            int temp = 0;
            for (int z = 0; z < *true_arg->n; ++z)
                temp += true_arg->A[i * (*true_arg->n) + z] * true_arg->B[z * (*true_arg->k) + j];
            true_arg->C[i * (*true_arg->k) + j] = temp;
        }
    }
    return NULL;
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