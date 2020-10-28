#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#define random(a, b) (rand() % (b - a) + a)

int thread_count;
int *A, *B, *C;
int m, n, k;
int block_size;

void *PthMatrixMul(void *rank);
void FillMatrix(int *matrix, int row, int col);
void MatrixMul(int *A, int *B, int *C, int m, int n, int k);
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k);

int main(int argc, char *argv[])
{
    if (argc != 5)
        return 1;
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);

    A = new int[m * n];
    B = new int[n * k];
    C = new int[m * k];

    FillMatrix(A, m, n);
    FillMatrix(B, n, k);

    float cal_time = 0;
    struct timeval start;
    struct timeval end;

    long thread;
    pthread_t *thread_handles;

    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));

    // 将矩阵A按行划分
    block_size = m / thread_count;

    gettimeofday(&start,NULL);
    // 创建线程运行矩阵计算
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, PthMatrixMul, (void *)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    free(thread_handles);
    // 判断是否划分完 主线程完成剩余部分的计算
    int remain = m - block_size * thread_count;
	if (remain > 0)
		MatrixMul(A + block_size * thread_count * n, B, C + block_size * thread_count * k, remain, n, k);
    
    gettimeofday(&end,NULL);
    // 计算时间
    cal_time = (end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0;
    printf("Calculation time is %.10f ms\n",cal_time);
    // PrintMatrix(A, B, C, m, n, k);
    return 0;
}

void *PthMatrixMul(void *rank)
{
    long my_rank = (long)rank;
    MatrixMul(A + my_rank * block_size * n, B, C + my_rank * block_size * k, block_size, n, k);
    return NULL;
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