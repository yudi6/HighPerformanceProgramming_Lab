#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#define random(a, b) (rand() % (b - a) + a)
void FillMatrix(float *matrix, int row, int col);
void PrintMatrix(float *A, float *B, float *C, int m, int n, int k);
__global__ void MatrixMulCUDA(const float *A, const float *B, float *C, int m, int n, int k, int ThreadBlockSize)
{
    const int tid = threadIdx.x;
    const int row = tid;
    for (int i = row; i < m; i = i + ThreadBlockSize)
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
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Wrong Input!\n");
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    float *A, *B, *C;
    A = new float[m * n];
    B = new float[n * k];
    C = new float[m * k];
    FillMatrix(A, m, n);
    FillMatrix(B, n, k);
    float elapsedTime;

    float *cuda_A, *cuda_B, *cuda_C;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMalloc((void **)&cuda_A, sizeof(float) * m * n);
    cudaMalloc((void **)&cuda_B, sizeof(float) * n * k);
    cudaMalloc((void **)&cuda_C, sizeof(float) * m * k);

    cudaMemcpy(cuda_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                k, //矩阵B的列数
                m, //矩阵A的行数
                n, //矩阵A的列数
                &alpha,
                cuda_B,
                k,
                cuda_A,
                n,
                &beta,
                cuda_C,
                k);

    cudaMemcpy(C, cuda_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Calculation time is %.10f ms\n", elapsedTime);
//     PrintMatrix(A, B, C, m, n, k);
    delete[] A;
    delete[] C;
    delete[] B;
    return 0;
}
void FillMatrix(float *matrix, int row, int col)
{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            matrix[i * col + j] = random(0, 9);
}

void PrintMatrix(float *A, float *B, float *C, int m, int n, int k)
{
    printf("Matrix A:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            printf("%f ", A[i * n + j]);
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
            printf("%f ", B[i * k + j]);
        printf("\n");
    }
    printf("Matrix C:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
            printf("%f ", C[i * k + j]);
        printf("\n");
    }
}