#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#define random(a, b) (rand() % (b - a) + a)
void FillMatrix(float *matrix, int row, int col);
void PrintMatrix(float *A, float *B, float *C, int m, int n, int k);
__global__ void MatrixMulCUDA(const float *A, const float *B, float *C, int m, int n, int k, int ThreadBlockSize)
{
    //     计算元素的行
    const int row = blockIdx.x;
    //     计算元素的列
    const int col = blockIdx.y * ThreadBlockSize + threadIdx.x;
    float temp = 0;
    if (row < m && col < k)
    {
        for (int i = 0; i < n; ++i)
            temp += A[row * n + i] * B[i * k + col];
        C[row * k + col] = temp;
    }
}
int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Wrong Input!\n");
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int ThreadBlockSize = atoi(argv[4]);

    float *A, *B, *C;
    A = new float[m * n];
    B = new float[n * k];
    C = new float[m * k];
    FillMatrix(A, m, n);
    FillMatrix(B, n, k);

    float *cuda_A, *cuda_B, *cuda_C;
    //     使用cuda内置API计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //     申请空间
    cudaMalloc((void **)&cuda_A, sizeof(float) * m * n);
    cudaMalloc((void **)&cuda_B, sizeof(float) * n * k);
    cudaMalloc((void **)&cuda_C, sizeof(float) * m * k);
    //     将A、B矩阵从CPU转移到GPU
    cudaMemcpy(cuda_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
    //     定义的结构网格
    dim3 grid(m, k / ThreadBlockSize);
    //     矩阵乘法
    MatrixMulCUDA<<<grid, ThreadBlockSize>>>(cuda_A, cuda_B, cuda_C, m, n, k, ThreadBlockSize);
    
    //     将结果C矩阵从GPU转移回CPU
    cudaMemcpy(C, cuda_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_C);
    //     计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Matrix Size is %d\nThreadBlockSize is %d\n", m,ThreadBlockSize);
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