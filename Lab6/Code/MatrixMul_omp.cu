#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <omp.h>
#define random(a, b) (rand() % (b - a) + a)
void FillMatrix(float *matrix, int row, int col);
void PrintMatrix(float *A, float *B, float *C, int m, int n, int k);
__global__ void MatrixMulCUDA(const float *A, const float *B, float *C, int m, int n, int k, int ThreadBlockSize)
{
    const int row = blockIdx.x;
    const int col = blockIdx.y*ThreadBlockSize + threadIdx.x;
    float temp = 0;
    if(row < m && col < k)
    {
        for (int i = 0; i < n; ++i)
            temp += A[row * n + i] * B[i * k + col];
        C[row * k + col] = temp;
    }
}
int main(int argc, char **argv)
{
    if(argc != 6)
    {
        printf("Wrong Input!\n");
        return 1;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int ThreadBlockSize = atoi(argv[4]);
    int thread_count = atoi(argv[5]);
    float *A, *B, *C;
    A = new float[m * n];
    B = new float[n * k];
    C = new float[m * k];
    FillMatrix(A, m, n);
    FillMatrix(B, n, k);
    int tid;
    int block_size = block_size = m / thread_count;
    omp_set_num_threads(thread_count);
    int my_block_size;
    float elapsedTime = 0;
#pragma omp parallel private(tid, my_block_size)
    {
        float my_elapsedTime = 0;
        tid = omp_get_thread_num();
        my_block_size = block_size;
//         最后的线程要计算矩阵划分的剩余部分
        if ((thread_count * block_size < m) && (tid == thread_count - 1))
            my_block_size = m - (thread_count - 1) * block_size;
        
        float * cuda_A_block, *cuda_B, *cuda_C_block;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        cudaMalloc((void **)&cuda_A_block, sizeof(float) * my_block_size * n);
        cudaMalloc((void **)&cuda_B, sizeof(float) * n * k);
        cudaMalloc((void **)&cuda_C_block, sizeof(float) * my_block_size * k);
//         根据线程号拷贝A矩阵
        cudaMemcpy(cuda_A_block, A + (block_size * tid) * n, sizeof(float) * my_block_size * n, cudaMemcpyHostToDevice);
//         B矩阵全部拷贝
        cudaMemcpy(cuda_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
//         采用与任务1相同的网格定义
        dim3 grid(my_block_size , k / ThreadBlockSize);
//         矩阵乘法
        MatrixMulCUDA<<<grid, ThreadBlockSize>>>(cuda_A_block, cuda_B, cuda_C_block, my_block_size, n, k, ThreadBlockSize); 
//         返回结果
        cudaMemcpy(C + (block_size * tid) * k, cuda_C_block, sizeof(float) * my_block_size * k, cudaMemcpyDeviceToHost);
        cudaFree(cuda_A_block);
        cudaFree(cuda_B);
        cudaFree(cuda_C_block);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&my_elapsedTime, start, stop);
//         计算各线程最长的运算时间
        #pragma omp critical
        {
            if (my_elapsedTime > elapsedTime)
                elapsedTime = my_elapsedTime;
        }   
    }

    printf("Calculation time is %.10f ms\n", elapsedTime);
//     PrintMatrix(A, B, C, m, n, k);
    delete[] A;
    delete[] C;
    delete[] B;
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