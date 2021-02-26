#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <omp.h>
#define random(a, b) (rand() % (b - a) + a)
void FillMatrix(float *matrix, int row, int col);
void PrintMatrix(float *A, float *B, float *C, int m, int n, int k);
void InitMatrix(float *matrix, int row, int col);
__global__ void MatrixMulCUDA(const float *A, const float *B, float *C, int m, int n, int k, int ThreadBlockSize)
{
    const int row = blockIdx.x;
    const int col = blockIdx.y * ThreadBlockSize + threadIdx.x;
    float temp = 0;
    if (row < m && col < k)
    {
        for (int i = 0; i < n; ++i)
            temp += A[row * n + i] * B[i * k + col];
        C[row * k + col] = temp;
    }
}
__global__ void MatrixAddCUDA(const float *matrix, float *result_matrix, int m, int k)
{
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    if (row < m && col < k)
        result_matrix[row * k + col] += matrix[row * k + col];
}
__global__ void MatrixDivideCUDA(const float *matrix, float *result_matrix, int m, int n, int col_begin, int my_block_size)
{
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    if (row < m && col < col_begin + my_block_size && col >= col_begin)
        result_matrix[row * my_block_size + col - col_begin] = matrix[row * n + col];
}

int main(int argc, char **argv)
{
    if (argc != 6)
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
    InitMatrix(C, m, k);
    int tid;
    int block_size = block_size = n / thread_count;
    omp_set_num_threads(thread_count);
    int my_block_size;
    float elapsedTime = 0;
#pragma omp parallel private(tid, my_block_size)
    {
        float my_elapsedTime = 0;
        tid = omp_get_thread_num();
        my_block_size = block_size;
        //         最后的线程要计算矩阵划分的剩余部分
        if ((thread_count * block_size < n) && (tid == thread_count - 1))
            my_block_size = n - (thread_count - 1) * block_size;

        float *cuda_A, *cuda_A_block, *cuda_B_block, *cuda_C;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMalloc((void **)&cuda_A, sizeof(float) * m * n);
        cudaMalloc((void **)&cuda_A_block, sizeof(float) * m * my_block_size);
        cudaMalloc((void **)&cuda_B_block, sizeof(float) * my_block_size * k);
        cudaMalloc((void **)&cuda_C, sizeof(float) * m * k);
        //         拷贝A矩阵
        cudaMemcpy(cuda_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
        //         B矩阵部分拷贝
        cudaMemcpy(cuda_B_block, B + (block_size * tid) * k, sizeof(float) * my_block_size * k, cudaMemcpyHostToDevice);
        
        dim3 grid_temp(m, n);
        
        float elapsedTime_2 = 0;
        cudaEvent_t start_2, stop_2;
        cudaEventCreate(&start_2);
        cudaEventCreate(&stop_2);
        cudaEventRecord(start_2, 0);
        MatrixDivideCUDA<<<grid_temp, ThreadBlockSize>>>(cuda_A, cuda_A_block, m, n, block_size * tid, my_block_size);
        //         采用与任务1相同的网格定义
        dim3 grid(m, k / ThreadBlockSize);
        //         矩阵乘法
        MatrixMulCUDA<<<grid, ThreadBlockSize>>>(cuda_A_block, cuda_B_block, cuda_C, m, my_block_size, k, ThreadBlockSize);
        cudaEventRecord(stop_2, 0);
        cudaEventSynchronize(stop_2);
        cudaEventElapsedTime(&elapsedTime_2, start_2, stop_2);
        printf("Calculation time of thread %d is %.10f ms\n", tid, elapsedTime_2);

        //         返回结果
        #pragma omp critical
        {
//         printf("Matrix C: from thread %d\n ",tid);
//         for (int i = 0; i < m; ++i)
//         {
//             for (int j = 0; j < k; ++j)
//                 printf("%f ", C[i * k + j]);
//             printf("\n");
//         }
            float *cuda_C_now;
            cudaMalloc((void **)&cuda_C_now, sizeof(float) * m * k);
            cudaMemcpy(cuda_C_now, C, sizeof(float) * m * k, cudaMemcpyHostToDevice);
            dim3 grid_temp(m, k);
            MatrixAddCUDA<<<grid_temp, 1>>>(cuda_C_now, cuda_C, m, k);
            cudaMemcpy(C, cuda_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
            cudaFree(cuda_C_now);
        }
        cudaFree(cuda_A);
        cudaFree(cuda_B_block);
        cudaFree(cuda_C);
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

    printf("Total calculation time is %.10f ms\n", elapsedTime);
//         PrintMatrix(A, B, C, m, n, k);
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
void InitMatrix(float *matrix, int row, int col)
{
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            matrix[i * col + j] = 0;
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