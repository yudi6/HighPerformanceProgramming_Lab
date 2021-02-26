#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>

#define random(a, b) (rand() % (b - a) + a)
#define index(i, j, col) (((i) * (col)) + (j))

void PrintMatrix(float *A, int row, int col);
void FillMatrix(float *matrix, int row, int col, int padding);

__global__ void convolution(float *matrix, float *filter, float *result, int height_stride, int width_stride, int matrix_height, int matrix_width, int filter_height, int filter_width, int result_height, int result_width)
{
    //     计算元素的行号 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
     //     计算元素的列号
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //     卷积结果
    float sum = 0;
    if (i < result_height && j < result_width)
    {
        for (int x = 0; x < filter_height; x++)
            for (int y = 0; y < filter_width; y++)
                sum += matrix[index(i * height_stride + x, j * width_stride + y, matrix_width)] * filter[index(x, y, filter_width)];
        //     结果累加
        *(result + index(i, j, result_width)) += sum;
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Wrong Input!\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int stride = atoi(argv[2]);
    
    int x = atoi(argv[3]);
    int y = atoi(argv[4]);
    dim3 threadsPerBlock(x, y);
    
    int channel = 3;
    float *matrix[channel];
    float *filter[channel];
    float *result;
    //     矩阵大小 定义为方阵
    int matrix_height = size;
    int matrix_width = size;
    //     卷积和大小
    int filter_height = 3;
    int filter_width = 3;
    //     根据步长计算出需要补全的长度
    int padding = ((((matrix_height - filter_height) / stride + 1) * stride - (matrix_height - filter_height)) % stride) / 2;
    
    int matrix_size = sizeof(float) * (matrix_height + 2 * padding) * (matrix_width + 2 * padding);
    int result_size = sizeof(float) * ((matrix_height - filter_height + 2 * padding) / stride + 1) * ((matrix_width - filter_width + 2 * padding) / stride + 1);
    int filter_size = sizeof(float) * filter_height * filter_width;
    //     初始化矩阵
    for (int i = 0; i < channel; i++)
    {
        matrix[i] = (float *)malloc(matrix_size);
        memset(matrix[i], 0, sizeof(matrix[i]));
        FillMatrix(matrix[i], matrix_height, matrix_width, padding);
    }
    for (int i = 0; i < channel; i++)
    {
        filter[i] = (float *)malloc(filter_size);
        for (int j = 0; j < filter_height * filter_width; j++)
            filter[i][j] = j + 1;
    }
    result = (float *)malloc(result_size);
    
    timeval t1, t2;
    gettimeofday(&t1, NULL);

    float *cuda_matrix[channel];
    float *cuda_filter[channel];
    float *cuda_result;

    for (int i = 0; i < channel; i++)
    {
        cudaMalloc(&cuda_matrix[i], matrix_size);
        cudaMemcpy(cuda_matrix[i], matrix[i], matrix_size, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < channel; i++)
    {
        cudaMalloc(&cuda_filter[i], filter_size);
        cudaMemcpy(cuda_filter[i], filter[i], filter_size, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&cuda_result, result_size);
    cudaMemset(cuda_result, 0, result_size);
    
    int result_height = (matrix_height - filter_height + 2 * padding) / stride + 1;
    int result_width = (matrix_width - filter_width + 2 * padding) / stride + 1;

    dim3 numBlocks((result_height % x) ? result_height / x + 1 : result_height / x, (result_width % y) ? result_width / y + 1 : result_width / y);

    for (int i = 0; i < channel; i++)
    {
        convolution<<<numBlocks, threadsPerBlock>>>(cuda_matrix[i],
                                                    cuda_filter[i],
                                                    cuda_result,
                                                    stride, stride,
                                                    matrix_height + 2 * padding,
                                                    matrix_width + 2 * padding,
                                                    filter_height,
                                                    filter_width,
                                                    result_height,
                                                    result_width);
    }
    gettimeofday(&t2, NULL);
    printf("Matrix Size:%d\tStride:%d\n", size, stride);
    printf("Calculation time:%ldms\n", t2.tv_sec * 1000 + t2.tv_usec/1000 - t1.tv_sec * 1000 - t1.tv_usec/1000);

    cudaMemcpy(result, cuda_result, result_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < channel; i++)
    {
        printf("Matrix after padding of channel %d:\n",i);
        PrintMatrix(matrix[i], matrix_height + 2 * padding, matrix_width + 2 * padding);
    }
    for (int i = 0; i < channel; i++)
    {
        printf("Filter of channel %d:\n",i);
        PrintMatrix(filter[i], filter_height, filter_width);
    }
    printf("Result:\n");
    PrintMatrix(result, ((matrix_height - filter_height + 2 * padding) / stride + 1), ((matrix_width - filter_width + 2 * padding) / stride + 1));
    
    for (int i = 0; i < channel; i++)
        cudaFree(cuda_matrix[i]);
    for (int i = 0; i < channel; i++)
        cudaFree(cuda_filter[i]);
    cudaFree(cuda_result);
    
    for (int i = 0; i < channel; i++)
        free(matrix[i]);
    for (int i = 0; i < channel; i++)
        free(filter[i]);
    free(result);
}

void FillMatrix(float *matrix, int row, int col, int padding)
{
    for (int i = padding; i < row + padding; i++)
        for (int j = padding; j < col + padding; j++)
            matrix[index(i, j, col + 2 * padding)] = random(0, 9);
}

void PrintMatrix(float *A, int row, int col)
{
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
            printf("%f ", A[i * col + j]);
        printf("\n");
    }
}