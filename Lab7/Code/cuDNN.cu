#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cudnn.h>

#define random(a, b) (rand() % (b - a) + a)
#define index(i, j, col) (((i) * (col)) + (j))

void PrintMatrix(float *A, int row, int col);
void FillMatrix(float *matrix, int row, int col, int padding);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Wrong Input!\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int stride = atoi(argv[2]);

    int channel = 3;
    float *matrix;
    float *filter;
    float *result;

    int matrix_height = size;
    int matrix_width = size;

    int filter_height = 3;
    int filter_width = 3;

    timeval t1, t2;
    //创建句柄
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int padding = ((((matrix_height - filter_height) / stride + 1) * stride - (matrix_height - filter_height)) % stride) / 2;
    int matrix_size = sizeof(float) * (matrix_height) * (matrix_width);
    int result_size = sizeof(float) * ((matrix_height - filter_height + 2 * padding) / stride + 1) * ((matrix_width - filter_width + 2 * padding) / stride + 1);
    int filter_size = sizeof(float) * filter_height * filter_width;

    int result_height = (matrix_height - filter_height + 2 * padding) / stride + 1;
    int result_width = (matrix_width - filter_width + 2 * padding) / stride + 1;

    matrix = (float *)malloc(matrix_size * channel);
    memset(matrix, 0, sizeof(matrix));

    for (int i = 0; i < channel; i++)
        FillMatrix(matrix + i * matrix_height * matrix_width, matrix_height, matrix_width, 0);

    filter = (float *)malloc(filter_size * channel);
    for (int i = 0; i < channel; i++)
        for (int j = 0; j < filter_height * filter_width; j++)
            filter[i * filter_height * filter_width + j] = j + 1;

    gettimeofday(&t1, NULL);
    
    //输入矩阵的描述子
    cudnnTensorDescriptor_t matrix_desc;
    cudnnCreateTensorDescriptor(&matrix_desc);
    cudnnSetTensor4dDescriptor(matrix_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, matrix_height, matrix_width);

    float *cuda_matrix;
    cudaMalloc(&cuda_matrix, 1 * channel * matrix_height * matrix_width * sizeof(float));
    
    //卷积核的描述子
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, channel, filter_height, filter_width);

    float *cuda_filter;
    cudaMalloc(&cuda_filter, 1 * channel * filter_height * filter_width * sizeof(float));
    
    //卷积的描述子
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int result_n;
    int result_c;
    int result_h;
    int result_w;
    result = (float *)malloc(result_size);

    cudnnGetConvolution2dForwardOutputDim(conv_desc, matrix_desc, filt_desc, &result_n, &result_c, &result_h, &result_w);
    
    //输出结果的描述子
    cudnnTensorDescriptor_t result_desc;
    cudnnCreateTensorDescriptor(&result_desc);
    cudnnSetTensor4dDescriptor(result_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result_n, result_c, result_h, result_w);

    float *cuda_result;
    cudaMalloc(&cuda_result, result_n * result_c * result_h * result_w * sizeof(float));
    
    //选择计算的算法
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, matrix_desc, filt_desc, conv_desc, result_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    //准备计算所用的空间
    size_t ws_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, matrix_desc, filt_desc, conv_desc, result_desc, algo, &ws_size);
    float *ws_data;
    cudaMalloc(&ws_data, ws_size);

    float alpha = 1.f;
    float beta = 0.f;

    cudaMemcpy(cuda_matrix, matrix, 1 * channel * matrix_height * matrix_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_filter, filter, 1 * channel * filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);

    cudnnConvolutionForward(cudnn, &alpha, matrix_desc, cuda_matrix, filt_desc, cuda_filter, conv_desc, algo, ws_data, ws_size, &beta, result_desc, cuda_result);

    gettimeofday(&t2, NULL);
    printf("Matrix Size:%d\tStride:%d\n", size, stride);
    printf("Calculation time:%ldms\n", t2.tv_sec * 1000 + t2.tv_usec/1000 - t1.tv_sec * 1000 - t1.tv_usec/1000);

//     for (int i = 0; i < channel; i++)
//     {
//         printf("Matrix of channel %d:\n", i);
//         PrintMatrix(matrix + i * matrix_height * matrix_width, matrix_height, matrix_width);
//     }

//     for (int i = 0; i < channel; i++)
//     {
//         printf("Filter of channel %d:\n", i);
//         PrintMatrix(filter + i * filter_height * filter_width, filter_height, filter_width);
//     }
    cudaMemcpy(result, cuda_result, result_size, cudaMemcpyDeviceToHost);
//     printf("Result:\n");
//     PrintMatrix(result, result_height, result_width);

    cudaFree(ws_data);
    cudaFree(cuda_result);
    cudnnDestroyTensorDescriptor(result_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFree(cuda_filter);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudaFree(cuda_matrix);
    cudnnDestroyTensorDescriptor(matrix_desc);
    cudnnDestroy(cudnn);
    return 0;
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