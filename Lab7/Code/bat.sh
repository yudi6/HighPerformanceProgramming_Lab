nvcc CNN_cuda.cu -o CNN_cuda
nvcc im2col_cuda.cu -o im2col_cuda
nvcc cuDNN.cu -o cuDNN -I/opt/conda/include -L/opt/conda/lib -lcudnn