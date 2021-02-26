nvcc MatrixMul_cuda.cu -o MatrixMul_cuda
nvcc -lcublas MatrixMul_cubla.cu -o MatrixMul_cubla
nvcc -Xcompiler -fopenmp  MatrixMul_omp.cu -o MatrixMul_omp
nvcc -Xcompiler -fopenmp  MatrixMul_omp_2.cu -o MatrixMul_omp_2