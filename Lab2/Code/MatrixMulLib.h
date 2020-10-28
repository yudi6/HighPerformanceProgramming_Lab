#include <iostream>
#include <cstdlib>
#include <ctime>
#define random(a, b) (rand() % (b - a) + a)
using namespace std;

int **PacketMatrix(int **A, int M, int K, int length);

clock_t OptimizationStorageMul(int **A, int **B, int **C, int length);
void OptimizationMul(int **A, int **B, int **C, int length);
void SparseMatrixMul(int **A, int **B, int **C, int length);
void VectorMul(int **A, int **B, int **C, int row, int col, int length);
void StorageVectorMul(int **PA, int **B, int **C, int row, int col, int length);
void Strassen(int **A, int **B, int **R, int length);
void FillMatrix(int **A, int **B, int **C, int M, int N, int K, int length);
void Add(int **A, int **B, int **R, int length);
void Sub(int **A, int **B, int **R, int length);
void Mul(int **A, int **B, int **R, int M, int N, int K);
void PrintMatrix(int **A, int **B, int **C, int M, int N, int K);
void matrix_multiply(int **A, int **B, int **C, int M, int N, int K);
