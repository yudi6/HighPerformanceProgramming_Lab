#include<iostream>
#include<cstdlib>
#include<ctime>
#include<vector>
#define random(a,b) (rand()%(b-a)+a)
using namespace std;
struct Point
{
	int i, j;
	Point(int ia, int ja)
	{
		i = ia;
		j = ja;
	}
};
int** PacketMatrix(int** A, int M, int K, int length);
vector<Point> getNonZeroPoints(int** matrix, int length);
void SparseMatrixMulv2(int** A, int** B, int** C, vector<Point> VA, vector<Point> VB, int length);
clock_t OptimizationStorageMul(int** A, int** B, int** C, int length);
void OptimizationMul(int** A, int** B, int** C, int length);
void SparseMatrixMul(int** A, int** B, int** C, int length);
void VectorMul(int** A, int** B, int** C, int row, int col, int length);
void StorageVectorMul(int** PA, int** B, int** C, int row, int col, int length);
void Strassen(int** A, int** B, int** R, int length);
void FillMatrix(int** A, int** B, int** C, int M, int N, int K, int length);
void Add(int** A, int** B, int** R, int length);
void Sub(int** A, int** B, int** R, int length);
void Mul(int** A, int** B, int** R, int M, int N, int K);
void PrintMatrix(int** A, int** B, int** C, int M, int N, int K);

