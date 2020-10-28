#include"Matrix.h"
int main()
{
	int M, N, K;
	cin >> M >> N >> K;
	if ((M > 2048) || (N > 2048) || (K > 2048))
	{
		cout << "Error" << endl;
		exit(1);
	}
	int length = 1;
	while (length < M || length < N || length < K)
		length *= 2;
	int** A = new int* [length];
	int** B = new int* [length];
	int** C = new int* [length];
	for (int i = 0; i < length; ++i)
	{
		A[i] = new int[length];
		B[i] = new int[length];
		C[i] = new int[length];
	}
	FillMatrix(A, B, C, M, N, K, length);
	clock_t start = clock();
	Mul(A, B, C, M, N, K);
	clock_t end = clock();
	//PrintMatrix(A, B, C, M, N, K);
	cout << "GEMM运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;
	FillMatrix(A, B, C, M, N, K, length);
	start = clock();
	Strassen(A, B, C, length);
	end = clock();
	////PrintMatrix(A, B, C, M, N, K);
	cout << "Strassen运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;
	if (length < 4)
		exit(0);
	FillMatrix(A, B, C, M, N, K, length);
	start = clock();
	OptimizationMul(A, B, C, length);
	end = clock();
	//PrintMatrix(A, B, C, M, N, K);
	cout << "OptimizationMul运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;

	start = clock();
	clock_t time = OptimizationStorageMul(A, B, C, length);
	end = clock();
	//PrintMatrix(A, B, C, M, N, K);
	cout << "OptimizationStorageMul运算时间：" << 1000 * (end - start - time) / CLOCKS_PER_SEC << "ms" << endl;
	if (length < 16)
		exit(0);
	int** A1 = new int* [length];
	for (int i = 0; i < length; ++i)
		A1[i] = new int[length]();
	int Step = length * 0.1;
	for (int i = 0; i < M; i++)
	{
		int j = 0;
		for (j = 0; j < length - Step; j += Step)
		{
			A1[i][rand() % Step + j] = rand() % 10 - 5;
		}
	}
	start = clock();
	SparseMatrixMul(A1, B, C, length);
	end = clock();
	////PrintMatrix(A, B, C, M, N, K);
	cout << "SparseMatrixMul运算时间：" << 1000 * (end - start - time) / CLOCKS_PER_SEC << "ms" << endl;

	int** A2 = new int* [length];
	for (int i = 0; i < length; ++i)
		A2[i] = new int[length]();
	int** B2 = new int* [length];
	for (int i = 0; i < length; ++i)
		B2[i] = new int[length]();
	for (int i = 0; i < M; i++)
	{
		int j = 0;
		for (j = 0; j < length - Step; j += Step)
		{
			A2[i][rand() % Step + j] = rand() % 10 - 5; \
			B2[i][rand() % Step + j] = rand() % 10 - 5;
		}
	}
	vector<Point> PA = getNonZeroPoints(A2, length);
	vector<Point> PB = getNonZeroPoints(B2, length);

	start = clock();
	SparseMatrixMulv2(A2, B2, C, PA, PB, length);
	end = clock();
	////PrintMatrix(A, B, C, M, N, K);
	cout << "SparseMatrixMulv2运算时间：" << 1000 * (end - start - time) / CLOCKS_PER_SEC << "ms" << endl;
}

