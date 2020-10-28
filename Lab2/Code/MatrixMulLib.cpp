#include"MatrixMulLib.h"
void OptimizationMul(int **A, int **B, int **C, int length)
{
	//对矩阵进行4*4的拆分
	for (int i = 0; i < length; i += 4)
	{
		for (int j = 0; j < length; j += 4)
		{
			VectorMul(A, B, C, i, j, length);
		}
	}
}

void VectorMul(int **A, int **B, int **C, int row, int col, int length)
{
	//使用寄存器保存各结果
	register int c_00_reg,
		c_01_reg, c_02_reg, c_03_reg, c_10_reg, c_11_reg, c_12_reg, c_13_reg, c_20_reg, c_21_reg, c_22_reg, c_23_reg, c_30_reg, c_31_reg, c_32_reg, c_33_reg,
		a_0i_reg, a_1i_reg, a_2i_reg, a_3i_reg, b_i0_reg, b_i1_reg, b_i2_reg, b_i3_reg;

	c_00_reg = 0.0;
	c_01_reg = 0.0;
	c_02_reg = 0.0;
	c_03_reg = 0.0;
	c_10_reg = 0.0;
	c_11_reg = 0.0;
	c_12_reg = 0.0;
	c_13_reg = 0.0;
	c_20_reg = 0.0;
	c_21_reg = 0.0;
	c_22_reg = 0.0;
	c_23_reg = 0.0;
	c_30_reg = 0.0;
	c_31_reg = 0.0;
	c_32_reg = 0.0;
	c_33_reg = 0.0;

	for (int i = 0; i < length; ++i)
	{
		a_0i_reg = A[row][i];
		a_1i_reg = A[row + 1][i];
		a_2i_reg = A[row + 2][i];
		a_3i_reg = A[row + 3][i];

		b_i0_reg = B[i][col];
		b_i1_reg = B[i][col + 1];
		b_i2_reg = B[i][col + 2];
		b_i3_reg = B[i][col + 3];

		c_00_reg += a_0i_reg * b_i0_reg;
		c_01_reg += a_0i_reg * b_i1_reg;
		c_02_reg += a_0i_reg * b_i2_reg;
		c_03_reg += a_0i_reg * b_i3_reg;

		c_10_reg += a_1i_reg * b_i0_reg;
		c_11_reg += a_1i_reg * b_i1_reg;
		c_12_reg += a_1i_reg * b_i2_reg;
		c_13_reg += a_1i_reg * b_i3_reg;

		c_20_reg += a_2i_reg * b_i0_reg;
		c_21_reg += a_2i_reg * b_i1_reg;
		c_22_reg += a_2i_reg * b_i2_reg;
		c_23_reg += a_2i_reg * b_i3_reg;

		c_30_reg += a_3i_reg * b_i0_reg;
		c_31_reg += a_3i_reg * b_i1_reg;
		c_32_reg += a_3i_reg * b_i2_reg;
		c_33_reg += a_3i_reg * b_i3_reg;
	}
	C[row][col] += c_00_reg;
	C[row][col + 1] += c_01_reg;
	C[row][col + 2] += c_02_reg;
	C[row][col + 3] += c_03_reg;

	C[row + 1][col] += c_10_reg;
	C[row + 1][col + 1] += c_11_reg;
	C[row + 1][col + 2] += c_12_reg;
	C[row + 1][col + 3] += c_13_reg;

	C[row + 2][col] += c_20_reg;
	C[row + 2][col + 1] += c_21_reg;
	C[row + 2][col + 2] += c_22_reg;
	C[row + 2][col + 3] += c_23_reg;

	C[row + 3][col] += c_30_reg;
	C[row + 3][col + 1] += c_31_reg;
	C[row + 3][col + 2] += c_32_reg;
	C[row + 3][col + 3] += c_33_reg;
}

void Strassen(int **A, int **B, int **C, int length)
{
	int Half = length / 2;
	if (length <= 64)
	{
		Mul(A, B, C, length, length, length);
		return;
	}

	int **A11 = new int *[Half];
	int **A12 = new int *[Half];
	int **A21 = new int *[Half];
	int **A22 = new int *[Half];

	int **B11 = new int *[Half];
	int **B12 = new int *[Half];
	int **B21 = new int *[Half];
	int **B22 = new int *[Half];

	int **C11 = new int *[Half];
	int **C12 = new int *[Half];
	int **C21 = new int *[Half];
	int **C22 = new int *[Half];

	int **M1 = new int *[Half];
	int **M2 = new int *[Half];
	int **M3 = new int *[Half];
	int **M4 = new int *[Half];
	int **M5 = new int *[Half];
	int **M6 = new int *[Half];
	int **M7 = new int *[Half];

	int **TempA = new int *[Half];
	int **TempB = new int *[Half];

	for (int i = 0; i < Half; ++i)
	{
		A11[i] = new int[Half];
		A12[i] = new int[Half];
		A21[i] = new int[Half];
		A22[i] = new int[Half];

		B11[i] = new int[Half];
		B12[i] = new int[Half];
		B21[i] = new int[Half];
		B22[i] = new int[Half];

		C11[i] = new int[Half];
		C12[i] = new int[Half];
		C21[i] = new int[Half];
		C22[i] = new int[Half];

		M1[i] = new int[Half];
		M2[i] = new int[Half];
		M3[i] = new int[Half];
		M4[i] = new int[Half];
		M5[i] = new int[Half];
		M6[i] = new int[Half];
		M7[i] = new int[Half];

		TempA[i] = new int[Half];
		TempB[i] = new int[Half];
	}

	for (int i = 0; i < Half; i++)
	{
		for (int j = 0; j < Half; j++)
		{
			A11[i][j] = A[i][j];
			A12[i][j] = A[i][j + Half];
			A21[i][j] = A[i + Half][j];
			A22[i][j] = A[i + Half][j + Half];

			B11[i][j] = B[i][j];
			B12[i][j] = B[i][j + Half];
			B21[i][j] = B[i + Half][j];
			B22[i][j] = B[i + Half][j + Half];
		}
	}

	Add(A11, A22, TempA, Half);
	Add(B11, B22, TempB, Half);
	Strassen(TempA, TempB, M1, Half);

	Add(A21, A22, TempA, Half);
	Strassen(TempA, B11, M2, Half);

	Sub(B12, B22, TempA, Half);
	Strassen(TempA, A11, M3, Half);

	Sub(B21, B11, TempA, Half);
	Strassen(A22, TempA, M4, Half);

	Add(A11, A12, TempA, Half);
	Strassen(TempA, B22, M5, Half);

	Sub(A21, A11, TempA, Half);
	Add(B11, B12, TempB, Half);
	Strassen(TempA, TempB, M6, Half);

	Sub(A12, A22, TempA, Half);
	Add(B21, B22, TempB, Half);
	Strassen(TempA, TempB, M7, Half);

	//C11 = M1 + M4 - M5 + M7;
	Add(M1, M4, TempA, Half);
	Sub(M7, M5, TempB, Half);
	Add(TempA, TempB, C11, Half);

	//C12 = M3 + M5;
	Add(M3, M5, C12, Half);

	//C21 = M2 + M4;
	Add(M2, M4, C21, Half);

	//C22 = M1 + M3 - M2 + M6;
	Add(M1, M3, TempA, Half);
	Sub(M6, M2, TempB, Half);
	Add(TempA, TempB, C22, Half);

	for (int i = 0; i < Half; ++i)
	{
		for (int j = 0; j < Half; ++j)
		{
			C[i][j] = C11[i][j];
			C[i][j + Half] = C12[i][j];
			C[i + Half][j] = C21[i][j];
			C[i + Half][j + Half] = C22[i][j];
		}
	}
	for (int i = 0; i < Half; i++)
	{
		delete[] A11[i];
		delete[] A12[i];
		delete[] A21[i];
		delete[] A22[i];

		delete[] B11[i];
		delete[] B12[i];
		delete[] B21[i];
		delete[] B22[i];

		delete[] C11[i];
		delete[] C12[i];
		delete[] C21[i];
		delete[] C22[i];

		delete[] M1[i];
		delete[] M2[i];
		delete[] M3[i];
		delete[] M4[i];
		delete[] M5[i];
		delete[] M6[i];
		delete[] M7[i];

		delete[] TempA[i];
		delete[] TempB[i];
	}
	delete[] A11;
	delete[] A12;
	delete[] A21;
	delete[] A22;

	delete[] B11;
	delete[] B12;
	delete[] B21;
	delete[] B22;

	delete[] C11;
	delete[] C12;
	delete[] C21;
	delete[] C22;

	delete[] M1;
	delete[] M2;
	delete[] M3;
	delete[] M4;
	delete[] M5;
	delete[] M6;
	delete[] M7;

	delete[] TempA;
	delete[] TempB;

	return;
}

int **PacketMatrix(int **A, int M, int K, int length)
{
	int **PA = new int *[length];
	for (int i = 0; i < length; ++i)
		PA[i] = new int[length];
	for (int i = 0; i < length; i += 4)
	{
		for (int j = i; j < i + 4; j++)
		{
			for (int l = 0; l < length; l++)
			{
				PA[i + l % 4][j % 4 * 4 + l % 4] = A[j][l];
			}
		}
	}
	return PA;
}

void StorageVectorMul(int **PA, int **B, int **C, int row, int col, int length)
{
	register int c_00_reg, c_01_reg, c_02_reg, c_03_reg,
		c_10_reg, c_11_reg, c_12_reg, c_13_reg,
		c_20_reg, c_21_reg, c_22_reg, c_23_reg,
		c_30_reg, c_31_reg, c_32_reg, c_33_reg,
		a_0i_reg, a_1i_reg, a_2i_reg, a_3i_reg,
		b_i0_reg, b_i1_reg, b_i2_reg, b_i3_reg;

	c_00_reg = 0;
	c_01_reg = 0;
	c_02_reg = 0;
	c_03_reg = 0;
	c_10_reg = 0;
	c_11_reg = 0;
	c_12_reg = 0;
	c_13_reg = 0;
	c_20_reg = 0;
	c_21_reg = 0;
	c_22_reg = 0;
	c_23_reg = 0;
	c_30_reg = 0;
	c_31_reg = 0;
	c_32_reg = 0;
	c_33_reg = 0;

	for (int i = 0; i < length; ++i)
	{
		a_0i_reg = PA[row + i % 4][row % 4 * 4 + i % 4];
		a_1i_reg = PA[row + i % 4][(row + 1) % 4 * 4 + i % 4];
		a_2i_reg = PA[row + i % 4][(row + 2) % 4 * 4 + i % 4];
		a_3i_reg = PA[row + i % 4][(row + 3) % 4 * 4 + i % 4];

		b_i0_reg = B[i][col];
		b_i1_reg = B[i][col + 1];
		b_i2_reg = B[i][col + 2];
		b_i3_reg = B[i][col + 3];

		c_00_reg += a_0i_reg * b_i0_reg;
		c_01_reg += a_0i_reg * b_i1_reg;
		c_02_reg += a_0i_reg * b_i2_reg;
		c_03_reg += a_0i_reg * b_i3_reg;

		c_10_reg += a_1i_reg * b_i0_reg;
		c_11_reg += a_1i_reg * b_i1_reg;
		c_12_reg += a_1i_reg * b_i2_reg;
		c_13_reg += a_1i_reg * b_i3_reg;

		c_20_reg += a_2i_reg * b_i0_reg;
		c_21_reg += a_2i_reg * b_i1_reg;
		c_22_reg += a_2i_reg * b_i2_reg;
		c_23_reg += a_2i_reg * b_i3_reg;

		c_30_reg += a_3i_reg * b_i0_reg;
		c_31_reg += a_3i_reg * b_i1_reg;
		c_32_reg += a_3i_reg * b_i2_reg;
		c_33_reg += a_3i_reg * b_i3_reg;
	}
	C[row][col] += c_00_reg;
	C[row][col + 1] += c_01_reg;
	C[row][col + 2] += c_02_reg;
	C[row][col + 3] += c_03_reg;

	C[row + 1][col] += c_10_reg;
	C[row + 1][col + 1] += c_11_reg;
	C[row + 1][col + 2] += c_12_reg;
	C[row + 1][col + 3] += c_13_reg;

	C[row + 2][col] += c_20_reg;
	C[row + 2][col + 1] += c_21_reg;
	C[row + 2][col + 2] += c_22_reg;
	C[row + 2][col + 3] += c_23_reg;

	C[row + 3][col] += c_30_reg;
	C[row + 3][col + 1] += c_31_reg;
	C[row + 3][col + 2] += c_32_reg;
	C[row + 3][col + 3] += c_33_reg;
}

void FillMatrix(int **A, int **B, int **C, int M, int N, int K, int length)
{
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < length; ++j)
		{
			if (i < M && j < N)
				A[i][j] = random(0, 9);
			else
				A[i][j] = 0;
		}

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < length; ++j)
		{
			if (i < N && j < K)
				B[i][j] = random(0, 9);
			else
				B[i][j] = 0;
		}
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < K; ++j)
			C[i][j] = 0;
}

void Add(int **A, int **B, int **R, int length)
{
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < length; ++j)
			R[i][j] = A[i][j] + B[i][j];
}

void Sub(int **A, int **B, int **R, int length)
{
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < length; ++j)
			R[i][j] = A[i][j] - B[i][j];
}

void Mul(int **A, int **B, int **R, int M, int N, int K)
{
	//通用矩阵乘法
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < K; ++j)
		{
			R[i][j] = 0;
			for (int z = 0; z < N; ++z)
				R[i][j] += A[i][z] * B[z][j];
		}
	}
}

void PrintMatrix(int **A, int **B, int **C, int M, int N, int K)
{
	cout << "Matrix A:" << endl;
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; ++j)
			printf("%d ", A[i][j]);
		printf("\n");
	}
	cout << "Matrix B:" << endl;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < K; ++j)
			printf("%d ", B[i][j]);
		printf("\n");
	}
	cout << "Matrix C:" << endl;
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < K; ++j)
			printf("%d ", C[i][j]);
		printf("\n");
	}
}

void matrix_multiply(int **A, int **B, int **C, int M, int N, int K)
{
	if ((M > 2048) || (N > 2048) || (K > 2048))
	{
		cout << "Error" << endl;
		return;
	}
	int length = 1;
	while (length < M || length < N || length < K)
		length *= 2;
	int **Atemp = new int *[length];
	int **Btemp = new int *[length];
	int **Ctemp = new int *[length];
	for (int i = 0; i < length; ++i)
	{
		Atemp[i] = new int[length];
		Btemp[i] = new int[length];
		Ctemp[i] = new int[length];
	}
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			Atemp[i][j] = A[i][j];
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < K; ++j)
			Btemp[i][j] = B[i][j];
	clock_t start = clock();
	Mul(Atemp, Btemp, Ctemp, M, N, K);
	clock_t end = clock();
	cout << "GEMM运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;
	FillMatrix(Atemp, Btemp, Ctemp, M, N, K, length);
	start = clock();
	Strassen(Atemp, Btemp, Ctemp, length);
	end = clock();
	cout << "Strassen运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;
	if (length < 4)
		exit(0);
	FillMatrix(Atemp, Btemp, Ctemp, M, N, K, length);
	start = clock();
	OptimizationMul(Atemp, Btemp, Ctemp, length);
	end = clock();
	cout << "OptimizationMul运算时间：" << 1000 * (end - start) / CLOCKS_PER_SEC << "ms" << endl;
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < K; ++j)
			C[i][j] = Ctemp[i][j];
	return;
}
