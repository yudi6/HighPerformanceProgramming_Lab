#include <stdio.h>

#include <mpi.h>
#define random(a, b) (rand() % (b - a) + a)

void FillMatrix(int *matrix, int row, int col);
void MatrixMul(int *A, int *B, int *C, int m, int n, int k);
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k);

int main(int argc, char **argv)
{
	// 输入m,n,k
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);

	int *A, *B, *C;
	int *block_A, *block_C;

	int comm_sz;
	int my_rank;

	double start, end;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// 划分的行数
	int block_size = m / comm_sz;
	// A矩阵块的缓冲
	block_A = new int[block_size * n];
	// B矩阵的缓冲
	B = new int[n * k];
	// C矩阵块的缓冲
	block_C = new int[block_size * k];

	if (my_rank == 0)
	{
		A = new int[m * n];
		C = new int[m * k];
		FillMatrix(A, m, n);
		FillMatrix(B, n, k);
		start = MPI_Wtime();
		// 发送A矩阵块与B矩阵
		for (int i = 1; i < comm_sz; ++i)
		{
			printf("rank 0 send A to %d\n", i);
			MPI_Send(A + i * block_size * n, block_size * n, MPI_INT, i, 0, MPI_COMM_WORLD);
			printf("rank 0 send B to %d\n", i);
			MPI_Send(B, n * k, MPI_INT, i, 1, MPI_COMM_WORLD);
		}
		// 计算属于0号线程的那部分矩阵
		MatrixMul(A, B, C, block_size, n, k);
		// 无法整除 计算剩余矩阵
		int remain = m - block_size * comm_sz - 1;
		if (remain > 0)
			MatrixMul(A + block_size * comm_sz * n, B, C + block_size * comm_sz * k, remain, n, k);
		// 接收来自其他核的结果
		for (int i = 1; i < comm_sz; ++i)
		{
			MPI_Recv(C + i * block_size * k, block_size * k, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			printf("rank 0 recv C from %d\n", i);
		}
		end = MPI_Wtime();
		// PrintMatrix(A, B, C, m, n, k);
		printf("Computation time is %fs\n", end - start);
	}
	else
	{
		// 接收A矩阵块
		MPI_Recv(block_A, block_size * n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		printf("rank %d recv A from 0\n", my_rank);
		// 接收B矩阵
		MPI_Recv(B, n * k, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		printf("rank %d recv B from 0\n", my_rank);
		// 计算矩阵
		MatrixMul(block_A, B, block_C, block_size, n, k);
		printf("rank %d send C to 0\n", my_rank);
		// 发送计算结果
		MPI_Send(block_C, block_size * k, MPI_INT, 0, my_rank, MPI_COMM_WORLD);
	}

	delete[] block_A;
	delete[] block_C;
	delete[] B;

	if (my_rank == 0)
	{
		delete[] A;
		delete[] C;
	}

	MPI_Finalize();

	return 0;
}

void FillMatrix(int *matrix, int row, int col)
{
	for (int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j)
			matrix[i * col + j] = random(0, 9);
}

void MatrixMul(int *A, int *B, int *C, int m, int n, int k)
{
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
		{
			int temp = 0;
			for (int z = 0; z < n; ++z)
				temp += A[i * n + z] * B[z * k + j];
			C[i * k + j] = temp;
		}
	}
}

void PrintMatrix(int *A, int *B, int *C, int m, int n, int k)
{
	printf("Matrix A:\n");
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
			printf("%d ", A[i * n + j]);
		printf("\n");
	}
	printf("Matrix B:\n");
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < k; ++j)
			printf("%d ", B[i * k + j]);
		printf("\n");
	}
	printf("Matrix C:\n");
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < k; ++j)
			printf("%d ", C[i * k + j]);
		printf("\n");
	}
}