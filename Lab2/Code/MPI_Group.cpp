#include <stdio.h>

#include <mpi.h>
#define random(a, b) (rand() % (b - a) + a)

void FillMatrix(int *matrix, int row, int col);
void MatrixMul(int *A, int *B, int *C, int m, int n, int k);
void PrintMatrix(int *A, int *B, int *C, int m, int n, int k);

int main(int argc, char **argv)
{
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

	int block_size = m / comm_sz;

	block_A = new int[block_size * n];
	B = new int[n * k];
	block_C = new int[block_size * k];

	if (my_rank == 0)
	{
		A = new int[m * n];
		C = new int[m * k];
		FillMatrix(A, m, n);
		FillMatrix(B, n, k);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	printf("rank 0 Scatter\n");
	// 对矩阵A进行划分并发送
	MPI_Scatter(A, block_size * n, MPI_INT, block_A, block_size * n, MPI_INT, 0, MPI_COMM_WORLD);
	printf("rank 0 Bcast\n");
	// 广播矩阵B
	MPI_Bcast(B, n * k, MPI_INT, 0, MPI_COMM_WORLD);

	MatrixMul(block_A, B, block_C, block_size, n, k);

	MPI_Barrier(MPI_COMM_WORLD);
	// 整合矩阵C
	MPI_Gather(block_C, block_size * k, MPI_INT, C, block_size * k, MPI_INT, 0, MPI_COMM_WORLD);
	printf("rank 0 Gather\n");
	// 计算划分剩下的矩阵
	if (my_rank == 0 && block_size * comm_sz < m)
	{
		int remain = m - block_size * comm_sz;
		MatrixMul(A + block_size * comm_sz * n, B, C + block_size * comm_sz * k, remain, n, k);
	}
	end = MPI_Wtime();

	if (my_rank == 0)
	{
		printf("Computation time is %fs\n", end - start);
		delete[] A;
		delete[] C;
	}
	delete[] block_A;
	delete[] block_C;
	delete[] B;

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