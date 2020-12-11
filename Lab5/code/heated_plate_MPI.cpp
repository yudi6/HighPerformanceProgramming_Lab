#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define M 500
#define N 500
double u[M][N];
double w[M][N];
char buffer[sizeof(double) * (N + 1)];
/******************************************************************************/

int main(int argc, char *argv[])

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_OPENMP.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/
{
  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double wtime;

  int comm_sz;
  int my_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0)
  {
    printf("\n");
    printf("HEATED_PLATE_MPI\n");
    printf("  C/MPI version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of processors available = %d\n", 4);
    printf("  Number of threads =              %d\n", comm_sz);
  }
      /*
  Set the boundary values, which don't change. 
*/
  int position = 0;
  mean = 0.0;
  int block = (M - 0) / comm_sz;          // 进程分块
  int for_start = 0 + (my_rank * block);  // 每块的开始
  int for_end = for_start + block;        // 结束
  if (my_rank == (comm_sz - 1))           // 最后一个进程获得无法整除的部分
    for_end = M;
  double my_mean = 0.0;
  if (my_rank == 0)
  {
    for (i = for_start; i < for_end; ++i)
    {
      for (j = 0; j < N; ++j)
      {
        // 第一行全初始化为0
        if (i == 0)
          w[i][j] = 0.0;
        // 最后一行全初始化为100.0
        else if (i == M - 1)
        {
          w[i][j] = 100.0;
          my_mean += w[i][j];
        }
        // 中间的行只有初始与末尾元素初始化为100.0
        else if (j == 0 || j == N - 1)
        {
          w[i][j] = 100.0;
          my_mean += w[i][j];
        }
      }
    }
  }
  else if (my_rank == (comm_sz - 1))
  {
    for (i = for_start; i < for_end; ++i)
    {
      for (j = 0; j < N; ++j)
      {
        // 最后一行全初始化为100.0
        if (i == M - 1)
        {
          w[i][j] = 100.0;
          my_mean += w[i][j];
        }
        // 中间的行只有初始与末尾元素初始化为100.0
        else if (j == 0 || j == N - 1)
        {
          w[i][j] = 100.0;
          my_mean += w[i][j];
        }
      }
    }
  }
  else
  {
    // 中间的行只有初始与末尾元素初始化为100.0
    for (i = for_start; i < for_end; ++i)
    {
      w[i][0] = 100.0;
      w[i][N - 1] = 100.0;
      my_mean += (w[i][0] + w[i][N - 1]);
    }
  }
  if (my_rank == 0)
  {
    mean += my_mean;
    for (i = 1; i < comm_sz; ++i)
    {
      // 接收来自其他进程的mean
      MPI_Recv(buffer, 100, MPI_PACKED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      position = 0;
      // 按位置unpack到my_mean中
      MPI_Unpack(buffer, 100, &position, &my_mean, 1, MPI_DOUBLE, MPI_COMM_WORLD);
      // 累加
      mean += my_mean;
    }
    // 计算总的平均值
    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("\n");
    printf("  MEAN = %f\n", mean);
    for (i = 1; i < comm_sz; ++i)
    {
      // 将计算的结果发送给其他进程
      MPI_Pack(&mean, 1, MPI_DOUBLE, buffer, 100, &position, MPI_COMM_WORLD);
      MPI_Send(buffer, position, MPI_PACKED, i, 1, MPI_COMM_WORLD);
    }
  }
  else
  {
    // 打包my_mean给主进程
    MPI_Pack(&my_mean, 1, MPI_DOUBLE, buffer, 100, &position, MPI_COMM_WORLD);
    // 发送给主进程
    MPI_Send(buffer, position, MPI_PACKED, 0, 0, MPI_COMM_WORLD);
    // 接受来自主进程的mean
    MPI_Recv(buffer, 100, MPI_PACKED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Unpack(buffer, 100, &position, &mean, 1, MPI_DOUBLE, MPI_COMM_WORLD);
  }
  // printf("%lf from %d\n", mean, my_rank);
  /*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/

  /*
    Initialize the interior solution to the mean value.
  */
  for (i = for_start; i < for_end; i++)
  {
    if (i == 0)
      continue;
    if (i == M - 1)
      continue;
    for (j = 1; j < N - 1; j++)
    {
      w[i][j] = mean;
    }
  }
  /*
    iterate until the  new solution W differs from the old solution U
    by no more than EPSILON.
  */
  iterations = 0;
  iterations_print = 1;
  if (my_rank == 0)
  {
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    wtime = MPI_Wtime();
  }

  diff = epsilon;
  while (epsilon <= diff)
  {
    for (i = for_start; i < for_end; i++)
    {
      for (j = 0; j < N; j++)
      {
        u[i][j] = w[i][j];
      }
    }
    // 进程间进行行的传递
    if (comm_sz > 1)
    {
      // Send中tag参数为1代表向下一块传输该块的最后一行
      // tag参数为2代表向上一块传输该块的第一行
      // 为了不发生阻塞，先向下再向上
      if (my_rank == 0)
      {
        // 第一块先向下传递
        position = 0;
        MPI_Pack(&w[for_end - 1][0], N, MPI_DOUBLE, buffer, sizeof(double) * (N + 1), &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, 1, 1, MPI_COMM_WORLD);
        // 再从下接收
        position = 0;
        MPI_Recv(buffer, sizeof(double) * (N + 1), MPI_PACKED, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buffer, sizeof(double) * (N + 1), &position, &u[for_end][0], N, MPI_DOUBLE, MPI_COMM_WORLD);
      }
      else if (my_rank == (comm_sz - 1))
      {
        // 最后一块先从上接收
        position = 0;
        MPI_Recv(buffer, sizeof(double) * (N + 1), MPI_PACKED, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buffer, sizeof(double) * (N + 1), &position, &u[for_start - 1][0], N, MPI_DOUBLE, MPI_COMM_WORLD);
        // 再向上传递
        position = 0;
        MPI_Pack(&w[for_start][0], N, MPI_DOUBLE, buffer, sizeof(double) * (N + 1), &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, my_rank - 1, 2, MPI_COMM_WORLD);
      }
      else
      {
        // 中间块先从上接收
        position = 0;
        MPI_Recv(buffer, sizeof(double) * (N + 1), MPI_PACKED, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buffer, sizeof(double) * (N + 1), &position, &u[for_start - 1][0], N, MPI_DOUBLE, MPI_COMM_WORLD);
        // 再向下传递
        position = 0;
        MPI_Pack(&w[for_end - 1][0], N, MPI_DOUBLE, buffer, sizeof(double) * (N + 1), &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, my_rank + 1, 1, MPI_COMM_WORLD);
        // 再再从下接收
        position = 0;
        MPI_Recv(buffer, sizeof(double) * (N + 1), MPI_PACKED, my_rank + 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buffer, sizeof(double) * (N + 1), &position, &u[for_end][0], N, MPI_DOUBLE, MPI_COMM_WORLD);
        // 最后向上传递
        position = 0;
        MPI_Pack(&w[for_start][0], N, MPI_DOUBLE, buffer, sizeof(double) * (N + 1), &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, my_rank - 1, 2, MPI_COMM_WORLD);
      }
    }
    for (i = for_start; i < for_end; i++)
    {
      if (i == 0 || i == M - 1)
        continue;
      for (j = 1; j < N - 1; j++)
      {
        w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
      }
    }
    diff = 0.0;
    my_diff = 0.0;
    for (i = for_start; i < for_end; i++)
    {
      if (i == 0 || i == M - 1)
        continue;
      for (j = 1; j < N - 1; j++)
      {
        if (my_diff < fabs(w[i][j] - u[i][j]))
        {
          my_diff = fabs(w[i][j] - u[i][j]);
        }
      }
    }
    position = 0;
    iterations++;
    if (my_rank == 0)
    {
      if (diff < my_diff)
        diff = my_diff;
      for (i = 1; i < comm_sz; ++i)
      {
        MPI_Recv(buffer, 100, MPI_PACKED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        position = 0;
        MPI_Unpack(buffer, 100, &position, &my_diff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if (diff < my_diff)
          diff = my_diff;
      }
      if (iterations == iterations_print)
      {
        printf("  %8d  %f\n", iterations, diff);
        iterations_print = 2 * iterations_print;
      }
      for (i = 1; i < comm_sz; ++i)
      {
        MPI_Pack(&diff, 1, MPI_DOUBLE, buffer, 100, &position, MPI_COMM_WORLD);
        MPI_Send(buffer, position, MPI_PACKED, i, 1, MPI_COMM_WORLD);
      }
    }
    else
    {
      MPI_Pack(&my_diff, 1, MPI_DOUBLE, buffer, 100, &position, MPI_COMM_WORLD);
      MPI_Send(buffer, position, MPI_PACKED, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(buffer, 100, MPI_PACKED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Unpack(buffer, 100, &position, &diff, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    }
  }
  if (my_rank == 0)
  {
    wtime = MPI_Wtime() - wtime;

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
    /*
    Terminate.
  */
    printf("\n");
    printf("HEATED_PLATE_MPI:\n");
    printf("  Normal end of execution.\n");
  }
  MPI_Finalize();
  return 0;
}
#undef M
#undef N