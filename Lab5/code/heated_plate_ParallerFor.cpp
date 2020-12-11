#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "ParallelForLib.h"
#define M 500
#define N 500
double u[M][N];
double w[M][N];
pthread_mutex_t mutex;
int main(int argc, char *argv[]);
struct args
{
  double (*w)[N];
  double *mean;
  double (*u)[N];
  double *diff;
  args(double (*tw)[N], double *tmean, double (*tu)[N], double *tdiff)
  {
    w = tw;
    mean = tmean;
    u = tu;
    diff = tdiff;
  }
};
void *functor0(void *arg);
void *functor1(void *arg);
void *functor2(void *arg);
void *functor3(void *arg);
void *functor4(void *arg);
void *functor5(void *arg);
void *functor6(void *arg);
void *functor7(void *arg);
void *functor8(void *arg);
void *functor9(void *arg);

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


  struct timeval start;
  struct timeval end;
  int thread_count = 4;
  printf("\n");
  printf("HEATED_PLATE_PARALLERFOR\n");
  printf("  C/ParallerFor version\n");
  printf("  A program to solve for the steady state temperature distribution\n");
  printf("  over a rectangular plate.\n");
  printf("\n");
  printf("  Spatial grid of %d by %d points.\n", M, N);
  printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
  printf("  Number of processors available = %d\n", 4);
  printf("  Number of threads =              %d\n", thread_count);
  /*
  Set the boundary values, which don't change. 
*/
  mean = 0.0;
  struct args *arg = new args(w, &mean, u, &diff);
  parallel_for(1, M - 1, 1, functor0, arg, thread_count);
  parallel_for(1, M - 1, 1, functor1, arg, thread_count);
  parallel_for(0, N, 1, functor2, arg, thread_count);
  parallel_for(0, N, 1, functor3, arg, thread_count);
  parallel_for(1, M - 1, 1, functor7, arg, thread_count);
  parallel_for(0, N, 1, functor8, arg, thread_count);

  /*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  mean = mean / (double)(2 * M + 2 * N - 4);
  printf("\n");
  printf("  MEAN = %f\n", mean);
  /* 
  Initialize the interior solution to the mean value.
*/
  parallel_for(1, M - 1, 1, functor4, arg, thread_count);

  /*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
  iterations = 0;
  iterations_print = 1;
  printf("\n");
  printf(" Iteration  Change\n");
  printf("\n");
  gettimeofday(&start,NULL);

      diff = epsilon;

  while (epsilon <= diff)
  {
    parallel_for(0, M, 1, functor5, arg, thread_count);
    parallel_for(1, M - 1, 1, functor6, arg, thread_count);

    /*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
    diff = 0.0;
    parallel_for(1, M - 1, 1, functor9, arg, thread_count);

    iterations++;
    if (iterations == iterations_print)
    {
      printf("  %8d  %f\n", iterations, diff);
      iterations_print = 2 * iterations_print;
    }
  }

  gettimeofday(&end,NULL);
  float cal_time = (end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/1000000.0;
  
  printf("\n");
  printf("  %8d  %f\n", iterations, diff);
  printf("\n");
  printf("  Error tolerance achieved.\n");
  printf("  Wallclock time = %f\n", cal_time);
  /*
  Terminate.
*/
  delete arg;
  printf("\n");
  printf("HEATED_PLATE_PARALLERFOR:\n");
  printf("  Normal end of execution.\n");
  return 0;
}
void *functor0(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    (true_arg->w)[i][0] = 100.0;
  }
  return NULL;
};
void *functor1(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    (true_arg->w)[i][N - 1] = 100.0;
  }
  return NULL;
};
void *functor2(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    (true_arg->w)[M - 1][i] = 100.0;
  }
  return NULL;
};
void *functor3(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    (true_arg->w)[0][i] = 0.0;
  }
  return NULL;
};
void *functor4(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    for (int j = 1; j < N - 1; j++)
    {
      (true_arg->w)[i][j] = *(true_arg->mean);
    }
  }
  return NULL;
};
void *functor5(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    for (int j = 0; j < N; j++)
    {
      (true_arg->u)[i][j] = (true_arg->w)[i][j];
    }
  }
  return NULL;
};
void *functor6(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    for (int j = 1; j < N - 1; j++)
    {
      (true_arg->w)[i][j] = ((true_arg->u)[i - 1][j] + (true_arg->u)[i + 1][j] + (true_arg->u)[i][j - 1] + (true_arg->u)[i][j + 1]) / 4.0;
    }
  }
  return NULL;
};
void *functor7(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  double sum = 0.0;
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    sum += (true_arg->w)[i][0] + (true_arg->w)[i][N - 1];
  }
  pthread_mutex_lock(&mutex);
  *(true_arg->mean) += sum;
  pthread_mutex_unlock(&mutex);
  return NULL;
};
void *functor8(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  double sum = 0.0;
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    sum += (true_arg->w)[M - 1][i] + (true_arg->w)[0][i];
  }
  pthread_mutex_lock(&mutex);
  *(true_arg->mean) += sum;
  pthread_mutex_unlock(&mutex);
  return NULL;
};
void *functor9(void *arg)
{
  struct for_index_arg *index = (struct for_index_arg *)arg;
  struct args *true_arg = (struct args *)(index->args);
  double temp_diff = 0.0;
  for (int i = index->start; i < index->end; i = i + index->increment)
  {
    for (int j = 1; j < N - 1; j++)
    {
      if (temp_diff < fabs((true_arg->w)[i][j] - (true_arg->u)[i][j]))
      {
        temp_diff = fabs((true_arg->w)[i][j] - (true_arg->u)[i][j]);
      }
    }
  }
  // 临界区的访问
  pthread_mutex_lock(&mutex);
  if (*(true_arg->diff) < temp_diff)
  {
    *(true_arg->diff) = temp_diff;
  }
  pthread_mutex_unlock(&mutex);
  return NULL;
};