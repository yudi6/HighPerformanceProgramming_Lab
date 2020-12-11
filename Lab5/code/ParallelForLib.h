#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
struct for_index_arg
{
    int start;
    int end;
    int increment;
    void *args;
};
void parallel_for(int start, int end, int crement, void *(*functor)(void *), void *arg, int num_threads);