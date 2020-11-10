#include"ParallelForLib.h"
void parallel_for(int start, int end, int crement, void *(*functor)(void *), void *arg, int num_threads)
{
    int thread_count = num_threads;
    pthread_t *thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    for_index_arg *for_index_arg_a = (for_index_arg *)malloc(thread_count * sizeof(for_index_arg));
    int block = (end - start) / thread_count;
    for (int thread = 0; thread < thread_count; thread++)
    {
        for_index_arg_a[thread].args = arg;
        for_index_arg_a[thread].start = start + thread * block;
        for_index_arg_a[thread].end = for_index_arg_a[thread].start + block;
        if (thread == (thread_count - 1))
            for_index_arg_a[thread].end = end;
        for_index_arg_a[thread].increment = crement;
        pthread_create(&thread_handles[thread], NULL, functor, (void *)(for_index_arg_a + thread));
    }
    for (int thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    free(thread_handles);
    free(for_index_arg_a);
    return;
}