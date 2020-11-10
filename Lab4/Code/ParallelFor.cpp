#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
struct for_index_arg
{
    int start;      //划分后循环的开始
    int end;        //划分后循环的结束
    int increment;  //循环递增量
    void *args;     //循环函数参数
};
// void型指针指向的结构体
struct args
{
    int *A;
    int *B;
    int *C;
    int *x;
    args(int *tA, int *tB, int *tC, int *tx)
    {
        A = tA;
        B = tB;
        C = tC;
        x = tx;
    }
};

void parallel_for(int start, int end, int crement, void *(*functor)(void *), void *arg, int num_threads);
void *functor(void *arg);
void PrintArg(int *A, int *B, int *C, int *x);
int main(int argc, char *argv[])
{
    int *A = new int[10];
    int *B = new int[10];
    int *C = new int[10];
    for (int i = 0; i < 10; ++i)
    {
        A[i] = i;
        B[i] = i;
        C[i] = i;
    }
    int *x = new int;
    *x = 3;
    struct args *arg = new args(A, B, C, x);
    PrintArg(A,B,C,x);
    parallel_for(0, 10, 1, functor, arg, 4);
    PrintArg(A,B,C,x);
    delete []A;
    delete []B;
    delete []C;
    delete x;
    delete arg;
    return 0;
}
void *functor(void *arg)
{
    struct for_index_arg *index = (struct for_index_arg *)arg;
    struct args *true_arg = (struct args *)(index->args);
    for (int i = index->start; i < index->end; i = i + index->increment)
    {
        (true_arg->A)[i] = (true_arg->B)[i] * (*(true_arg->x)) + (true_arg->C)[i];
    }
    return NULL;
}
void parallel_for(int start, int end, int crement, void *(*functor)(void *), void *arg, int num_threads)
{
    // 线程数
    int thread_count = num_threads;
    pthread_t *thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    // 每个线程对应的for_index_arg结构体
    for_index_arg *for_index_arg_a = (for_index_arg *)malloc(thread_count * sizeof(for_index_arg));
    // 每个线程分到的循环次数
    int block = (end - start) / thread_count;
    // 对每个循环的for_index_arg结构体赋值
    for (int thread = 0; thread < thread_count; thread++)
    {
        for_index_arg_a[thread].args = arg;
        for_index_arg_a[thread].start = start + thread * block;
        for_index_arg_a[thread].end = for_index_arg_a[thread].start + block;
        // 对于最后一个线程 需要保证循环被完整划分
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
void PrintArg(int *A, int *B, int *C, int *x)
{
    printf("A：\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", A[i]);
    printf("\n");
    printf("B：\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", B[i]);
    printf("\n");
    printf("C：\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", C[i]);
    printf("\n");
    printf("x:\n%d\n",*x);
}