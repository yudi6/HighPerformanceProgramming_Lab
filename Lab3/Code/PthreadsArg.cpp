#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

int thread_count;
int *A;
// 索引的全局变量
int gloal_index = 0;
// 计算求和结果
int sum = 0;
// 互斥量用以保护临界区
pthread_mutex_t mutex;

void *PthArgSum(void *rank);

int main(int argc, char *argv[])
{
    // 初始化互斥量
    pthread_mutex_init(&mutex, NULL);
    // 计算从0到999的和
    A = new int[1000];
    for (int i = 0; i < 1000; ++i)
        A[i] = i;
        
    float cal_time = 0;
    struct timeval start;
    struct timeval end;

    long thread;
    pthread_t *thread_handles;

    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    
    gettimeofday(&start,NULL);
    // 创建线程
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, PthArgSum, (void *)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    gettimeofday(&end,NULL);
    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    printf("Sum of A is %d\n", sum);
    cal_time = (end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0;
    printf("Calculation time is %.10f ms\n",cal_time);
    return 0;
}

void *PthArgSum(void *rank)
{
    // 各线程循环计算
    while (true)
    {
        // 临界区
        pthread_mutex_lock(&mutex);
        if (gloal_index < 1000)
        {
            sum += A[gloal_index];
            gloal_index++;
            pthread_mutex_unlock(&mutex);
        }
        // 下标超过999则计算结束
        else
        {
            pthread_mutex_unlock(&mutex);
            break;
        }
    }
    return NULL;
}
