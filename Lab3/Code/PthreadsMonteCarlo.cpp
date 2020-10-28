#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

int thread_count;
// “飞镖"在区域内的总数
long long int sum = 0;
// 投掷的“飞镖”总数
long long number_size;
// 互斥量
pthread_mutex_t mutex;

void *PthMonteCarlo(void *rank);
long long int MonteCarlo(long long int input);

int main(int argc, char *argv[])
{
    if (argc != 3)
        return 1;
    srand(time(0));
    
    long long int number = strtoll(argv[2], NULL, 10);
    
    thread_count = strtol(argv[1], NULL, 10);
    // 每个线程分到的投掷数 
    number_size = number / thread_count;

    long thread;
    pthread_t *thread_handles;

    pthread_mutex_init(&mutex, NULL);

    thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    // 创建线程
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, PthMonteCarlo, (void *)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);
    
    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    // 判断是否投掷完 由主线程投掷剩下的“飞镖”
    long long int remain = number - number_size * thread_count;
    if (remain > 0)
        sum += MonteCarlo(remain);
    double result = sum / (double)number;
    printf("Result:%f\n", result);
    return 0;
}

void *PthMonteCarlo(void *rank)
{
    long long temp = MonteCarlo(number_size);
    // sum的临界区
    pthread_mutex_lock(&mutex);
    sum += temp; 
    pthread_mutex_unlock(&mutex);
    return NULL;
}

long long int MonteCarlo(long long int input)
{
    long long int ret = 0;
    // 投掷"飞镖"
    for (long long int i = 0; i < input; ++i)
    {
        double x = rand() / (double)(RAND_MAX);
        double y = rand() / (double)(RAND_MAX);
        // 判断是否在区域内
        if (y < x * x)
            ret++;
    }
    return ret;
}
