#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <semaphore.h>

int thread_count;
double a, b, c;
// delta = b*b-4*a*c
double delta;
// root = sqrt(delta)
double root;
// 最终结果
double x1;
double x2;

int count = 0;
// 信号量 实现线程同步
sem_t sem_delta;
sem_t sem_root;
// 条件变量 确定是否完成计算
pthread_mutex_t mutex_finish;
pthread_cond_t cond_finish;

void *PthRoot(void *rank);

int main(int argc, char *argv[])
{
    if (argc != 4)
        return 1;
    a = strtod(argv[1], NULL);
    b = strtod(argv[2], NULL);
    c = strtod(argv[3], NULL);
    if (b * b - 4 * a * c < 0)
    {
        printf("The eqution has no root!\n");
        return 1;
    }
    // printf("%lf %lf %lf",a,b,c);
    // 初始化
    sem_init(&sem_delta, 0, 0);
    sem_init(&sem_root, 0, 0);

    pthread_mutex_init(&mutex_finish, NULL);
    pthread_cond_init(&cond_finish, NULL);

    long thread;
    pthread_t *thread_handles;
    // 四个线程完成计算
    thread_count = 4;

    thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    // 创建线程
    for (thread = 0; thread < thread_count; thread++)
        pthread_create(&thread_handles[thread], NULL, PthRoot, (void *)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread], NULL);

    sem_destroy(&sem_delta);
    sem_destroy(&sem_root);

    pthread_mutex_destroy(&mutex_finish);
    pthread_cond_destroy(&cond_finish);

    free(thread_handles);

    printf("Roots of ax^2+bx+c = 0 are x1 = %lf and x2 = %lf\n", x1, x2);
    return 0;
}

void *PthRoot(void *rank)
{
    long my_rank = (long)rank;
    // 0号线程计算delta
    // sem_delta 实现0号线程与1号线程的同步
    if (my_rank == 0)
    {
        delta = b * b - 4 * a * c;
        sem_post(&sem_delta);
    }
    // 1号线程计算root
    // sem_root 实现0号线程与1号线程的同步
    else if (my_rank == 1)
    {
        sem_wait(&sem_delta);
        root = sqrt(delta);
        sem_post(&sem_root);
    }
    // 2号线程计算根
    // 条件变量判断计算结束
    else if (my_rank == 2)
    {
        sem_wait(&sem_root);
        x1 = (-b + root) / (2 * a);
        x2 = (-b - root) / (2 * a);
        // printf("%lf", root);
        pthread_mutex_lock(&mutex_finish);
        pthread_cond_signal(&cond_finish);
        // 已经发过信号唤醒
        count++;
        pthread_mutex_unlock(&mutex_finish);
    }
    // 3号进程打印"Calculate Finished"
    else
    {
        pthread_mutex_lock(&mutex_finish);
        // 判断是否signal过
        if (count == 0)
            pthread_cond_wait(&cond_finish, &mutex_finish);
        printf("Calculate Finished\n");
        pthread_mutex_unlock(&mutex_finish);
    }
    return NULL;
}
