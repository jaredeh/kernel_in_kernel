#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define N 1000000

// Function prototype for dtime_usec
unsigned long long dtime_usec(unsigned long long start);

void vector_add_cpu(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

unsigned long long dtime_usec(unsigned long long start) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000000ULL) + tv.tv_usec) - start;
}

int main() {
    int *a = (int *) malloc(N * sizeof(int));
    int *b = (int *) malloc(N * sizeof(int));
    int *c = (int *) malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    unsigned long long start = dtime_usec(0);
    vector_add_cpu(a, b, c, N);
    unsigned long long end = dtime_usec(start);

    printf("CPU Duration: %llu us\n", end);

    free(a);
    free(b);
    free(c);

    return 0;
}
