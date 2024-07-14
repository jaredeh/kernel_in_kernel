#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define N 1000000

__global__ void vector_add_gpu(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

unsigned long long dtime_usec(unsigned long long start=0) {
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

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N * sizeof(int));
    cudaMalloc((void **) &d_b, N * sizeof(int));
    cudaMalloc((void **) &d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Warm-up kernel launch
    vector_add_gpu<<<1, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    vector_add_gpu<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Record stop event
    cudaEventRecord(stop);

    // Synchronize to wait for the kernel to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Convert milliseconds to microseconds
    float microseconds = milliseconds * 1000;
    printf("GPU Duration: %f us\n", microseconds);

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
