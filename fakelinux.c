#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fakefolio.h"
#include <time.h>
#include <sys/time.h>

unsigned long long dtime_usec(unsigned long long start) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000000ULL) + tv.tv_usec) - start;
}


void *fakefolio_init(struct fakefolio **fakefolios, unsigned long **output) {
    *output = malloc(sizeof(unsigned long)*TOTAL_ENTRIES);
    *fakefolios = malloc(sizeof(struct fakefolio)*TOTAL_ENTRIES);
    memset(*fakefolios, 0, sizeof(struct fakefolio)*TOTAL_ENTRIES);
    memset(*output, 0, sizeof(unsigned long)*TOTAL_ENTRIES);
    for (int i = 0; i < TOTAL_ENTRIES; i++) {
        (*fakefolios)[i].mapcount = i;
    }    
}

int main() {
    struct fakefolio *fakefolios;
    unsigned long *output;
    fakefolio_init(&fakefolios, &output);
    printf("initialized fakefolios\n");
    
    unsigned long long start = dtime_usec(0);
    for (int i = 0; i < TOTAL_ENTRIES; i++) {
        if (fakefolios[i].check != 0) {
            output[i] = 0;
        }
        output[i] = fakefolios[i].mapcount;
    }
    unsigned long long end = dtime_usec(start);

    printf("CPU Duration: %llu us\n", end);

    // write output to binary file
    FILE *fp = fopen("coutput.bin", "wb");
    fwrite(output, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    return 0;
}
