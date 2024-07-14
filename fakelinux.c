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

    double duration_milliseconds = end / 1000.0;
    printf("CPU Duration: %f milliseconds\n", duration_milliseconds);

    // write output to binary file
    FILE *fp = fopen("coutput.bin", "wb");
    fwrite(output, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    return 0;
}
