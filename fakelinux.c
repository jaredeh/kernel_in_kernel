#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fakefolio.h"


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
    

    for (int i = 0; i < TOTAL_ENTRIES; i++) {
        if (fakefolios[i].check != 0) {
            output[i] = 0;
        }
        output[i] = fakefolios[i].mapcount;
    }

    // write output to binary file
    FILE *fp = fopen("coutput.bin", "wb");
    fwrite(output, sizeof(unsigned long), TOTAL_ENTRIES, fp);
    fclose(fp);

    return 0;
}
