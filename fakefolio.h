struct fakefolio {
    unsigned long check;
    char filler[72];
	unsigned long mapcount;
	unsigned long filler2;
};



#define TOTALRAM 32*1024*1024
#define PAGE_SIZE 4
#define TOTAL_ENTRIES (TOTALRAM/PAGE_SIZE)


void fakefolio_init(struct fakefolio **fakefolios, unsigned long **output) {
    *output = (unsigned long *) malloc(sizeof(unsigned long)*TOTAL_ENTRIES);
    *fakefolios = (struct fakefolio *) malloc(sizeof(struct fakefolio)*TOTAL_ENTRIES);
    memset(*fakefolios, 0, sizeof(struct fakefolio)*TOTAL_ENTRIES);
    memset(*output, 0, sizeof(unsigned long)*TOTAL_ENTRIES);
    for (int i = 0; i < TOTAL_ENTRIES; i++) {
        (*fakefolios)[i].mapcount = i;
    }
}