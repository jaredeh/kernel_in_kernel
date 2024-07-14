struct fakefolio {
    unsigned long check;
    char filler[72];
	unsigned long mapcount;
	unsigned long filler2;
};


#define TOTALRAM 1024*1024*1024
#define PAGE_SIZE 4
#define TOTAL_ENTRIES (TOTALRAM/PAGE_SIZE)
