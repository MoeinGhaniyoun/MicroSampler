#include <stdio.h>
#include <stdlib.h>
#include "inner.h"
#include "utils.h"

int main(int argc, char *argv[])
{
    int iters;
    unsigned char keyval[128];
    size_t blen;
    uint32_t x1[35] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    uint32_t m1[35] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    uint32_t tmp1[35], tmp2[35];

    str2hex(argv[1], keyval, 128);

    if (argc > 2) {
        iters = atoi(argv[2]);
#ifdef DEBUG
        printf("iters: %d\n", iters);
        fflush(stdout);
#endif
    }   

    blen = (iters + 7) >> 3;

#ifdef DEBUG
    printf("private key: ");
    int i;
    for(i = 0; i < 128; i++)
        printf("%x", keyval[i]);
    printf("\n");
    fflush(stdout);
#endif

    br_i31_modpow(x1, keyval, blen, m1, br_i31_ninv31(m1[1]), tmp1, tmp2);

    printf(" done.\n");
    fflush(stdout);
}
