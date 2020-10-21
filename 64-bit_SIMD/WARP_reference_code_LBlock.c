/*********************************************
 * Reference implementation by WARP Team     *
**********************************************/
#include <stdio.h>
#include <string.h>

#define R 41    /*round number*/
#define RN    6    /*rotation number*/
#define BR    32  /*branch number*/
#define BR_HALF    (BR / 2)  /*half of the branch number*/

int Sbox[BR_HALF] = {0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6};

int Ml[BR_HALF] = {3,7,6,4,1,0,2,5,11,15,14,12,9,8,10,13};
int Mr[BR_HALF] = {15, 14, 0, 10, 13, 1, 12, 11, 7, 6, 8, 2, 5, 9, 4, 3};
int invMl[BR_HALF] = {5,4,6,0,3,7,2,1,13,12,14,8,11,15,10,9};
int invMr[BR_HALF] = { 2, 5, 11, 15, 14, 12, 9, 8, 10, 13, 3, 7, 6, 4, 1, 0};
int Rot[BR_HALF] = {10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9};
int invRot[BR_HALF] = {6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5};

int PIl[BR_HALF] = {5,4,6,0,3,7,2,1,13,12,14,8,11,15,10,9};
int PIr[BR_HALF] = {6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5};

int POl[BR_HALF] = {3,7,6,4,1,0,2,5,11,15,14,12,9,8,10,13};
int POr[BR_HALF] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};



int RC0[R] = { 0x0U, 0x0U, 0x1U, 0x3U, 0x7U, 0xfU, 0xfU, 0xfU, 0xeU, 0xdU, 0xaU, 0x5U, 0xaU, 0x5U, 0xbU, 0x6U, 0xcU, 0x9U, 0x3U, 0x6U, 0xdU, 0xbU, 0x7U, 0xeU, 0xdU, 0xbU, 0x6U, 0xdU, 0xaU, 0x4U, 0x9U, 0x2U, 0x4U, 0x9U, 0x3U, 0x7U, 0xeU, 0xcU, 0x8U, 0x1U, 0x2U};
int RC1[R] = { 0x4U, 0xcU, 0xcU, 0xcU, 0xcU, 0xcU, 0x8U, 0x4U, 0x8U, 0x4U, 0x8U, 0x4U, 0xcU, 0x8U, 0x0U, 0x4U, 0xcU, 0x8U, 0x4U, 0xcU, 0xcU, 0x8U, 0x4U, 0xcU, 0x8U, 0x4U, 0x8U, 0x0U, 0x4U, 0x8U, 0x0U, 0x4U, 0xcU, 0xcU, 0x8U, 0x0U, 0x0U, 0x4U, 0x8U, 0x4U, 0xcU};

void enc(int *m, int *c, int *k);
void sboxkey(int *middle, int *rk);
void exchange(int *left, int *right);

#define PRINT_INTER 1

void print_state(int *m)
{
    for (int i = 0; i < BR; i++)
    {
        printf("%x ", m[i]&0xf);
    }
    printf("\n");
};

void printState(int *left, int*right)
{
    printf("L: ");
    for (int x = 0; x < BR_HALF; x++)
    {
        printf("%x ", left[x]);
    }
    printf("R: ");
    for (int x = 0; x < BR_HALF; x++)
    {
        printf("%x ", right[x]);
    }
    printf("\n");
}


int main()
{
    int k[4][BR] = {
        {0x0U, 0xaU, 0xcU, 0xdU, 0x0U, 0x2U, 0x2U, 0xfU, 0x6U, 0x8U, 0x0U, 0xaU, 0x5U, 0x4U, 0x7U, 0xfU, 0xeU, 0xeU, 0x0U, 0x3U, 0xcU, 0x0U, 0x8U, 0x6U, 0x7U, 0xbU, 0x0U, 0x9U, 0xeU, 0x3U, 0xdU, 0x7U},
        {0x0U, 0x1U, 0x2U, 0x3U, 0x4U, 0x5U, 0x6U, 0x7U, 0x8U, 0x9U, 0xaU, 0xbU, 0xcU, 0xdU, 0xeU, 0xfU, 0xfU, 0xeU, 0xdU, 0xcU, 0xbU, 0xaU, 0x9U, 0x8U, 0x7U, 0x6U, 0x5U, 0x4U, 0x3U, 0x2U, 0x1U, 0x0U},
        {0x0U, 0x1U, 0x2U, 0x3U, 0x4U, 0x5U, 0x6U, 0x7U, 0x8U, 0x9U, 0xaU, 0xbU, 0xcU, 0xdU, 0xeU, 0xfU, 0xfU, 0xeU, 0xdU, 0xcU, 0xbU, 0xaU, 0x9U, 0x8U, 0x7U, 0x6U, 0x5U, 0x4U, 0x3U, 0x2U, 0x1U, 0x0U},
        {0x1U, 0x0U, 0x3U, 0x2U, 0x5U, 0x4U, 0x7U, 0x6U, 0x9U, 0x8U, 0xbU, 0xaU, 0xdU, 0xcU, 0xfU, 0xeU, 0xeU, 0xfU, 0xcU, 0xdU, 0xaU, 0xbU, 0x8U, 0x9U, 0x6U, 0x7U, 0x4U, 0x5U, 0x2U, 0x3U, 0x0U, 0x1U},
        };
    int m[4][BR] = {
        {0xaU, 0xfU, 0x6U, 0xcU, 0xdU, 0xdU, 0x9U, 0x0U, 0xfU, 0xcU, 0x5U, 0xaU, 0x6U, 0xeU, 0xaU, 0xaU, 0x8U, 0x9U, 0x7U, 0xbU, 0xcU, 0xdU, 0x1U, 0x2U, 0x0U, 0x8U, 0xdU, 0x3U, 0x9U, 0x1U, 0xeU, 0x1U},
        {0x0U, 0x1U, 0x2U, 0x3U, 0x4U, 0x5U, 0x6U, 0x7U, 0x8U, 0x9U, 0xaU, 0xbU, 0xcU, 0xdU, 0xeU, 0xfU, 0xfU, 0xeU, 0xdU, 0xcU, 0xbU, 0xaU, 0x9U, 0x8U, 0x7U, 0x6U, 0x5U, 0x4U, 0x3U, 0x2U, 0x1U, 0x0U},
        {0x0U, 0x0U, 0x1U, 0x1U, 0x2U, 0x2U, 0x3U, 0x3U, 0x4U, 0x4U, 0x5U, 0x5U, 0x6U, 0x6U, 0x7U, 0x7U, 0x8U, 0x8U, 0x9U, 0x9U, 0xaU, 0xaU, 0xbU, 0xbU, 0xcU, 0xcU, 0xdU, 0xdU, 0xeU, 0xeU, 0xfU, 0xfU},
        {0x1U, 0x0U, 0x3U, 0x2U, 0x5U, 0x4U, 0x7U, 0x6U, 0x9U, 0x8U, 0xbU, 0xaU, 0xdU, 0xcU, 0xfU, 0xeU, 0xeU, 0xfU, 0xcU, 0xdU, 0xaU, 0xbU, 0x8U, 0x9U, 0x6U, 0x7U, 0x4U, 0x5U, 0x2U, 0x3U, 0x0U, 0x1U},
        };
    int c[4][BR];

    for (int ti = 0; ti < 4; ti++)
    {
        /*key*/
        printf("key :\n");
        print_state(k[ti]);

        /*plaintext*/
        printf("plaintext :\n");
        print_state(m[ti]);

        enc(m[ti], c[ti], k[ti]);

        /*ciphertext*/
        printf("ciphertext :\n");
        print_state(c[ti]);
        printf("\n");
    }
    return 0;
}


void enc(int *m, int *c, int *k)
{
    /*intermediate value*/
    int left[BR_HALF];
    int right[BR_HALF];
    int middle[BR_HALF];
    int rk[2][BR_HALF];
    int temp[BR_HALF];

    for (int i = 0; i < BR_HALF; i++)
    {
        left[i]  = m[2 * i + 0];
        right[i] = m[2 * i + 1];

        rk[0][invMl[i]] = k[i];
        rk[1][invMl[i]] = k[BR_HALF + i];
    }


    /* pre-permutation */
    memcpy(temp, left, BR_HALF * sizeof(int));
    for (int i = 0; i < BR_HALF; i++)
    {
        left[PIl[i]] = temp[i];
    }
    memcpy(temp, right, BR_HALF * sizeof(int));
    for (int i = 0; i < BR_HALF; i++)
    {
        right[PIr[i]] = temp[i];
    }

    /*round function(1 to 40 round)*/
    for (int ri = 0; ri < R - 1; ri++)
    {
        #if PRINT_INTER
        printf("%d round\n", ri + 1);
        printState(left, right);
        #endif

        memcpy(temp, right, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++)
        {
            right[Rot[i]] = temp[i];
        }

        memcpy(middle, left, sizeof(middle));
        /*Sbox and insert key*/
        sboxkey(middle, rk[ri&1]);

        memcpy(temp, middle, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++)
        {
            middle[Ml[i]] = temp[i];
        }
        /*XOR*/
        for (int j = 0; j < BR_HALF; j++)
        {
            right[j] ^= middle[j];
        }
        /*add round constants*/
        right[0] ^= RC0[ri];
        right[1] ^= RC1[ri];

        exchange(left, right);
    }

    /*last round function */
    #if PRINT_INTER
    printf("%d round\n", R);
    printState(left, right);
    #endif

    memcpy(temp, right, BR_HALF * sizeof(int));
    for (int i = 0; i < BR_HALF; i++)
    {
        right[Rot[i]] = temp[i];
    }
    memcpy(middle, left, BR_HALF * sizeof(int));
    /*Sbox and insert key*/
    sboxkey(middle, rk[(R - 1) & 1]);

    memcpy(temp, middle, BR_HALF * sizeof(int));
    for (int i = 0; i < BR_HALF; i++)
    {
        middle[Ml[i]] = temp[i];
    }
    /*XOR*/
    for (int j = 0; j < BR_HALF; j++)
    {
        right[j] ^= middle[j];
    }
    /*add round constants*/
    right[0] ^= RC0[R - 1];
    right[1] ^= RC1[R - 1];

    /*no permutation in the last round*/

    /* post-permutation */
    memcpy(temp, left, BR_HALF * sizeof(int));
    for (int i = 0; i < BR_HALF; i++)
    {
        left[POl[i]] = temp[i];
    }
    //memcpy(temp, right, BR_HALF * sizeof(int));
    //for (int i = 0; i < BR_HALF; i++)
    //{
    //    right[POr[i]] = temp[i];
    //}

    #if PRINT_INTER
    printState(left, right);
    #endif

    /*copy ciphertext*/
    for (int i = 0; i < BR_HALF; i++)
    {
        c[2 * i + 0] = left[i];
        c[2 * i + 1] = right[i];
    }
}

void sboxkey(int *middle, int *rk)
{
    for (int i = 0; i < BR_HALF; i++)
    {
        middle[i] = Sbox[middle[i]] ^ rk[i];
    }
}

void exchange(int *left, int *right)
{
    int temp[BR_HALF];
    memcpy(temp, left, BR_HALF * sizeof(int));
    memcpy(left, right, BR_HALF * sizeof(int));
    memcpy(right, temp, BR_HALF * sizeof(int));
}
