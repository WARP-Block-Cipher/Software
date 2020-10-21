/*********************************************
 * Reference implementation by WARP Team     *
**********************************************/
#include <stdio.h>
#include <string.h>

#define R     41   /*round number*/
#define RN    6    /*rotation number*/
#define BR    32   /*branch number*/
#define BR_HALF    (BR / 2)  /*half of the branch number*/


int Sbox[BR_HALF] = { 0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6 };

int Ml[BR_HALF] = {3,7,6,4,1,0,2,5,11,15,14,12,9,8,10,13};
int Mr[BR_HALF] = {15, 14, 0, 10, 13, 1, 12, 11, 7, 6, 8, 2, 5, 9, 4, 3};
int invMl[BR_HALF] = {5,4,6,0,3,7,2,1,13,12,14,8,11,15,10,9};
int invMr[BR_HALF] = { 2, 5, 11, 15, 14, 12, 9, 8, 10, 13, 3, 7, 6, 4, 1, 0};
int Rot[BR_HALF] = {10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9};
int invRot[BR_HALF] = {6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5};

int invMr4_Ml_MR3[BR_HALF] = {15, 8, 9, 14, 5, 2, 11, 12, 7, 0, 1, 6, 13, 10, 3, 4};
int invMr3_Ml_MR2[BR_HALF] = {14, 15, 0, 5, 3, 10, 4, 9, 6, 7, 8, 13, 11, 2, 12, 1};
int invMr2_Ml_MR1[BR_HALF] = {5, 6, 1, 10, 11, 0, 7, 4, 13, 14, 9, 2, 3, 8, 15, 12};
int invMr1_Ml_MR0[BR_HALF] = {6, 0, 12, 13, 10, 9, 15, 11, 14, 8, 4, 5, 2, 1, 7, 3};

int MR7[BR_HALF] = {2, 5, 11, 15, 14, 12, 9, 8, 10, 13, 3, 7, 6, 4, 1, 0};
int MR6[BR_HALF] = {11, 12, 7, 0, 1, 6, 13, 10, 3, 4, 15, 8, 9, 14, 5, 2};
int MR5[BR_HALF] = {7, 6, 8, 2, 5, 9, 4, 3, 15, 14, 0, 10, 13, 1, 12, 11};
int MR4[BR_HALF] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
int MR3[BR_HALF] = {10, 13, 3, 7, 6, 4, 1, 0, 2, 5, 11, 15, 14, 12, 9, 8};
int MR2[BR_HALF] = {3, 4, 15, 8, 9, 14, 5, 2, 11, 12, 7, 0, 1, 6, 13, 10};
int MR1[BR_HALF] = {15, 14, 0, 10, 13, 1, 12, 11, 7, 6, 8, 2, 5, 9, 4, 3};
int MR0[BR_HALF] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};


int RC0[R] = { 0x0U, 0x0U, 0x1U, 0x3U, 0x7U, 0xfU, 0xfU, 0xfU, 0xeU, 0xdU, 0xaU, 0x5U, 0xaU, 0x5U, 0xbU, 0x6U, 0xcU, 0x9U, 0x3U, 0x6U, 0xdU, 0xbU, 0x7U, 0xeU, 0xdU, 0xbU, 0x6U, 0xdU, 0xaU, 0x4U, 0x9U, 0x2U, 0x4U, 0x9U, 0x3U, 0x7U, 0xeU, 0xcU, 0x8U, 0x1U, 0x2U};
int RC1[R] = { 0x4U, 0xcU, 0xcU, 0xcU, 0xcU, 0xcU, 0x8U, 0x4U, 0x8U, 0x4U, 0x8U, 0x4U, 0xcU, 0x8U, 0x0U, 0x4U, 0xcU, 0x8U, 0x4U, 0xcU, 0xcU, 0x8U, 0x4U, 0xcU, 0x8U, 0x4U, 0x8U, 0x0U, 0x4U, 0x8U, 0x0U, 0x4U, 0xcU, 0xcU, 0x8U, 0x0U, 0x0U, 0x4U, 0x8U, 0x4U, 0xcU};

void enc(int *m, int *c, int *k);
void sboxkey(int *middle, int *rk);
void exchange(int *left, int *right);

#define PRINT_INTER 0

void print_state(int *m)
{
    for (int i = 0; i < BR; i++)
    {
        printf("%x ", m[i]&0xf);
    }
    printf("\n");
};

void printRKey(int *roundkey, int ri)
{
     printf("%d round key: ", ri);
    for (int x = 0; x < BR_HALF; x++)
    {
        printf("%x ", roundkey[x]);
    }
    printf("\n");
}

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
    int permedrk[R][BR_HALF];

    for (int i = 0; i < BR_HALF; i++)
    {
        left[i]  = m[2 * i + 0];
        right[i] = m[2 * i + 1];

        rk[0][i] = k[i];
        rk[1][i] = k[BR_HALF + i];
    }

    /* permutate round key */
    permedrk[0][0] = rk[0][0] ^ RC0[0];
    permedrk[0][1] = rk[0][1] ^ RC1[0];
    for (int i = 2; i < BR_HALF; i++)
    {
        permedrk[0][i] = rk[0][i];
    }
    for (int ri = 0; ri < 5; ri++)
    {
        /*add round constants*/
        permedrk[ri * 8 + 1][MR7[0]] = rk[1][0] ^ RC0[ri * 8 + 1];
        permedrk[ri * 8 + 2][MR6[0]] = rk[0][0] ^ RC0[ri * 8 + 2];
        permedrk[ri * 8 + 3][MR5[0]] = rk[1][0] ^ RC0[ri * 8 + 3];
        permedrk[ri * 8 + 4][MR4[0]] = rk[0][0] ^ RC0[ri * 8 + 4];
        permedrk[ri * 8 + 5][MR3[0]] = rk[1][0] ^ RC0[ri * 8 + 5];
        permedrk[ri * 8 + 6][MR2[0]] = rk[0][0] ^ RC0[ri * 8 + 6];
        permedrk[ri * 8 + 7][MR1[0]] = rk[1][0] ^ RC0[ri * 8 + 7];
        permedrk[ri * 8 + 8][MR0[0]] = rk[0][0] ^ RC0[ri * 8 + 8];

        permedrk[ri * 8 + 1][MR7[1]] = rk[1][1] ^ RC1[ri * 8 + 1];
        permedrk[ri * 8 + 2][MR6[1]] = rk[0][1] ^ RC1[ri * 8 + 2];
        permedrk[ri * 8 + 3][MR5[1]] = rk[1][1] ^ RC1[ri * 8 + 3];
        permedrk[ri * 8 + 4][MR4[1]] = rk[0][1] ^ RC1[ri * 8 + 4];
        permedrk[ri * 8 + 5][MR3[1]] = rk[1][1] ^ RC1[ri * 8 + 5];
        permedrk[ri * 8 + 6][MR2[1]] = rk[0][1] ^ RC1[ri * 8 + 6];
        permedrk[ri * 8 + 7][MR1[1]] = rk[1][1] ^ RC1[ri * 8 + 7];
        permedrk[ri * 8 + 8][MR0[1]] = rk[0][1] ^ RC1[ri * 8 + 8];

        for (int i = 2; i < BR_HALF; i++)
        {
            permedrk[ri * 8 + 1][MR7[i]] = rk[1][i];
            permedrk[ri * 8 + 2][MR6[i]] = rk[0][i];
            permedrk[ri * 8 + 3][MR5[i]] = rk[1][i];
            permedrk[ri * 8 + 4][MR4[i]] = rk[0][i];
            permedrk[ri * 8 + 5][MR3[i]] = rk[1][i];
            permedrk[ri * 8 + 6][MR2[i]] = rk[0][i];
            permedrk[ri * 8 + 7][MR1[i]] = rk[1][i];
            permedrk[ri * 8 + 8][MR0[i]] = rk[0][i];
        }
    }
    #if PRINT_INTER
    for (int ri = 0; ri < R; ri++)
        printRKey(permedrk[ri], ri + 1);
    #endif

    /*first function*/
    #if PRINT_INTER
    printf("%d round\n", 1);
    printState(left, right);
    #endif

    /* round function 1 */
    memcpy(middle, left, sizeof(middle));
    sboxkey(middle, permedrk[0]);
    for (int j = 0; j < BR_HALF; j++) right[j] ^= middle[j];

    /*round function(2 to 41 round)*/
    for (int ri = 0; ri < 10; ri++)
    {
        exchange(left, right);
        #if PRINT_INTER
        printf("%d round\n", ri * 4 + 1 + 1);
        printState(left, right);
        #endif
        memcpy(temp, left, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++) left[invMr4_Ml_MR3[i]] = temp[i];
        memcpy(middle, left, sizeof(middle));
        sboxkey(middle, permedrk[ri * 4 + 1]);
        for (int j = 0; j < BR_HALF; j++) right[j] ^= middle[j];

        exchange(left, right);
        #if PRINT_INTER
        printf("%d round\n", ri * 4 + 2 + 1);
        printState(left, right);
        #endif
        memcpy(temp, left, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++) left[invMr3_Ml_MR2[i]] = temp[i];
        memcpy(middle, left, sizeof(middle));
        sboxkey(middle, permedrk[ri * 4 + 2]);
        for (int j = 0; j < BR_HALF; j++) right[j] ^= middle[j];

        exchange(left, right);
        #if PRINT_INTER
        printf("%d round\n", ri * 4 + 3 + 1);
        printState(left, right);
        #endif
        memcpy(temp, left, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++) left[invMr2_Ml_MR1[i]] = temp[i];
        memcpy(middle, left, sizeof(middle));
        sboxkey(middle, permedrk[ri * 4 + 3]);
        for (int j = 0; j < BR_HALF; j++) right[j] ^= middle[j];

        exchange(left, right);
        #if PRINT_INTER
        printf("%d round\n", ri * 4 + 4 + 1);
        printState(left, right);
        #endif
        memcpy(temp, left, BR_HALF * sizeof(int));
        for (int i = 0; i < BR_HALF; i++) left[invMr1_Ml_MR0[i]] = temp[i];
        memcpy(middle, left, sizeof(middle));
        sboxkey(middle, permedrk[ri * 4 + 4]);
        for (int j = 0; j < BR_HALF; j++) right[j] ^= middle[j];
    }
    #if PRINT_INTER
    printf("%d round\n", R);
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
