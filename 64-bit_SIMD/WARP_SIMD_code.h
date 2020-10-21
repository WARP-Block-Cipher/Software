/*******************************************************
 * Software implementation using SIMD by WARP Team     *
 *******************************************************/
#ifndef WARP_SIMD_CODE_H__
#define WARP_SIMD_CODE_H__
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>

enum RETURNEnum {
	SUCCESS,   /**< 0 */
	FAILURE	   /**< 1 */
};

typedef struct
{
	int keylen;
	__m128i round_key[2];
} ctx_t;

#define R   41  /*round number*/
#define RN  6   /*rotation number*/
#define BR  32  /*brunch number*/

#define BLOCK_SIZE_INBYTES ((BR) >> 1)
#define KEY_SIZE_INBITS    (128)
#define KEY_SIZE_INBYTES   (((KEY_SIZE_INBITS)+7)/8)

void permRC();
int  ecb_enc(unsigned char *m, size_t in_len, unsigned char *c, unsigned char *k);
int  ecb_only_enc(unsigned char *m, size_t in_len, unsigned char *c, __m128i round_key[R], __m256i double_round_key[R]);
void gen_test_vector();

#ifndef PIPE
#define PIPE 8
#endif

#endif