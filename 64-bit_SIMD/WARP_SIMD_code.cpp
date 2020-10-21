/*******************************************************
 * Software implementation using SIMD by WARP Team     *
 *******************************************************/
#include "WARP_SIMD_code.h"
#include <iostream>
#include <iomanip>

using namespace std;

#define PRINT_INTER 0

#define REVERSE_16_BYTES(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd, xe, xf) \
xf, xe, xd, xc, xb, xa, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0

static const __m128i MASK_0F = _mm_set1_epi8(0x0F);
static const __m128i MASK_00FF = _mm_set1_epi16(0x00FF);
static const __m128i INTERWOVEN_SHUFFLE_16 = _mm_set_epi8(0xf,0xd,0xb,0x9,0x7,0x5,0x3,0x1,0xe,0xc,0xa,0x8,0x6,0x4,0x2,0x0);
static const __m128i Sbox = _mm_set_epi8(REVERSE_16_BYTES(0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6));
static const __m128i iperm = _mm_set_epi8(REVERSE_16_BYTES(5, 4, 6, 0, 3, 7, 2, 1, 13, 12, 14, 8, 11, 15, 10, 9));
static const __m128i rot6 = _mm_set_epi8(REVERSE_16_BYTES(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5));

static const __m256i Double_MASK_0F = _mm256_set1_epi8(0x0F);
static const __m256i Double_MASK_00FF = _mm256_set1_epi16(0x00FF);
static const __m256i Double_INTERWOVEN_SHUFFLE_16 = _mm256_set_epi8(0xf,0xd,0xb,0x9,0x7,0x5,0x3,0x1,0xe,0xc,0xa,0x8,0x6,0x4,0x2,0x0,
                                                                    0xf,0xd,0xb,0x9,0x7,0x5,0x3,0x1,0xe,0xc,0xa,0x8,0x6,0x4,0x2,0x0);
static const __m256i Double_Sbox = _mm256_set_epi8(REVERSE_16_BYTES(0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6),
                                                   REVERSE_16_BYTES(0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6));
static const __m256i Double_iperm = _mm256_set_epi8(REVERSE_16_BYTES(5, 4, 6, 0, 3, 7, 2, 1, 13, 12, 14, 8, 11, 15, 10, 9),
                                                    REVERSE_16_BYTES(5, 4, 6, 0, 3, 7, 2, 1, 13, 12, 14, 8, 11, 15, 10, 9));
static const __m256i Double_rot6 = _mm256_set_epi8(REVERSE_16_BYTES(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5),
                                                   REVERSE_16_BYTES(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5));

static const __m128i EqPerm0 = _mm_set_epi8(REVERSE_16_BYTES( 9, 10,  5, 14, 15,  4, 11,  8,  1,  2, 13,  6,  7, 12,  3,  0));
static const __m128i EqPerm1 = _mm_set_epi8(REVERSE_16_BYTES( 2, 15, 13,  4,  6,  3,  8,  9, 10,  7,  5, 12, 14, 11,  0,  1));
static const __m128i EqPerm2 = _mm_set_epi8(REVERSE_16_BYTES( 5,  2, 11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14));
static const __m128i EqPerm3 = _mm_set_epi8(REVERSE_16_BYTES( 1, 13, 12, 15, 10, 11,  0, 14,  9,  5,  4,  7,  2,  3,  8,  6));

static const __m128i EqKeyPerm0 = _mm_set_epi8(REVERSE_16_BYTES(15, 14,  0, 10, 13,  1, 12, 11,  7,  6,  8,  2,  5,  9,  4,  3));
static const __m128i EqKeyPerm1 = _mm_set_epi8(REVERSE_16_BYTES( 3,  4, 15,  8,  9, 14,  5,  2, 11, 12,  7,  0,  1,  6, 13, 10));
static const __m128i EqKeyPerm2 = _mm_set_epi8(REVERSE_16_BYTES(10, 13,  3,  7,  6,  4,  1,  0,  2,  5, 11, 15, 14, 12,  9,  8));
static const __m128i EqKeyPerm3 = _mm_set_epi8(REVERSE_16_BYTES( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7));
static const __m128i EqKeyPerm4 = _mm_set_epi8(REVERSE_16_BYTES( 7,  6,  8,  2,  5,  9,  4,  3, 15, 14,  0, 10, 13,  1, 12, 11));
static const __m128i EqKeyPerm5 = _mm_set_epi8(REVERSE_16_BYTES(11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14,  5,  2));
static const __m128i EqKeyPerm6 = _mm_set_epi8(REVERSE_16_BYTES( 2,  5, 11, 15, 14, 12,  9,  8, 10, 13,  3,  7,  6,  4,  1,  0));
static const __m128i EqKeyPerm7 = _mm_set_epi8(REVERSE_16_BYTES( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15));

static const __m256i Double_EqPerm0 = _mm256_set_epi8(REVERSE_16_BYTES( 9, 10,  5, 14, 15,  4, 11,  8,  1,  2, 13,  6,  7, 12,  3,  0), REVERSE_16_BYTES( 9, 10,  5, 14, 15,  4, 11,  8,  1,  2, 13,  6,  7, 12,  3,  0));
static const __m256i Double_EqPerm1 = _mm256_set_epi8(REVERSE_16_BYTES( 2, 15, 13,  4,  6,  3,  8,  9, 10,  7,  5, 12, 14, 11,  0,  1), REVERSE_16_BYTES( 2, 15, 13,  4,  6,  3,  8,  9, 10,  7,  5, 12, 14, 11,  0,  1));
static const __m256i Double_EqPerm2 = _mm256_set_epi8(REVERSE_16_BYTES( 5,  2, 11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14), REVERSE_16_BYTES( 5,  2, 11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14));
static const __m256i Double_EqPerm3 = _mm256_set_epi8(REVERSE_16_BYTES( 1, 13, 12, 15, 10, 11,  0, 14,  9,  5,  4,  7,  2,  3,  8,  6), REVERSE_16_BYTES( 1, 13, 12, 15, 10, 11,  0, 14,  9,  5,  4,  7,  2,  3,  8,  6));

static const __m256i Double_EqKeyPerm0 = _mm256_set_epi8(REVERSE_16_BYTES(15, 14,  0, 10, 13,  1, 12, 11,  7,  6,  8,  2,  5,  9,  4,  3), REVERSE_16_BYTES(15, 14,  0, 10, 13,  1, 12, 11,  7,  6,  8,  2,  5,  9,  4,  3));
static const __m256i Double_EqKeyPerm1 = _mm256_set_epi8(REVERSE_16_BYTES( 3,  4, 15,  8,  9, 14,  5,  2, 11, 12,  7,  0,  1,  6, 13, 10), REVERSE_16_BYTES( 3,  4, 15,  8,  9, 14,  5,  2, 11, 12,  7,  0,  1,  6, 13, 10));
static const __m256i Double_EqKeyPerm2 = _mm256_set_epi8(REVERSE_16_BYTES(10, 13,  3,  7,  6,  4,  1,  0,  2,  5, 11, 15, 14, 12,  9,  8), REVERSE_16_BYTES(10, 13,  3,  7,  6,  4,  1,  0,  2,  5, 11, 15, 14, 12,  9,  8));
static const __m256i Double_EqKeyPerm3 = _mm256_set_epi8(REVERSE_16_BYTES( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7), REVERSE_16_BYTES( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7));
static const __m256i Double_EqKeyPerm4 = _mm256_set_epi8(REVERSE_16_BYTES( 7,  6,  8,  2,  5,  9,  4,  3, 15, 14,  0, 10, 13,  1, 12, 11), REVERSE_16_BYTES( 7,  6,  8,  2,  5,  9,  4,  3, 15, 14,  0, 10, 13,  1, 12, 11));
static const __m256i Double_EqKeyPerm5 = _mm256_set_epi8(REVERSE_16_BYTES(11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14,  5,  2), REVERSE_16_BYTES(11, 12,  7,  0,  1,  6, 13, 10,  3,  4, 15,  8,  9, 14,  5,  2));
static const __m256i Double_EqKeyPerm6 = _mm256_set_epi8(REVERSE_16_BYTES( 2,  5, 11, 15, 14, 12,  9,  8, 10, 13,  3,  7,  6,  4,  1,  0), REVERSE_16_BYTES( 2,  5, 11, 15, 14, 12,  9,  8, 10, 13,  3,  7,  6,  4,  1,  0));
static const __m256i Double_EqKeyPerm7 = _mm256_set_epi8(REVERSE_16_BYTES( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15), REVERSE_16_BYTES( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15));

static const __m128i r128_RC[R] = 
{
_mm_set_epi64x(0ULL, 0x0400ULL),
_mm_set_epi64x(0ULL, 0x0c00ULL),
_mm_set_epi64x(0ULL, 0x0c01ULL),
_mm_set_epi64x(0ULL, 0x0c03ULL),
_mm_set_epi64x(0ULL, 0x0c07ULL),
_mm_set_epi64x(0ULL, 0x0c0fULL),
_mm_set_epi64x(0ULL, 0x080fULL),
_mm_set_epi64x(0ULL, 0x040fULL),
_mm_set_epi64x(0ULL, 0x080eULL),
_mm_set_epi64x(0ULL, 0x040dULL),
_mm_set_epi64x(0ULL, 0x080aULL),
_mm_set_epi64x(0ULL, 0x0405ULL),
_mm_set_epi64x(0ULL, 0x0c0aULL),
_mm_set_epi64x(0ULL, 0x0805ULL),
_mm_set_epi64x(0ULL, 0x000bULL),
_mm_set_epi64x(0ULL, 0x0406ULL),
_mm_set_epi64x(0ULL, 0x0c0cULL),
_mm_set_epi64x(0ULL, 0x0809ULL),
_mm_set_epi64x(0ULL, 0x0403ULL),
_mm_set_epi64x(0ULL, 0x0c06ULL),
_mm_set_epi64x(0ULL, 0x0c0dULL),
_mm_set_epi64x(0ULL, 0x080bULL),
_mm_set_epi64x(0ULL, 0x0407ULL),
_mm_set_epi64x(0ULL, 0x0c0eULL),
_mm_set_epi64x(0ULL, 0x080dULL),
_mm_set_epi64x(0ULL, 0x040bULL),
_mm_set_epi64x(0ULL, 0x0806ULL),
_mm_set_epi64x(0ULL, 0x000dULL),
_mm_set_epi64x(0ULL, 0x040aULL),
_mm_set_epi64x(0ULL, 0x0804ULL),
_mm_set_epi64x(0ULL, 0x0009ULL),
_mm_set_epi64x(0ULL, 0x0402ULL),
_mm_set_epi64x(0ULL, 0x0c04ULL),
_mm_set_epi64x(0ULL, 0x0c09ULL),
_mm_set_epi64x(0ULL, 0x0803ULL),
_mm_set_epi64x(0ULL, 0x0007ULL),
_mm_set_epi64x(0ULL, 0x000eULL),
_mm_set_epi64x(0ULL, 0x040cULL),
_mm_set_epi64x(0ULL, 0x0808ULL),
_mm_set_epi64x(0ULL, 0x0401ULL),
_mm_set_epi64x(0ULL, 0x0c02ULL)
};

static const __m128i r128_RC_Perm[R] = 
{
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000400ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x00000c0000000000ULL),
_mm_set_epi64x(0x0000000c01000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x030c000000000000ULL),
_mm_set_epi64x(0x0000000000000c07ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x00000c00000f0000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x000000080f000000ULL),
_mm_set_epi64x(0x0f04000000000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x000000000000080eULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x00000400000d0000ULL),
_mm_set_epi64x(0x000000080a000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0504000000000000ULL),
_mm_set_epi64x(0x0000000000000c0aULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000080000050000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x000000000b000000ULL),
_mm_set_epi64x(0x0604000000000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000c0cULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000080000090000ULL),
_mm_set_epi64x(0x0000000403000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x060c000000000000ULL),
_mm_set_epi64x(0x0000000000000c0dULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x00000800000b0000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000407000000ULL),
_mm_set_epi64x(0x0e0c000000000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x000000000000080dULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x00000400000b0000ULL),
_mm_set_epi64x(0x0000000806000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0d00000000000000ULL),
_mm_set_epi64x(0x000000000000040aULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000080000040000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000009000000ULL),
_mm_set_epi64x(0x0204000000000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000c04ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x00000c0000090000ULL),
_mm_set_epi64x(0x0000000803000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0700000000000000ULL),
_mm_set_epi64x(0x000000000000000eULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x00000400000c0000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000808000000ULL),
_mm_set_epi64x(0x0104000000000000ULL, 0x0000000000000000ULL),
_mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000c02ULL),
};

static const __m256i r256_RC[R] = 
{
_mm256_set_epi64x(0ULL, 0x0400ULL, 0ULL, 0x0400ULL),
_mm256_set_epi64x(0ULL, 0x0c00ULL, 0ULL, 0x0c00ULL),
_mm256_set_epi64x(0ULL, 0x0c01ULL, 0ULL, 0x0c01ULL),
_mm256_set_epi64x(0ULL, 0x0c03ULL, 0ULL, 0x0c03ULL),
_mm256_set_epi64x(0ULL, 0x0c07ULL, 0ULL, 0x0c07ULL),
_mm256_set_epi64x(0ULL, 0x0c0fULL, 0ULL, 0x0c0fULL),
_mm256_set_epi64x(0ULL, 0x080fULL, 0ULL, 0x080fULL),
_mm256_set_epi64x(0ULL, 0x040fULL, 0ULL, 0x040fULL),
_mm256_set_epi64x(0ULL, 0x080eULL, 0ULL, 0x080eULL),
_mm256_set_epi64x(0ULL, 0x040dULL, 0ULL, 0x040dULL),
_mm256_set_epi64x(0ULL, 0x080aULL, 0ULL, 0x080aULL),
_mm256_set_epi64x(0ULL, 0x0405ULL, 0ULL, 0x0405ULL),
_mm256_set_epi64x(0ULL, 0x0c0aULL, 0ULL, 0x0c0aULL),
_mm256_set_epi64x(0ULL, 0x0805ULL, 0ULL, 0x0805ULL),
_mm256_set_epi64x(0ULL, 0x000bULL, 0ULL, 0x000bULL),
_mm256_set_epi64x(0ULL, 0x0406ULL, 0ULL, 0x0406ULL),
_mm256_set_epi64x(0ULL, 0x0c0cULL, 0ULL, 0x0c0cULL),
_mm256_set_epi64x(0ULL, 0x0809ULL, 0ULL, 0x0809ULL),
_mm256_set_epi64x(0ULL, 0x0403ULL, 0ULL, 0x0403ULL),
_mm256_set_epi64x(0ULL, 0x0c06ULL, 0ULL, 0x0c06ULL),
_mm256_set_epi64x(0ULL, 0x0c0dULL, 0ULL, 0x0c0dULL),
_mm256_set_epi64x(0ULL, 0x080bULL, 0ULL, 0x080bULL),
_mm256_set_epi64x(0ULL, 0x0407ULL, 0ULL, 0x0407ULL),
_mm256_set_epi64x(0ULL, 0x0c0eULL, 0ULL, 0x0c0eULL),
_mm256_set_epi64x(0ULL, 0x080dULL, 0ULL, 0x080dULL),
_mm256_set_epi64x(0ULL, 0x040bULL, 0ULL, 0x040bULL),
_mm256_set_epi64x(0ULL, 0x0806ULL, 0ULL, 0x0806ULL),
_mm256_set_epi64x(0ULL, 0x000dULL, 0ULL, 0x000dULL),
_mm256_set_epi64x(0ULL, 0x040aULL, 0ULL, 0x040aULL),
_mm256_set_epi64x(0ULL, 0x0804ULL, 0ULL, 0x0804ULL),
_mm256_set_epi64x(0ULL, 0x0009ULL, 0ULL, 0x0009ULL),
_mm256_set_epi64x(0ULL, 0x0402ULL, 0ULL, 0x0402ULL),
_mm256_set_epi64x(0ULL, 0x0c04ULL, 0ULL, 0x0c04ULL),
_mm256_set_epi64x(0ULL, 0x0c09ULL, 0ULL, 0x0c09ULL),
_mm256_set_epi64x(0ULL, 0x0803ULL, 0ULL, 0x0803ULL),
_mm256_set_epi64x(0ULL, 0x0007ULL, 0ULL, 0x0007ULL),
_mm256_set_epi64x(0ULL, 0x000eULL, 0ULL, 0x000eULL),
_mm256_set_epi64x(0ULL, 0x040cULL, 0ULL, 0x040cULL),
_mm256_set_epi64x(0ULL, 0x0808ULL, 0ULL, 0x0808ULL),
_mm256_set_epi64x(0ULL, 0x0401ULL, 0ULL, 0x0401ULL),
_mm256_set_epi64x(0ULL, 0x0c02ULL, 0ULL, 0x0c02ULL)
};

static const __m256i r256_RC_Perm[R] = 
{
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000000000400ULL, 0x0000000000000000ULL, 0x0000000000000400ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x00000c0000000000ULL, 0x0000000000000000ULL, 0x00000c0000000000ULL),
_mm256_set_epi64x(0x0000000c01000000ULL, 0x0000000000000000ULL, 0x0000000c01000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x030c000000000000ULL, 0x0000000000000000ULL, 0x030c000000000000ULL),
_mm256_set_epi64x(0x0000000000000c07ULL, 0x0000000000000000ULL, 0x0000000000000c07ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x00000c00000f0000ULL, 0x0000000000000000ULL, 0x00000c00000f0000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x000000080f000000ULL, 0x0000000000000000ULL, 0x000000080f000000ULL),
_mm256_set_epi64x(0x0f04000000000000ULL, 0x0000000000000000ULL, 0x0f04000000000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x000000000000080eULL, 0x0000000000000000ULL, 0x000000000000080eULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x00000400000d0000ULL, 0x0000000000000000ULL, 0x00000400000d0000ULL),
_mm256_set_epi64x(0x000000080a000000ULL, 0x0000000000000000ULL, 0x000000080a000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0504000000000000ULL, 0x0000000000000000ULL, 0x0504000000000000ULL),
_mm256_set_epi64x(0x0000000000000c0aULL, 0x0000000000000000ULL, 0x0000000000000c0aULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000080000050000ULL, 0x0000000000000000ULL, 0x0000080000050000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x000000000b000000ULL, 0x0000000000000000ULL, 0x000000000b000000ULL),
_mm256_set_epi64x(0x0604000000000000ULL, 0x0000000000000000ULL, 0x0604000000000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000000000c0cULL, 0x0000000000000000ULL, 0x0000000000000c0cULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000080000090000ULL, 0x0000000000000000ULL, 0x0000080000090000ULL),
_mm256_set_epi64x(0x0000000403000000ULL, 0x0000000000000000ULL, 0x0000000403000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x060c000000000000ULL, 0x0000000000000000ULL, 0x060c000000000000ULL),
_mm256_set_epi64x(0x0000000000000c0dULL, 0x0000000000000000ULL, 0x0000000000000c0dULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x00000800000b0000ULL, 0x0000000000000000ULL, 0x00000800000b0000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000407000000ULL, 0x0000000000000000ULL, 0x0000000407000000ULL),
_mm256_set_epi64x(0x0e0c000000000000ULL, 0x0000000000000000ULL, 0x0e0c000000000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x000000000000080dULL, 0x0000000000000000ULL, 0x000000000000080dULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x00000400000b0000ULL, 0x0000000000000000ULL, 0x00000400000b0000ULL),
_mm256_set_epi64x(0x0000000806000000ULL, 0x0000000000000000ULL, 0x0000000806000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0d00000000000000ULL, 0x0000000000000000ULL, 0x0d00000000000000ULL),
_mm256_set_epi64x(0x000000000000040aULL, 0x0000000000000000ULL, 0x000000000000040aULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000080000040000ULL, 0x0000000000000000ULL, 0x0000080000040000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000009000000ULL, 0x0000000000000000ULL, 0x0000000009000000ULL),
_mm256_set_epi64x(0x0204000000000000ULL, 0x0000000000000000ULL, 0x0204000000000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000000000c04ULL, 0x0000000000000000ULL, 0x0000000000000c04ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x00000c0000090000ULL, 0x0000000000000000ULL, 0x00000c0000090000ULL),
_mm256_set_epi64x(0x0000000803000000ULL, 0x0000000000000000ULL, 0x0000000803000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0700000000000000ULL, 0x0000000000000000ULL, 0x0700000000000000ULL),
_mm256_set_epi64x(0x000000000000000eULL, 0x0000000000000000ULL, 0x000000000000000eULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x00000400000c0000ULL, 0x0000000000000000ULL, 0x00000400000c0000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000808000000ULL, 0x0000000000000000ULL, 0x0000000808000000ULL),
_mm256_set_epi64x(0x0104000000000000ULL, 0x0000000000000000ULL, 0x0104000000000000ULL, 0x0000000000000000ULL),
_mm256_set_epi64x(0x0000000000000000ULL, 0x0000000000000c02ULL, 0x0000000000000000ULL, 0x0000000000000c02ULL),
};

void print_state(unsigned char *m)
{
    for (int i = 0; i < BLOCK_SIZE_INBYTES; i++)
    {
        printf("%x ", (m[i]>>0)&0xf);
        printf("%x ", (m[i]>>4)&0xf);
    }
    printf("\n");
};

void printRKey(__m128i roundkey, int ri)
{
    unsigned char *roundkey_pt = (unsigned char *) &roundkey;

    printf("%d round key: ", ri);
    for (int i = 0; i < BLOCK_SIZE_INBYTES; i++)
    {
        printf("%x ", (roundkey_pt[i])&0xf);
    }
    printf("\n");
};


void printState(__m128i state_left, __m128i state_right)
{
    unsigned char *left_pt = (unsigned char *) &state_left;
    unsigned char *right_pt = (unsigned char *) &state_right;

    printf("L: ");
    for (int i = 0; i < BLOCK_SIZE_INBYTES; i++)
    {
        printf("%x ", (left_pt[i])&0xf);
    }
    printf("R: ");
    for (int i = 0; i < BLOCK_SIZE_INBYTES; i++)
    {
        printf("%x ", (right_pt[i])&0xf);
    }
    printf("\n");
};

/**
 * @input  input  16 consecutive bytes
 * @output low   16 nibbles <= 8 bytes (less significant) of input
 * @output high  16 nibbles <= 8 bytes (most significant) of input
 * input: fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l
 * low  = *7h *7l *6h *6l *5h *5l *4h *4l *3h *3l *2h *2l *1h *1l *0h *0l
 * high = *fh *fl *eh *el *dh *dl *ch *cl *bh *bl *ah *al *9h *9l *8h *8l
 **/
#define packkey(tmp0, tmp1, low, high, input)                                                      \
{                                                                                                  \
    tmp0 = _mm_loadu_si128((__m128i *)(input));                                                    \
    /* tmp0 = fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l */   \
    tmp1 = _mm_srli_epi16(tmp0, 4);                                                                \
    tmp1 = _mm_and_si128(tmp1, MASK_0F);                                                           \
    /* tmp1 = *fh *eh *dh *ch *bh *ah *9h *8h *7h *6h *5h *4h *3h *2h *1h *0h */                   \
    tmp0 = _mm_and_si128(tmp0, MASK_0F);                                                           \
    /* tmp0 = *fl *el *dl *cl *bl *al *9l *8l *7l *6l *5l *4l *3l *2l *1l *0l */                   \
    low  = _mm_unpacklo_epi8(tmp0, tmp1);                                                          \
    /* low  = *7h *7l *6h *6l *5h *5l *4h *4l *3h *3l *2h *2l *1h *1l *0h *0l */                   \
    high = _mm_unpackhi_epi8(tmp0, tmp1);                                                          \
    /* high = *fh *fl *eh *el *dh *dl *ch *cl *bh *bl *ah *al *9h *9l *8h *8l */                   \
}

/**
 * @input  input  16 consecutive bytes
 * @output left   16 even nibbles of input
 * @output right  16  odd nibbles of input
 * input: fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l
 * left  = *fl *el *dl *cl *bl *al *9l *8l *7l *6l *5l *4l *3l *2l *1l *0l
 * right = *fh *eh *dh *ch *bh *ah *9h *8h *7h *6h *5h *4h *3h *2h *1h *0h
 **/
#define pack(left, right, input)                                                                        \
{                                                                                                       \
    left     = _mm_loadu_si128((__m128i *)(input));                                                     \
    /* left  = fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l */       \
    right    = _mm_srli_epi16(left, 4);                                                                 \
    right    = _mm_and_si128(right, MASK_0F);                                                           \
    /* right = *fh *eh *dh *ch *bh *ah *9h *8h *7h *6h *5h *4h *3h *2h *1h *0h */                       \
    left     = _mm_and_si128(left, MASK_0F);                                                            \
    /* left  = *fl *el *dl *cl *bl *al *9l *8l *7l *6l *5l *4l *3l *2l *1l *0l */                       \
}

#define Double_pack(left, right, input)                               \
{                                                                     \
    left  = _mm256_loadu_si256((__m256i *)(input));                   \
    right = _mm256_srli_epi16(left, 4);                               \
    right = _mm256_and_si256(right, Double_MASK_0F);                  \
    left  = _mm256_and_si256(left, Double_MASK_0F);                   \
}

#define Four_pack(left0, left1, right0, right1, input)                \
{                                                                     \
    left0  = _mm256_loadu_si256((__m256i *)(input) + 0);              \
    left1  = _mm256_loadu_si256((__m256i *)(input) + 1);              \
    right0 = _mm256_srli_epi16(left0, 4);                             \
    right1 = _mm256_srli_epi16(left1, 4);                             \
    right0 = _mm256_and_si256(right0, Double_MASK_0F);                \
    right1 = _mm256_and_si256(right1, Double_MASK_0F);                \
    left0  = _mm256_and_si256(left0, Double_MASK_0F);                 \
    left1  = _mm256_and_si256(left1, Double_MASK_0F);                 \
}

#define Eight_pack(                 \
    left0, left1, left2, left3,     \
    right0, right1, right2, right3, \
    input)                          \
{                                                         \
    left0  = _mm256_loadu_si256((__m256i *)(input) + 0);  \
    left1  = _mm256_loadu_si256((__m256i *)(input) + 1);  \
    left2  = _mm256_loadu_si256((__m256i *)(input) + 2);  \
    left3  = _mm256_loadu_si256((__m256i *)(input) + 3);  \
    right0 = _mm256_srli_epi16(left0, 4);                 \
    right1 = _mm256_srli_epi16(left1, 4);                 \
    right2 = _mm256_srli_epi16(left2, 4);                 \
    right3 = _mm256_srli_epi16(left3, 4);                 \
    right0 = _mm256_and_si256(right0, Double_MASK_0F);    \
    right1 = _mm256_and_si256(right1, Double_MASK_0F);    \
    right2 = _mm256_and_si256(right2, Double_MASK_0F);    \
    right3 = _mm256_and_si256(right3, Double_MASK_0F);    \
    left0  = _mm256_and_si256(left0, Double_MASK_0F);     \
    left1  = _mm256_and_si256(left1, Double_MASK_0F);     \
    left2  = _mm256_and_si256(left2, Double_MASK_0F);     \
    left3  = _mm256_and_si256(left3, Double_MASK_0F);     \
}


/**
 * @output left         16 even nibbles of output
 * @output right        16  odd nibbles of output
 * @input  output       16 consecutive bytes
 * left  = *fl *el *dl *cl *bl *al *9l *8l *7l *6l *5l *4l *3l *2l *1l *0l
 * right = *fh *eh *dh *ch *bh *ah *9h *8h *7h *6h *5h *4h *3h *2h *1h *0h
 * output: fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l
 **/
#define unpack(output, left, right)                                                                 \
{                                                                                                   \
    left     = _mm_and_si128(left, MASK_0F);                                                        \
    /* left  = *fl *el *dl *cl *bl *al *9l *8l *7l *6l *5l *4l *3l *2l *1l *0l */                   \
    right    = _mm_and_si128(right, MASK_0F);                                                       \
    /* right = *fh *eh *dh *ch *bh *ah *9h *8h *7h *6h *5h *4h *3h *2h *1h *0h */                   \
    right    = _mm_slli_epi16(right, 4);                                                            \
    /* right = fh* eh* dh* ch* bh* ah* 9h* 8h* 7h* 6h* 5h* 4h* 3h* 2h* 1h* 0h* */                   \
    right    = _mm_or_si128(right, left);                                                           \
    /* right = fhfl ehel dhdl chcl bhbl ahal 9h9l 8h8l 7h7l 6h6l 5h5l 4h4l 3h3l 2h2l 1h1l 0h0l */   \
    _mm_storeu_si128((__m128i *)(output), right);                                                   \
}

#define Double_unpack(output, left, right)                                          \
{                                                                                   \
    left     = _mm256_and_si256(left, Double_MASK_0F);                              \
    right    = _mm256_and_si256(right, Double_MASK_0F);                             \
    right    = _mm256_slli_epi16(right, 4);                                         \
    right    = _mm256_or_si256(right, left);                                        \
    _mm256_storeu_si256((__m256i *)(output), right);                                \
}

#define Four_unpack(output, left0, left1, right0, right1)               \
{                                                                       \
    left0     = _mm256_and_si256(left0, Double_MASK_0F);                \
    right0    = _mm256_and_si256(right0, Double_MASK_0F);               \
    right0    = _mm256_slli_epi16(right0, 4);                           \
    right0    = _mm256_or_si256(right0, left0);                         \
    _mm256_storeu_si256((__m256i *)(output) + 0, right0);               \
    left1     = _mm256_and_si256(left1, Double_MASK_0F);                \
    right1    = _mm256_and_si256(right1, Double_MASK_0F);               \
    right1    = _mm256_slli_epi16(right1, 4);                           \
    right1    = _mm256_or_si256(right1, left1);                         \
    _mm256_storeu_si256((__m256i *)(output) + 1, right1);               \
}

#define Eight_unpack(                                                   \
    output,                                                             \
    left0, left1, left2, left3,                                         \
    right0, right1, right2, right3)                                     \
{                                                                       \
    left0     = _mm256_and_si256(left0, Double_MASK_0F);                \
    right0    = _mm256_and_si256(right0, Double_MASK_0F);               \
    right0    = _mm256_slli_epi16(right0, 4);                           \
    right0    = _mm256_or_si256(right0, left0);                         \
    _mm256_storeu_si256((__m256i *)(output) + 0, right0);               \
    left1     = _mm256_and_si256(left1, Double_MASK_0F);                \
    right1    = _mm256_and_si256(right1, Double_MASK_0F);               \
    right1    = _mm256_slli_epi16(right1, 4);                           \
    right1    = _mm256_or_si256(right1, left1);                         \
    _mm256_storeu_si256((__m256i *)(output) + 1, right1);               \
    left2     = _mm256_and_si256(left2, Double_MASK_0F);                \
    right2    = _mm256_and_si256(right2, Double_MASK_0F);               \
    right2    = _mm256_slli_epi16(right2, 4);                           \
    right2    = _mm256_or_si256(right2, left2);                         \
    _mm256_storeu_si256((__m256i *)(output) + 2, right2);               \
    left3     = _mm256_and_si256(left3, Double_MASK_0F);                \
    right3    = _mm256_and_si256(right3, Double_MASK_0F);               \
    right3    = _mm256_slli_epi16(right3, 4);                           \
    right3    = _mm256_or_si256(right3, left3);                         \
    _mm256_storeu_si256((__m256i *)(output) + 3, right3);               \
}

#define First_F(tmp, left, right, key)                    \
{                                                         \
    /* Add round constant and round key */                \
    right = _mm_xor_si128(right, key);                    \
    right = _mm_xor_si128(right, *r128_RC_pt++);          \
    /* Left branches: Sbox */                             \
    tmp = _mm_shuffle_epi8(Sbox, left);                   \
    /* Right branches xor Left branches */                \
    right = _mm_xor_si128(right, tmp);                    \
}

#define First_F_RCRK(tmp, left, right, key)               \
{                                                         \
    /* Add round constant and round key */                \
    right = _mm_xor_si128(right, key);                    \
    /* Left branches: Sbox */                             \
    tmp = _mm_shuffle_epi8(Sbox, left);                   \
    /* Right branches xor Left branches */                \
    right = _mm_xor_si128(right, tmp);                    \
}

#define ROUND_F(tmp, left, right, key, eqperm)            \
{                                                         \
    /* Add round constant and round key */                \
    right = _mm_xor_si128(right, key);                    \
    right = _mm_xor_si128(right, *r128_RC_pt++);          \
    /* Left branches: Perm */                             \
    left = _mm_shuffle_epi8(left, eqperm);                \
    /* Left branches: Sbox */                             \
    tmp = _mm_shuffle_epi8(Sbox, left);                   \
    /* Right branches xor Left branches */                \
    right = _mm_xor_si128(right, tmp);                    \
}

#define ROUND_F_RCRK(tmp, left, right, key, eqperm)       \
{                                                         \
    /* Add round constant and round key */                \
    right = _mm_xor_si128(right, key);                    \
    /* Left branches: Perm */                             \
    left = _mm_shuffle_epi8(left, eqperm);                \
    /* Left branches: Sbox */                             \
    tmp = _mm_shuffle_epi8(Sbox, left);                   \
    /* Right branches xor Left branches */                \
    right = _mm_xor_si128(right, tmp);                    \
}


#define Double_First_F(tmp, left, right, key)             \
{                                                         \
    right = _mm256_xor_si256(right, key);                 \
    right = _mm256_xor_si256(right, *r256_RC_pt++);       \
    tmp = _mm256_shuffle_epi8(Double_Sbox, left);         \
    right = _mm256_xor_si256(right, tmp);                 \
}

#define Double_First_F_RCRK(tmp, left, right, key)        \
{                                                         \
    right = _mm256_xor_si256(right, key);                 \
    tmp = _mm256_shuffle_epi8(Double_Sbox, left);         \
    right = _mm256_xor_si256(right, tmp);                 \
}

#define Double_ROUND_F(tmp, left, right, key, eqperm)     \
{                                                         \
    right = _mm256_xor_si256(right, key);                 \
    right = _mm256_xor_si256(right, *r256_RC_pt++);       \
    left = _mm256_shuffle_epi8(left, eqperm);             \
    tmp = _mm256_shuffle_epi8(Double_Sbox, left);         \
    right = _mm256_xor_si256(right, tmp);                 \
}

#define Double_ROUND_F_RCRK(tmp, left, right, key, eqperm)\
{                                                         \
    right = _mm256_xor_si256(right, key);                 \
    left = _mm256_shuffle_epi8(left, eqperm);             \
    tmp = _mm256_shuffle_epi8(Double_Sbox, left);         \
    right = _mm256_xor_si256(right, tmp);                 \
}

#define Four_First_F(tmp0, tmp1, left0, left1, right0, right1, key)         \
{                                                                           \
    right0 = _mm256_xor_si256(right0, key);                                 \
    right1 = _mm256_xor_si256(right1, key);                                 \
    right0 = _mm256_xor_si256(right0, *r256_RC_pt);                         \
    right1 = _mm256_xor_si256(right1, *r256_RC_pt++);                       \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);                         \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);                         \
    right0 = _mm256_xor_si256(right0, tmp0);                                \
    right1 = _mm256_xor_si256(right1, tmp1);                                \
}

#define Four_First_F_RCRK(tmp0, tmp1, left0, left1, right0, right1, key)    \
{                                                                           \
    right0 = _mm256_xor_si256(right0, key);                                 \
    right1 = _mm256_xor_si256(right1, key);                                 \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);                         \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);                         \
    right0 = _mm256_xor_si256(right0, tmp0);                                \
    right1 = _mm256_xor_si256(right1, tmp1);                                \
}

#define Four_ROUND_F(tmp0, tmp1, left0, left1, right0, right1, key, eqperm) \
{                                                                           \
    right0 = _mm256_xor_si256(right0, key);                                 \
    right1 = _mm256_xor_si256(right1, key);                                 \
    right0 = _mm256_xor_si256(right0, *r256_RC_pt);                         \
    right1 = _mm256_xor_si256(right1, *r256_RC_pt++);                       \
    left0 = _mm256_shuffle_epi8(left0, eqperm);                             \
    left1 = _mm256_shuffle_epi8(left1, eqperm);                             \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);                         \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);                         \
    right0 = _mm256_xor_si256(right0, tmp0);                                \
    right1 = _mm256_xor_si256(right1, tmp1);                                \
}

#define Four_ROUND_F_RCRK(tmp0, tmp1, left0, left1, right0, right1, key, eqperm) \
{                                                                           \
    right0 = _mm256_xor_si256(right0, key);                                 \
    right1 = _mm256_xor_si256(right1, key);                                 \
    left0 = _mm256_shuffle_epi8(left0, eqperm);                             \
    left1 = _mm256_shuffle_epi8(left1, eqperm);                             \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);                         \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);                         \
    right0 = _mm256_xor_si256(right0, tmp0);                                \
    right1 = _mm256_xor_si256(right1, tmp1);                                \
}

#define Eight_First_F(                                  \
    tmp0, tmp1, tmp2, tmp3,                             \
    left0, left1, left2, left3,                         \
    right0, right1, right2, right3,                     \
    key)                                                \
{                                                       \
    right0 = _mm256_xor_si256(right0, key);             \
    right1 = _mm256_xor_si256(right1, key);             \
    right2 = _mm256_xor_si256(right2, key);             \
    right3 = _mm256_xor_si256(right3, key);             \
    right0 = _mm256_xor_si256(right0, *r256_RC_pt);     \
    right1 = _mm256_xor_si256(right1, *r256_RC_pt);     \
    right2 = _mm256_xor_si256(right2, *r256_RC_pt);     \
    right3 = _mm256_xor_si256(right3, *r256_RC_pt++);   \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);     \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);     \
    tmp2 = _mm256_shuffle_epi8(Double_Sbox, left2);     \
    tmp3 = _mm256_shuffle_epi8(Double_Sbox, left3);     \
    right0 = _mm256_xor_si256(right0, tmp0);            \
    right1 = _mm256_xor_si256(right1, tmp1);            \
    right2 = _mm256_xor_si256(right2, tmp2);            \
    right3 = _mm256_xor_si256(right3, tmp3);            \
}

#define Eight_First_F_RCRK(                             \
    tmp0, tmp1, tmp2, tmp3,                             \
    left0, left1, left2, left3,                         \
    right0, right1, right2, right3,                     \
    key)                                                \
{                                                       \
    right0 = _mm256_xor_si256(right0, key);             \
    right1 = _mm256_xor_si256(right1, key);             \
    right2 = _mm256_xor_si256(right2, key);             \
    right3 = _mm256_xor_si256(right3, key);             \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);     \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);     \
    tmp2 = _mm256_shuffle_epi8(Double_Sbox, left2);     \
    tmp3 = _mm256_shuffle_epi8(Double_Sbox, left3);     \
    right0 = _mm256_xor_si256(right0, tmp0);            \
    right1 = _mm256_xor_si256(right1, tmp1);            \
    right2 = _mm256_xor_si256(right2, tmp2);            \
    right3 = _mm256_xor_si256(right3, tmp3);            \
}

#define Eight_ROUND_F(                                  \
    tmp0, tmp1, tmp2, tmp3,                             \
    left0, left1, left2, left3,                         \
    right0, right1, right2, right3,                     \
    key, eqperm)                                        \
{                                                       \
    right0 = _mm256_xor_si256(right0, key);             \
    right1 = _mm256_xor_si256(right1, key);             \
    right2 = _mm256_xor_si256(right2, key);             \
    right3 = _mm256_xor_si256(right3, key);             \
    right0 = _mm256_xor_si256(right0, *r256_RC_pt);     \
    right1 = _mm256_xor_si256(right1, *r256_RC_pt);     \
    right2 = _mm256_xor_si256(right2, *r256_RC_pt);     \
    right3 = _mm256_xor_si256(right3, *r256_RC_pt++);   \
    left0 = _mm256_shuffle_epi8(left0, eqperm);         \
    left1 = _mm256_shuffle_epi8(left1, eqperm);         \
    left2 = _mm256_shuffle_epi8(left2, eqperm);         \
    left3 = _mm256_shuffle_epi8(left3, eqperm);         \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);     \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);     \
    tmp2 = _mm256_shuffle_epi8(Double_Sbox, left2);     \
    tmp3 = _mm256_shuffle_epi8(Double_Sbox, left3);     \
    right0 = _mm256_xor_si256(right0, tmp0);            \
    right1 = _mm256_xor_si256(right1, tmp1);            \
    right2 = _mm256_xor_si256(right2, tmp2);            \
    right3 = _mm256_xor_si256(right3, tmp3);            \
}

#define Eight_ROUND_F_RCRK(                             \
    tmp0, tmp1, tmp2, tmp3,                             \
    left0, left1, left2, left3,                         \
    right0, right1, right2, right3,                     \
    key, eqperm)                                        \
{                                                       \
    right0 = _mm256_xor_si256(right0, key);             \
    right1 = _mm256_xor_si256(right1, key);             \
    right2 = _mm256_xor_si256(right2, key);             \
    right3 = _mm256_xor_si256(right3, key);             \
    left0 = _mm256_shuffle_epi8(left0, eqperm);         \
    left1 = _mm256_shuffle_epi8(left1, eqperm);         \
    left2 = _mm256_shuffle_epi8(left2, eqperm);         \
    left3 = _mm256_shuffle_epi8(left3, eqperm);         \
    tmp0 = _mm256_shuffle_epi8(Double_Sbox, left0);     \
    tmp1 = _mm256_shuffle_epi8(Double_Sbox, left1);     \
    tmp2 = _mm256_shuffle_epi8(Double_Sbox, left2);     \
    tmp3 = _mm256_shuffle_epi8(Double_Sbox, left3);     \
    right0 = _mm256_xor_si256(right0, tmp0);            \
    right1 = _mm256_xor_si256(right1, tmp1);            \
    right2 = _mm256_xor_si256(right2, tmp2);            \
    right3 = _mm256_xor_si256(right3, tmp3);            \
}

void permRC()
{
    __m128i tmp;
    uint64_t * pt = (uint64_t *)&tmp;

    cout << hex << setfill('0');

    tmp = r128_RC[0]; cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
    for (int i = 0; i < 5; i++)
    {
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 1], EqKeyPerm0); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 2], EqKeyPerm1); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 3], EqKeyPerm2); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 4], EqKeyPerm3); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 5], EqKeyPerm4); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 6], EqKeyPerm5); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = _mm_shuffle_epi8(r128_RC[(i << 3) + 7], EqKeyPerm6); cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
        tmp = r128_RC[(i << 3) + 8]; cout << "0x" << setw(16) << pt[1] << "ULL, " << "0x" << setw(16) << pt[0] << "ULL" << endl;
    }
}

int ecb_enc(unsigned char *m, size_t in_len, unsigned char *c, unsigned char *k)
{
    __m128i state_left;
    __m128i state_right;
    __m128i master_key[2];
    __m128i round_key[R];
    __m128i state_tmp0;
    __m128i state_tmp1;
    __m128i static const * r128_RC_pt = r128_RC_Perm + R;
#if (PIPE >= 2)
    __m256i double_state_left0 ;
    __m256i double_state_right0;
    __m256i double_state_tmp00 ;
    __m256i double_state_tmp10 ;
    __m256i double_round_key[R];
    __m256i static const * r256_RC_pt = r256_RC_Perm + R;
#endif
#if (PIPE >= 4)
    __m256i double_state_left1 ;
    __m256i double_state_right1;
    __m256i double_state_tmp01 ;
    __m256i double_state_tmp11 ;
#endif
#if (PIPE >= 8)
    __m256i double_state_left2 , double_state_left3 ;
    __m256i double_state_right2, double_state_right3;
    __m256i double_state_tmp02 , double_state_tmp03 ;
    __m256i double_state_tmp12 , double_state_tmp13 ;
#endif

	if ((in_len & (BLOCK_SIZE_INBYTES-1)) != 0)
	{
		printf("mlen should be divisible by %d;\n", BLOCK_SIZE_INBYTES);
		return FAILURE;
	}

    if (in_len < (256 * BLOCK_SIZE_INBYTES))
    {
        packkey(state_tmp0, state_tmp1, round_key[0], round_key[7], k);
        round_key[1] = _mm_shuffle_epi8(round_key[7], EqKeyPerm0);
        round_key[2] = _mm_shuffle_epi8(round_key[0], EqKeyPerm1);
        round_key[3] = _mm_shuffle_epi8(round_key[7], EqKeyPerm2);
        round_key[4] = _mm_shuffle_epi8(round_key[0], EqKeyPerm3);
        round_key[5] = _mm_shuffle_epi8(round_key[7], EqKeyPerm4);
        round_key[6] = _mm_shuffle_epi8(round_key[0], EqKeyPerm5);
        round_key[7] = _mm_shuffle_epi8(round_key[7], EqKeyPerm6);
        #if PRINT_INTER
        for (int ri = 0; ri < 8; ri++)
            printRKey(round_key[ri], ri + 1);
        #endif
    
    #if (PIPE >= 2)
        if (in_len >= 2 * BLOCK_SIZE_INBYTES)
        {
            double_round_key[0] = _mm256_broadcastsi128_si256(round_key[0]);
            double_round_key[1] = _mm256_broadcastsi128_si256(round_key[1]);
            double_round_key[2] = _mm256_broadcastsi128_si256(round_key[2]);
            double_round_key[3] = _mm256_broadcastsi128_si256(round_key[3]);
            double_round_key[4] = _mm256_broadcastsi128_si256(round_key[4]);
            double_round_key[5] = _mm256_broadcastsi128_si256(round_key[5]);
            double_round_key[6] = _mm256_broadcastsi128_si256(round_key[6]);
            double_round_key[7] = _mm256_broadcastsi128_si256(round_key[7]);
        }
    #endif
    
    #if (PIPE >= 8)
        while (in_len >= 8 * BLOCK_SIZE_INBYTES)
        {
            r256_RC_pt -= R;
            Eight_pack( double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                        double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                        m);
    
            Eight_First_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[1], Double_EqPerm0);
    
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[2], Double_EqPerm1);
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[3], Double_EqPerm2);
    
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[4], Double_EqPerm3);
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[5], Double_EqPerm0);
    
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[6], Double_EqPerm1);
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[7], Double_EqPerm2);
    
                Eight_ROUND_F( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[0], Double_EqPerm3);
            }
            Eight_unpack( c, 
                          double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                          double_state_right0, double_state_right1, double_state_right2, double_state_right3);
            m = m + 8 * BLOCK_SIZE_INBYTES;
            c = c + 8 * BLOCK_SIZE_INBYTES;
            in_len -= 8 * BLOCK_SIZE_INBYTES;
        }
    #endif
    #if (PIPE >= 4)
        while (in_len >= 4 * BLOCK_SIZE_INBYTES)
        {
            r256_RC_pt -= R;
            Four_pack( double_state_left0, double_state_left1,
                       double_state_right0, double_state_right1, m);
    
            Four_First_F( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[1], Double_EqPerm0);
    
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[2], Double_EqPerm1);
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[3], Double_EqPerm2);
    
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[4], Double_EqPerm3);
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[5], Double_EqPerm0);
    
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[6], Double_EqPerm1);
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[7], Double_EqPerm2);
    
                Four_ROUND_F( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[0], Double_EqPerm3);
            }
            Four_unpack( c,
                         double_state_left0, double_state_left1,
                         double_state_right0, double_state_right1);
            m = m + 4 * BLOCK_SIZE_INBYTES;
            c = c + 4 * BLOCK_SIZE_INBYTES;
            in_len -= 4 * BLOCK_SIZE_INBYTES;
        }
    #endif
    #if (PIPE >= 2)
        while (in_len >= 2 * BLOCK_SIZE_INBYTES)
        {
            r256_RC_pt -= R;
            Double_pack(double_state_left0, double_state_right0, m);
            Double_First_F(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Double_ROUND_F(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[1], Double_EqPerm0);
                Double_ROUND_F(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[2], Double_EqPerm1);
                Double_ROUND_F(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[3], Double_EqPerm2);
                Double_ROUND_F(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[4], Double_EqPerm3);
                Double_ROUND_F(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[5], Double_EqPerm0);
                Double_ROUND_F(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[6], Double_EqPerm1);
                Double_ROUND_F(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[7], Double_EqPerm2);
                Double_ROUND_F(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[0], Double_EqPerm3);
            }
            Double_unpack(c, double_state_left0, double_state_right0);
            m = m + 2 * BLOCK_SIZE_INBYTES;
            c = c + 2 * BLOCK_SIZE_INBYTES;
            in_len -= 2 * BLOCK_SIZE_INBYTES;
        }
    #endif
        while (in_len != 0)
        {
            r128_RC_pt -= R;
            pack(state_left, state_right, m);
    
            /*first function*/
            #if PRINT_INTER
            printf("%d round\n", 1);
            printState(state_left, state_right);
            #endif
    
            First_F(state_tmp0, state_left, state_right, round_key[0]);
    
            for (int i = 0; i < 5; i++)
            {
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F(state_tmp0, state_right, state_left, round_key[1], EqPerm0);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F(state_tmp0, state_left, state_right, round_key[2], EqPerm1);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F(state_tmp0, state_right, state_left, round_key[3], EqPerm2);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F(state_tmp0, state_left, state_right, round_key[4], EqPerm3);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F(state_tmp0, state_right, state_left, round_key[5], EqPerm0);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F(state_tmp0, state_left, state_right, round_key[6], EqPerm1);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F(state_tmp0, state_right, state_left, round_key[7], EqPerm2);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F(state_tmp0, state_left, state_right, round_key[0], EqPerm3);
            }
            #if PRINT_INTER
            printf("%d round\n", R);
            printState(state_left, state_right);
            #endif
            unpack(c, state_left, state_right);
            m = m + BLOCK_SIZE_INBYTES;
            c = c + BLOCK_SIZE_INBYTES;
            in_len -= BLOCK_SIZE_INBYTES;
        }
    }
    else
    {
        packkey(state_tmp0, state_tmp1, master_key[0], master_key[1], k);

        round_key[0] = master_key[0];
        round_key[1] = _mm_shuffle_epi8(master_key[1], EqKeyPerm0);
        round_key[2] = _mm_shuffle_epi8(master_key[0], EqKeyPerm1);
        round_key[3] = _mm_shuffle_epi8(master_key[1], EqKeyPerm2);
        round_key[4] = _mm_shuffle_epi8(master_key[0], EqKeyPerm3);
        round_key[5] = _mm_shuffle_epi8(master_key[1], EqKeyPerm4);
        round_key[6] = _mm_shuffle_epi8(master_key[0], EqKeyPerm5);
        round_key[7] = _mm_shuffle_epi8(master_key[1], EqKeyPerm6);
        for (int i = 1; i < 5; i++)
        {
            round_key[(i << 3) + 0] = _mm_xor_si128(round_key[0], r128_RC_Perm[(i << 3) + 0]);
            round_key[(i << 3) + 1] = _mm_xor_si128(round_key[1], r128_RC_Perm[(i << 3) + 1]);
            round_key[(i << 3) + 2] = _mm_xor_si128(round_key[2], r128_RC_Perm[(i << 3) + 2]);
            round_key[(i << 3) + 3] = _mm_xor_si128(round_key[3], r128_RC_Perm[(i << 3) + 3]);
            round_key[(i << 3) + 4] = _mm_xor_si128(round_key[4], r128_RC_Perm[(i << 3) + 4]);
            round_key[(i << 3) + 5] = _mm_xor_si128(round_key[5], r128_RC_Perm[(i << 3) + 5]);
            round_key[(i << 3) + 6] = _mm_xor_si128(round_key[6], r128_RC_Perm[(i << 3) + 6]);
            round_key[(i << 3) + 7] = _mm_xor_si128(round_key[7], r128_RC_Perm[(i << 3) + 7]);
        }
        round_key[R-1] = _mm_xor_si128(round_key[0], r128_RC_Perm[R-1]);
        round_key[0] = _mm_xor_si128(round_key[0], r128_RC_Perm[0]);
        round_key[1] = _mm_xor_si128(round_key[1], r128_RC_Perm[1]);
        round_key[2] = _mm_xor_si128(round_key[2], r128_RC_Perm[2]);
        round_key[3] = _mm_xor_si128(round_key[3], r128_RC_Perm[3]);
        round_key[4] = _mm_xor_si128(round_key[4], r128_RC_Perm[4]);
        round_key[5] = _mm_xor_si128(round_key[5], r128_RC_Perm[5]);
        round_key[6] = _mm_xor_si128(round_key[6], r128_RC_Perm[6]);
        round_key[7] = _mm_xor_si128(round_key[7], r128_RC_Perm[7]);

        #if PRINT_INTER
        for (int ri = 0; ri < R; ri++)
            printRKey(round_key[ri], ri + 1);
        #endif
    
    #if (PIPE >= 2)
        if (in_len >= 2 * BLOCK_SIZE_INBYTES)
        {
            double_round_key[0] = _mm256_broadcastsi128_si256(round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                double_round_key[(i << 3) + 1] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 1]);
                double_round_key[(i << 3) + 2] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 2]);
                double_round_key[(i << 3) + 3] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 3]);
                double_round_key[(i << 3) + 4] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 4]);
                double_round_key[(i << 3) + 5] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 5]);
                double_round_key[(i << 3) + 6] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 6]);
                double_round_key[(i << 3) + 7] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 7]);
                double_round_key[(i << 3) + 8] = _mm256_broadcastsi128_si256(round_key[(i << 3) + 8]);
            }
            
        }
    #endif
    
    #if (PIPE >= 8)
        while (in_len >= 8 * BLOCK_SIZE_INBYTES)
        {
            Eight_pack( double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                        double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                        m);
    
            Eight_First_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[(i << 3) + 1], Double_EqPerm0);
    
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[(i << 3) + 2], Double_EqPerm1);
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[(i << 3) + 3], Double_EqPerm2);
    
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[(i << 3) + 4], Double_EqPerm3);
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[(i << 3) + 5], Double_EqPerm0);
    
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[(i << 3) + 6], Double_EqPerm1);
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                               double_round_key[(i << 3) + 7], Double_EqPerm2);
    
                Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                               double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                               double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                               double_round_key[(i << 3) + 8], Double_EqPerm3);
            }
            Eight_unpack( c, 
                          double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                          double_state_right0, double_state_right1, double_state_right2, double_state_right3);
            m = m + 8 * BLOCK_SIZE_INBYTES;
            c = c + 8 * BLOCK_SIZE_INBYTES;
            in_len -= 8 * BLOCK_SIZE_INBYTES;
        }
    #endif
    #if (PIPE >= 4)
        while (in_len >= 4 * BLOCK_SIZE_INBYTES)
        {
            Four_pack( double_state_left0, double_state_left1,
                       double_state_right0, double_state_right1, m);
    
            Four_First_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[(i << 3) + 1], Double_EqPerm0);
    
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[(i << 3) + 2], Double_EqPerm1);
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[(i << 3) + 3], Double_EqPerm2);
    
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[(i << 3) + 4], Double_EqPerm3);
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[(i << 3) + 5], Double_EqPerm0);
    
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[(i << 3) + 6], Double_EqPerm1);
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_right0, double_state_right1,
                              double_state_left0, double_state_left1,
                              double_round_key[(i << 3) + 7], Double_EqPerm2);
    
                Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                              double_state_left0, double_state_left1,
                              double_state_right0,  double_state_right1,
                              double_round_key[(i << 3) + 8], Double_EqPerm3);
            }
            Four_unpack( c,
                         double_state_left0, double_state_left1,
                         double_state_right0, double_state_right1);
            m = m + 4 * BLOCK_SIZE_INBYTES;
            c = c + 4 * BLOCK_SIZE_INBYTES;
            in_len -= 4 * BLOCK_SIZE_INBYTES;
        }
    #endif
    #if (PIPE >= 2)
        while (in_len >= 2 * BLOCK_SIZE_INBYTES)
        {
            Double_pack(double_state_left0, double_state_right0, m);
            Double_First_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[0]);
            for (int i = 0; i < 5; i++)
            {
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 1], Double_EqPerm0);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 2], Double_EqPerm1);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 3], Double_EqPerm2);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 4], Double_EqPerm3);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 5], Double_EqPerm0);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 6], Double_EqPerm1);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 7], Double_EqPerm2);
                Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 8], Double_EqPerm3);
            }
            Double_unpack(c, double_state_left0, double_state_right0);
            m = m + 2 * BLOCK_SIZE_INBYTES;
            c = c + 2 * BLOCK_SIZE_INBYTES;
            in_len -= 2 * BLOCK_SIZE_INBYTES;
        }
    #endif
        while (in_len != 0)
        {
            pack(state_left, state_right, m);
    
            /*first function*/
            #if PRINT_INTER
            printf("%d round\n", 1);
            printState(state_left, state_right);
            #endif
    
            First_F_RCRK(state_tmp0, state_left, state_right, round_key[0]);
    
            for (int i = 0; i < 5; i++)
            {
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 1], EqPerm0);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 2], EqPerm1);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 3], EqPerm2);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 4], EqPerm3);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 5], EqPerm0);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 6], EqPerm1);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_right, state_left);
                #endif
                ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 7], EqPerm2);
                #if PRINT_INTER
                printf("%d round\n", i * 8 + 1 + 1);
                printState(state_left, state_right);
                #endif
                ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 8], EqPerm3);
            }
            #if PRINT_INTER
            printf("%d round\n", R);
            printState(state_left, state_right);
            #endif
            unpack(c, state_left, state_right);
            m = m + BLOCK_SIZE_INBYTES;
            c = c + BLOCK_SIZE_INBYTES;
            in_len -= BLOCK_SIZE_INBYTES;
        }
    }
    

    return SUCCESS;
}

int  ecb_only_enc(unsigned char *m, size_t in_len, unsigned char *c, __m128i round_key[R], __m256i double_round_key[R])
{
    __m128i state_left;
    __m128i state_right;
    __m128i state_tmp0;
    __m128i state_tmp1;

#if (PIPE >= 2)
    __m256i double_state_left0 ;
    __m256i double_state_right0;
    __m256i double_state_tmp00 ;
    __m256i double_state_tmp10 ;
#endif
#if (PIPE >= 4)
    __m256i double_state_left1 ;
    __m256i double_state_right1;
    __m256i double_state_tmp01 ;
    __m256i double_state_tmp11 ;
#endif
#if (PIPE >= 8)
    __m256i double_state_left2 , double_state_left3 ;
    __m256i double_state_right2, double_state_right3;
    __m256i double_state_tmp02 , double_state_tmp03 ;
    __m256i double_state_tmp12 , double_state_tmp13 ;
#endif

	if ((in_len & (BLOCK_SIZE_INBYTES-1)) != 0)
	{
		printf("mlen should be divisible by %d;\n", BLOCK_SIZE_INBYTES);
		return FAILURE;
	}

#if (PIPE >= 8)
    while (in_len >= 8 * BLOCK_SIZE_INBYTES)
    {
        Eight_pack( double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                    double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                    m);
        Eight_First_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                       double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                       double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                       double_round_key[0]);
        for (int i = 0; i < 5; i++)
        {
            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                           double_round_key[(i << 3) + 1], Double_EqPerm0);

            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[(i << 3) + 2], Double_EqPerm1);
            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                           double_round_key[(i << 3) + 3], Double_EqPerm2);

            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[(i << 3) + 4], Double_EqPerm3);
            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                           double_round_key[(i << 3) + 5], Double_EqPerm0);

            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[(i << 3) + 6], Double_EqPerm1);
            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_right0, double_state_right1, double_state_right2, double_state_right3,
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                           double_round_key[(i << 3) + 7], Double_EqPerm2);

            Eight_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01, double_state_tmp02, double_state_tmp03, 
                           double_state_left0, double_state_left1, double_state_left2, double_state_left3, 
                           double_state_right0,  double_state_right1, double_state_right2,  double_state_right3,
                           double_round_key[(i << 3) + 8], Double_EqPerm3);
        }
        Eight_unpack( c, 
                      double_state_left0, double_state_left1, double_state_left2, double_state_left3,
                      double_state_right0, double_state_right1, double_state_right2, double_state_right3);
        m = m + 8 * BLOCK_SIZE_INBYTES;
        c = c + 8 * BLOCK_SIZE_INBYTES;
        in_len -= 8 * BLOCK_SIZE_INBYTES;
    }
#endif
#if (PIPE >= 4)
    while (in_len >= 4 * BLOCK_SIZE_INBYTES)
    {
        Four_pack( double_state_left0, double_state_left1,
                   double_state_right0, double_state_right1, m);
        Four_First_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_left0, double_state_left1,
                          double_state_right0,  double_state_right1,
                          double_round_key[0]);
        for (int i = 0; i < 5; i++)
        {
            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_right0, double_state_right1,
                          double_state_left0, double_state_left1,
                          double_round_key[(i << 3) + 1], Double_EqPerm0);

            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_left0, double_state_left1,
                          double_state_right0,  double_state_right1,
                          double_round_key[(i << 3) + 2], Double_EqPerm1);
            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_right0, double_state_right1,
                          double_state_left0, double_state_left1,
                          double_round_key[(i << 3) + 3], Double_EqPerm2);

            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_left0, double_state_left1,
                          double_state_right0,  double_state_right1,
                          double_round_key[(i << 3) + 4], Double_EqPerm3);
            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_right0, double_state_right1,
                          double_state_left0, double_state_left1,
                          double_round_key[(i << 3) + 5], Double_EqPerm0);

            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_left0, double_state_left1,
                          double_state_right0,  double_state_right1,
                          double_round_key[(i << 3) + 6], Double_EqPerm1);
            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_right0, double_state_right1,
                          double_state_left0, double_state_left1,
                          double_round_key[(i << 3) + 7], Double_EqPerm2);

            Four_ROUND_F_RCRK( double_state_tmp00, double_state_tmp01,
                          double_state_left0, double_state_left1,
                          double_state_right0,  double_state_right1,
                          double_round_key[(i << 3) + 8], Double_EqPerm3);
        }
        Four_unpack( c,
                     double_state_left0, double_state_left1,
                     double_state_right0, double_state_right1);
        m = m + 4 * BLOCK_SIZE_INBYTES;
        c = c + 4 * BLOCK_SIZE_INBYTES;
        in_len -= 4 * BLOCK_SIZE_INBYTES;
    }
#endif
#if (PIPE >= 2)
    while (in_len >= 2 * BLOCK_SIZE_INBYTES)
    {
        Double_pack(double_state_left0, double_state_right0, m);
        Double_First_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[0]);
        for (int i = 0; i < 5; i++)
        {
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 1], Double_EqPerm0);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 2], Double_EqPerm1);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 3], Double_EqPerm2);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 4], Double_EqPerm3);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 5], Double_EqPerm0);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 6], Double_EqPerm1);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_right0, double_state_left0,  double_round_key[(i << 3) + 7], Double_EqPerm2);
            Double_ROUND_F_RCRK(double_state_tmp00, double_state_left0,  double_state_right0, double_round_key[(i << 3) + 8], Double_EqPerm3);
        }
        Double_unpack(c, double_state_left0, double_state_right0);
        m = m + 2 * BLOCK_SIZE_INBYTES;
        c = c + 2 * BLOCK_SIZE_INBYTES;
        in_len -= 2 * BLOCK_SIZE_INBYTES;
    }
#endif
    while (in_len != 0)
    {
        pack(state_left, state_right, m);
        First_F_RCRK(state_tmp0, state_left, state_right, round_key[0]);
        for (int i = 0; i < 5; i++)
        {
            ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 1], EqPerm0);
            ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 2], EqPerm1);
            ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 3], EqPerm2);
            ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 4], EqPerm3);
            ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 5], EqPerm0);
            ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 6], EqPerm1);
            ROUND_F_RCRK(state_tmp0, state_right, state_left, round_key[(i << 3) + 7], EqPerm2);
            ROUND_F_RCRK(state_tmp0, state_left, state_right, round_key[(i << 3) + 8], EqPerm3);
        }
        unpack(c, state_left, state_right);
        m = m + BLOCK_SIZE_INBYTES;
        c = c + BLOCK_SIZE_INBYTES;
        in_len -= BLOCK_SIZE_INBYTES;
    }
    return SUCCESS;
}

void gen_test_vector()
{
	unsigned char m[PIPE * BLOCK_SIZE_INBYTES] = { 
#if (PIPE == 8)
        0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
        0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
        0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
        0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
#endif
#if (PIPE >= 4)
        0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
	    0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
#endif
#if (PIPE >= 2)
	    0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
#endif
	    0xfa, 0xc6, 0xdd, 0x09, 0xcf, 0xa5, 0xe6, 0xaa, 0x98, 0xb7, 0xdc, 0x21, 0x80, 0x3d, 0x19, 0x1e,
};
	unsigned char c[PIPE * BLOCK_SIZE_INBYTES];
	unsigned char k[2 * BLOCK_SIZE_INBYTES] = {0xa0, 0xdc, 0x20, 0xf2, 0x86, 0xa0, 0x45, 0xf7, 0xee, 0x30, 0x0c, 0x68, 0xb7, 0x90, 0x3e, 0x7d };


	/*key*/
	printf("key : \n");
	print_state(k);

	/*plaintext*/
	printf("plaintext : \n");
    for (int i = 0; i < PIPE; i++)
    {
        print_state(m + i * BLOCK_SIZE_INBYTES);
    }

	ecb_enc(m, PIPE * BLOCK_SIZE_INBYTES, c, k);

	/*ciphertext*/
	printf("ciphertext : \n");
    for (int i = 0; i < PIPE; i++)
    {
        print_state(c + i * BLOCK_SIZE_INBYTES);
    }
}