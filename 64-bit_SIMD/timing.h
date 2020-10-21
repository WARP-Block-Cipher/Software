#ifndef TIMING_H__
#define TIMING_H__
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <sstream>
#include <stdint.h>
#include <math.h>
#include "WARP_SIMD_code.h"

#ifdef _MSC_VER
unsigned long CurrentProcessorNumber(void);
__inline unsigned long long read_tsc(void);
#endif

#ifdef __GNUC__
inline unsigned long long read_tsc(void);
#endif

void setCPUaffinity();

void block_rndfill(unsigned char *buf, const int len);

int time_base(double *av, double *sig);

int time_enc16(double *av, double *sig, unsigned int k_len, unsigned long long dataLengthInBytes);
int time_only_enc16(double *av, double *sig, unsigned int k_len, unsigned long long dataLengthInBytes);

void timing();

#endif  //TIMING_H__