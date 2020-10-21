#ifndef TIMING_H
#define TIMING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "WARP_SIMD_code.h"

#define NUM_TIMINGS 1000

unsigned long long int start_rdtsc(void)
{
   unsigned long long int x;
   unsigned a, d;

   __asm__ volatile("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t": "=r" (d), 
                    "=r" (a):: "%rax", "%rbx", "%rcx", "%rdx");

   return ((unsigned long long)a) | (((unsigned long long)d) << 32);;
}

unsigned long long int end_rdtsc(void)
{
   unsigned long long int x;
   unsigned a, d;

   __asm__ volatile("RDTSCP\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax,%1\n\t"
                    "CPUID\n\t": "=r" (d), "=r" (a):: 
                    "%rax", "%rbx", "%rcx", "%rdx");

   return ((unsigned long long)a) | (((unsigned long long)d) << 32);;
}

int cmp_dbl(const void *x, const void *y)
{
  double xx = *(double*)x, yy = *(double*)y;
  if (xx < yy) return -1;
  if (xx > yy) return  1;
  return 0;
}

int timing_median() {
  unsigned char in[512 * NUM_TIMINGS];
  unsigned char out[512 * NUM_TIMINGS];
  unsigned char k[16];
  unsigned long long inlen;
  unsigned long long timer = 0;
  double timings[NUM_TIMINGS];

  int i,j;

  srand(0);
  inlen = 512 * NUM_TIMINGS;
  for(i =- 100; i < NUM_TIMINGS; i++){
    //Get random input
    for(j = 0; j < inlen; j++) 
      in[j] = rand() & 0xff;
    for(j = 0; j < 16; j++) 
      k[j] = rand() & 0xff;

    timer = start_rdtsc();
    ecb_enc(in, inlen, out, k);
    timer = end_rdtsc() - timer;

    if(i >= 0 && i < NUM_TIMINGS) 
      timings[i] = ((double)timer) / inlen;
  }
  //Get Median
  qsort(timings, NUM_TIMINGS, sizeof(double), cmp_dbl);
  printf("Timing median: %f cycles per byte\n", timings[NUM_TIMINGS / 2]);
}

#endif