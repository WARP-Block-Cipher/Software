/*******************************************************
 * Software implementation using SIMD by WARP Team     *
 *******************************************************/
#include "WARP_SIMD_code.h"
#include "timing.h"
#include "timing_median.h"

int main()
{
    gen_test_vector();
    timing();
    timing_median();
    return 0;
}