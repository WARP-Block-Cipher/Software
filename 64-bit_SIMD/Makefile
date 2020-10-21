objects = timing.o WARP_SIMD_code.o
DEFINES = -DPIPE=$(p)
CC = g++ -mavx2 -O3 -std=c++11 $(DEFINES)

A_p$(p) : WARP_SIMD_main.cpp WARP_SIMD_code.cpp timing.cpp
	$(CC) -c WARP_SIMD_code.cpp -o WARP_SIMD_code.o
	$(CC) -c timing.cpp -o timing.o
	$(CC) WARP_SIMD_main.cpp timing.o WARP_SIMD_code.o -o A_p$(p)
	make clean

clean:
	rm $(objects)

clear:
	rm A_p$(p)
