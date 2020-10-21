## Software Implementations of WARP

WARP is a lightweight 128-bit block cipher with a 128-bit key. It aims at small-footprint circuit in the field of 128-bit block ciphers, possibly for a unified encryption and decryption functionality. The overall structure of WARP is a variant of 32-nibble Type-2 Generalized Feistel Network (GFN), with a permutation over nibbles designed to optimize the security and efficiency. For more details on the design, security, and performance of WARP, please refer to \[1\].

This repository provides the software implementations (reference codes and software implementations) for WARP.

To compile the SIMD implementation, please specify the degree of parallelization as follows.

- Processing one block at a time `make p=1`
- Processing two blocks at a time `make p=2`
- Processing four blocks at a time `make p=4`
- Processing eight blocks at a time `make p=8`

## References

[1] Subhadeep Banik, Zhenzhen Bao, Takanori Isobe, Hiroyasu Kubo, Fukang Liu, Kazuhiko Minematsu, Kosei Sakamoto, Nao Shibata, and Maki Shigeri, "WARP : Revisiting GFN for Lightweight 128-bit Block Cipher", SAC 2020
