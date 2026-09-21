// Variable-head native TF3 models, sharing the established attention arithmetic.
#define FA64_MID (heads * 32)
#define FA64_KERNEL_NAME attention_fa2_q64_generic_kernel
#include "attention_fa2_q64.cu"
