// Variable-head native TF3 models, sharing the established attention arithmetic.
#define FA2_MID (heads * 32)
#define FA2_KERNEL_NAME attention_fa2_generic_kernel
#include "attention_fa2.cu"
