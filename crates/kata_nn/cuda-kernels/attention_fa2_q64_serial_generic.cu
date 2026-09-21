// Variable-head q64 with serial softmax reduction.
#define FA64_MID (heads * 32)
#define FA64_SERIAL_SUM 1
#define FA64_KERNEL_NAME attention_fa2_q64_serial_generic_kernel
#include "attention_fa2_q64.cu"
