// Diagnostic candidate: q64 register layout with q128's softmax reduction
// order. Compiled as a separate entry so neither established kernel changes.
#define FA64_SERIAL_SUM 1
#define FA64_KERNEL_NAME attention_fa2_q64_serial_kernel
#include "attention_fa2_q64.cu"
