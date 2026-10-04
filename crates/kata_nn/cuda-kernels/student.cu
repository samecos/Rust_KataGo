// FP32 operators for the 19x19 compact/dense research StudentNet.
// NCHW tensors and PyTorch's [out,in,ky,kx] weight order are preserved.
// Convolutions use these im2col values with FP32 cuBLAS SGEMM.
extern "C" __global__ void student_im2col(
    const float* input, float* columns, int channels, int batch) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int area = 361, kernel_channels = channels * 9;
  const int count = batch * kernel_channels * area;
  if (index >= count) return;
  const int point = index % area;
  const int k = (index / area) % kernel_channels;
  const int b = index / (area * kernel_channels);
  const int y = point / 19 + (k % 9) / 3 - 1;
  const int x = point % 19 + k % 3 - 1;
  columns[index] = (y >= 0 && y < 19 && x >= 0 && x < 19)
      ? input[(b * channels + k / 9) * area + y * 19 + x] : 0.0f;
}

extern "C" __global__ void student_conv_bias(
    float* output, const float* bias, int channels, int count, int relu) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float v = output[index] + bias[(index / 361) % channels];
  // Comparison keeps NaNs visible to the host's finite-output check, unlike
  // fmaxf which would replace NaN with its finite second operand.
  output[index] = relu && v < 0.0f ? 0.0f : v;
}

extern "C" __global__ void student_global_relu(
    float* spatial, const float* global, int channels, int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int channel = index / 361;
  const float v = spatial[index] + global[channel];
  spatial[index] = v < 0.0f ? 0.0f : v;
}

extern "C" __global__ void student_residual_relu(
    float* output, const float* residual, int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float v = output[index] + residual[index];
    output[index] = v < 0.0f ? 0.0f : v;
  }
}

extern "C" __global__ void student_pool_concat(
    const float* spatial, const float* global, float* combined,
    int channels, int batch) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int combined_channels = channels + 19;
  if (index >= batch * combined_channels) return;
  const int b = index / combined_channels;
  const int c = index % combined_channels;
  if (c >= channels) {
    combined[index] = global[b * 19 + c - channels];
  } else {
    float sum = 0.0f;
    for (int p = 0; p < 361; ++p) sum += spatial[(b * channels + c) * 361 + p];
    combined[index] = sum / 361.0f;
  }
}

extern "C" __global__ void student_linear(
    const float* input, const float* weight, const float* bias,
    float* output, int input_channels, int output_channels, int batch, int relu) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch * output_channels) return;
  const int b = index / output_channels;
  const int o = index % output_channels;
  float sum = 0.0f;
  for (int c = 0; c < input_channels; ++c)
    sum = fmaf(input[b * input_channels + c], weight[o * input_channels + c], sum);
  sum += bias[o];
  output[index] = relu && sum < 0.0f ? 0.0f : sum;
}

extern "C" __global__ void student_pack_policy(
    const float* board, const float* pass, float* policy, int batch) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= batch * 362) return;
  const int b = index / 362, p = index % 362;
  policy[index] = p < 361 ? board[b * 361 + p] : pass[b];
}
