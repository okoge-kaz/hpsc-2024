#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void count_keys(int *keys, int *buckets, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    atomicAdd(&buckets[keys[index]], 1);
  }
}

__global__ void fill_keys(int *keys, int *buckets, int *sum, int range) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < range) {
    int start = sum[index];
    int key = index;
    for (int i = 0; i < buckets[index]; i++) {
      keys[start+i] = key;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *keys, *buckets, *sum;
  int *d_keys, *d_buckets, *d_sum;

  keys = (int *)malloc(n * sizeof(int));
  buckets = (int *)calloc(range, sizeof(int));
  sum = (int *)malloc(range * sizeof(int));

  // d_keys, d_buckets are accessible from only GPU
  // cudaMalloc: allocate memory on GPU
  // cudaMallocManaged: allocate memory on unified memory
  cudaMalloc(&d_keys, n * sizeof(int));
  cudaMalloc(&d_buckets, range * sizeof(int));
  cudaMalloc(&d_sum, range * sizeof(int));
  cudaMemset(d_sum, 0, range * sizeof(int));     // set 0 to all elements

  for (int i = 0; i < n; i++) {
    keys[i] = rand() % range;
    printf("%d ", keys[i]);
  }
  printf("\n");

  // cudaMemcpy: copy memory from CPU to GPU
  cudaMemcpy(d_keys, keys, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_buckets, buckets, range * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_buckets, 0, range * sizeof(int)); // set 0 to all elements

  int blockSize = 1024;
  int numBlocks = (n + blockSize - 1) / blockSize;

  count_keys<<<numBlocks, blockSize>>>(d_keys, d_buckets, n);
  cudaDeviceSynchronize();

  // cudaMemcpy: copy memory from GPU to CPU
  cudaMemcpy(buckets, d_buckets, range * sizeof(int), cudaMemcpyDeviceToHost);

  sum[0] = 0;
  for (int i = 1; i < range; i++) {
    sum[i] = sum[i-1] + buckets[i-1];
  }
  cudaMemcpy(d_sum, sum, range * sizeof(int), cudaMemcpyHostToDevice);

  fill_keys<<<numBlocks, blockSize>>>(d_keys, d_buckets, d_sum, range);
  cudaDeviceSynchronize();

  // cudaMemcpy: copy memory from GPU to CPU
  cudaMemcpy(keys, d_keys, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    printf("%d ", keys[i]);
  }
  printf("\n");

  free(keys);
  free(buckets);
  free(sum);
  cudaFree(d_keys);
  cudaFree(d_buckets);
  cudaFree(d_sum);

  return 0;
}
