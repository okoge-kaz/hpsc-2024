#include <cstdio>
#include <cstdlib>
#include <vector>

void parallel_prefix_sum(std::vector<int> &a) {
  int n = a.size();
  std::vector<int> b(n);

#pragma omp parallel
  for (int j = 1; j < n; j <<= 1) {
#pragma omp for
    for (int i = 0; i < n; i++)
      b[i] = a[i];
#pragma omp barrier
#pragma omp for
    for (int i = j; i < n; i++)
      a[i] += b[i - j];
#pragma omp barrier
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
#pragma omp atomic
    bucket[key[i]]++;
  }
  std::vector<int> offset(range, 0);
#pragma omp parallel for
  for (int i = 1; i < range; i++) {
    offset[i] = bucket[i - 1];
  }
  parallel_prefix_sum(offset);

#pragma omp parallel for
  for (int i = 0; i < range; i++) {
    int start_index = offset[i];
    int end_index = start_index + bucket[i];
    for (int j = start_index; j < end_index; j++) {
      key[j] = i;
    }
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}
