#include <immintrin.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];

  for (int i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  for (int i = 0; i < N; i++) {
    __m256 x_i = _mm256_set1_ps(x[i]);
    __m256 y_i = _mm256_set1_ps(y[i]);

    __m256 x_j = _mm256_load_ps(x);
    __m256 y_j = _mm256_load_ps(y);
    __m256 m_j = _mm256_load_ps(m);

    __m256 rx = _mm256_sub_ps(x_i, x_j);
    __m256 ry = _mm256_sub_ps(y_i, y_j);
    __m256 r_sq = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));
    __m256 r = _mm256_sqrt_ps(r_sq);
    __m256 r_cube = _mm256_mul_ps(r_sq, r);

    __m256 mask = _mm256_cmp_ps(x_i, x_j, _CMP_NEQ_OQ);

    __m256 force_x = _mm256_div_ps(_mm256_mul_ps(rx, m_j), r_cube);
    __m256 force_y = _mm256_div_ps(_mm256_mul_ps(ry, m_j), r_cube);

    force_x = _mm256_and_ps(force_x, mask);
    force_y = _mm256_and_ps(force_y, mask);

    // accumulate forces
    for (int j = 0; j < 8; j++) {
      fx[i] = fx[i] - force_x[j];
      fy[i] = fy[i] - force_y[j];
    }
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
