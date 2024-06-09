#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <omp.h>

using namespace std;
typedef vector<vector<double> > matrix;

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny, vector<double>(nx));
  matrix v(ny, vector<double>(nx));
  matrix p(ny, vector<double>(nx));
  matrix b(ny, vector<double>(nx));
  matrix un(ny, vector<double>(nx));
  matrix vn(ny, vector<double>(nx));
  matrix pn(ny, vector<double>(nx));

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  for (int n = 0; n < nt; n++) {
    for (int j = 1; j < ny - 1; j++) {
      #pragma omp parallel for
      for (int i = 1; i < nx - 1; i++) {
        // Compute b[j][i]
        b[j][i] = rho * (1 / dt * ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
                         ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) *
                             ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) -
                         2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
                              (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) -
                         ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)) *
                             ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)));
      }
    }
    for (int it = 0; it < nit; it++) {
      for (int j = 0; j < ny; j++)
      #pragma omp parallel for
        for (int i = 0; i < nx; i++)
          pn[j][i] = p[j][i];
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          // Compute p[j][i]
          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) + dx * dx * (pn[j + 1][i] + pn[j - 1][i]) - b[j][i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
        }
      }
      for (int j = 0; j < ny; j++) {
        // Compute p[j][0] and p[j][nx-1]
        p[j][0] = p[j][1];            // p[:, 0] = p[:, 1]
        p[j][nx - 1] = p[j][nx - 2];  // p[:, -1] = p[:, -2]
      }
      for (int i = 0; i < nx; i++) {
        // Compute p[0][i] and p[ny-1][i]
        p[0][i] = p[1][i];  // p[0, :] = p[1, :]
        p[ny - 1][i] = 0;   // p[-1, :] = 0
      }
    }

    // copy
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }

    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        // Compute u[j][i] and v[j][i]
        // u[j][i]
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * (dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) + dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]));

        // v[j][i]
        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) + nu * (dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) + dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]));
      }
    }
    for (int j = 0; j < ny; j++) {
      // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      u[j][0] = 0;       // u[j, 0] = 0
      u[j][nx - 1] = 0;  // u[j, -1] = 0
      v[j][0] = 0;       // v[j, 0] = 0
      v[j][nx - 1] = 0;  // v[j, -1] = 0
    }
    for (int i = 0; i < nx; i++) {
      // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      u[0][i] = 0;       // u[0, i] = 0
      u[ny - 1][i] = 1;  // u[-1, i] = 1
      v[0][i] = 0;       // v[0, i] = 0
      v[ny - 1][i] = 0;  // v[-1, i] = 0
    }
    if (n % 10 == 0) {
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
