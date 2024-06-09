#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace std;
typedef vector<vector<double> > matrix;

void initialize(matrix &u, matrix &v, matrix &p, matrix &b, int local_ny, int nx);
void updateBoundaries_1(matrix &p, int local_ny, int nx, int rank, int size);
void updateBoundaries_2(matrix &u, matrix &v, int local_ny, int nx, int rank, int size);
void exchangeHalo_1(matrix &p, int local_ny, int nx, int rank, int size, MPI_Comm comm);
void exchangeHalo_2(matrix &u, matrix &v, int local_ny, int nx, int rank, int size, MPI_Comm comm);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int nx = 41;
  int ny = 41;
  int local_ny = ny / size;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(local_ny + 2, vector<double>(nx));
  matrix v(local_ny + 2, vector<double>(nx));
  matrix p(local_ny + 2, vector<double>(nx));
  matrix b(local_ny + 2, vector<double>(nx));
  matrix un(local_ny + 2, vector<double>(nx));
  matrix vn(local_ny + 2, vector<double>(nx));
  matrix pn(local_ny + 2, vector<double>(nx));

  initialize(u, v, p, b, local_ny, nx);

  ofstream ufile, vfile, pfile;
  if (rank == 0) {
    ufile.open("u.dat");
    vfile.open("v.dat");
    pfile.open("p.dat");
  }

  for (int n = 0; n < nt; n++) {
    for (int j = 1; j <= local_ny; j++) {
      for (int i = 1; i < nx - 1; i++) {
        if (rank == size - 1 && j == local_ny) {
          continue;
        }

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
      for (int j = 0; j < local_ny + 2; j++)
        for (int i = 0; i < nx; i++)
          pn[j][i] = p[j][i];

      for (int j = 1; j <= local_ny; j++) {
        for (int i = 1; i < nx - 1; i++) {
          if (rank == size - 1 && j == local_ny) {
            continue;
          }

          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) + dx * dx * (pn[j + 1][i] + pn[j - 1][i]) - b[j][i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
        }
      }
      updateBoundaries_1(p, local_ny, nx, rank, size);
      exchangeHalo_1(p, local_ny, nx, rank, size, MPI_COMM_WORLD);
    }

    // copy
    for (int j = 0; j < local_ny + 2; j++) {
      for (int i = 0; i < nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }

    for (int j = 1; j <= local_ny; j++) {
      for (int i = 1; i < nx - 1; i++) {
        if (rank == size - 1 && j == local_ny) {
          continue;
        }

        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) - dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) + nu * (dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) + dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]));
        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) - dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) + nu * (dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) + dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]));
      }
    }
    updateBoundaries_2(u, v, local_ny, nx, rank, size);
    exchangeHalo_2(u, v, local_ny, nx, rank, size, MPI_COMM_WORLD);

    if (n % 10 == 0) {
      vector<double> u_global, v_global, p_global;
      if (rank == 0) {
        u_global.resize(ny * nx);
        v_global.resize(ny * nx);
        p_global.resize(ny * nx);
      }

      MPI_Gather(&u[1][0], local_ny * nx, MPI_DOUBLE, &u_global[0], local_ny * nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gather(&v[1][0], local_ny * nx, MPI_DOUBLE, &v_global[0], local_ny * nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Gather(&p[1][0], local_ny * nx, MPI_DOUBLE, &p_global[0], local_ny * nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            ufile << u_global[j * nx + i] << " ";
            vfile << v_global[j * nx + i] << " ";
            pfile << p_global[j * nx + i] << " ";
          }
        }
        ufile << "\n";
        vfile << "\n";
        pfile << "\n";
      }
    }
  }

  if (rank == 0) {
    ufile.close();
    vfile.close();
    pfile.close();
  }

  MPI_Finalize();
  return 0;
}

void initialize(matrix &u, matrix &v, matrix &p, matrix &b, int local_ny, int nx) {
  for (int j = 0; j < local_ny + 2; j++) {
    for (int i = 0; i < nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
}

void updateBoundaries_1(matrix &p, int local_ny, int nx, int rank, int size) {
  for (int j = 0; j <= local_ny; j++) {
    if (rank == size - 1 && j == local_ny) {
      continue;
    }

    p[j][0] = p[j][1];
    p[j][nx - 1] = p[j][nx - 2];
  }
  for (int i = 0; i < nx; i++) {
    if (rank == 0) {
      p[0][i] = p[1][i];
    }
    if (rank == size - 1) {
      // local_ny + 1: overlap area
      p[local_ny][i] = 0;
    }
  }
}

void updateBoundaries_2(matrix &u, matrix &v, int local_ny, int nx, int rank, int size) {
  for (int j = 0; j <= local_ny; j++) {
    if (rank == size - 1 && j == local_ny) {
      continue;
    }

    u[j][0] = 0;
    u[j][nx - 1] = 0;
    v[j][0] = 0;
    v[j][nx - 1] = 0;
  }
  for (int i = 0; i < nx; i++) {
    if (rank == 0) {
      u[0][i] = 0;
      v[0][i] = 0;
    }
    if (rank == size - 1) {
      // local_ny + 1: overlap area
      u[local_ny][i] = 1;
      v[local_ny][i] = 0;
    }
  }
}

void exchangeHalo_1(matrix &p, int local_ny, int nx, int rank, int size, MPI_Comm comm) {
  MPI_Request reqs[4];
  MPI_Status stats[4];

  if (rank != 0) {
    MPI_Irecv(&p[0][0], nx, MPI_DOUBLE, rank - 1, 1, comm, &reqs[0]);
  }

  if (rank != size - 1) {
    MPI_Irecv(&p[local_ny + 1][0], nx, MPI_DOUBLE, rank + 1, 0, comm, &reqs[2]);
  }

  if (rank != 0) {
    MPI_Isend(&p[1][0], nx, MPI_DOUBLE, rank - 1, 0, comm, &reqs[1]);
  }

  if (rank != size - 1) {
    MPI_Isend(&p[local_ny][0], nx, MPI_DOUBLE, rank + 1, 1, comm, &reqs[3]);
  }

  if (rank != 0) {
    MPI_Waitall(2, reqs, stats);
  }

  if (rank != size - 1) {
    MPI_Waitall(2, reqs + 2, stats + 2);
  }
}

void exchangeHalo_2(matrix &u, matrix &v, int local_ny, int nx, int rank, int size, MPI_Comm comm) {
  MPI_Request reqs[8];
  MPI_Status stats[8];

  if (rank != 0) {
    MPI_Irecv(&u[0][0], nx, MPI_DOUBLE, rank - 1, 1, comm, &reqs[0]);
    MPI_Irecv(&v[0][0], nx, MPI_DOUBLE, rank - 1, 3, comm, &reqs[1]);
  }

  if (rank != size - 1) {
    MPI_Irecv(&u[local_ny + 1][0], nx, MPI_DOUBLE, rank + 1, 0, comm, &reqs[4]);
    MPI_Irecv(&v[local_ny + 1][0], nx, MPI_DOUBLE, rank + 1, 2, comm, &reqs[5]);
  }

  if (rank != 0) {
    MPI_Isend(&u[1][0], nx, MPI_DOUBLE, rank - 1, 0, comm, &reqs[2]);
    MPI_Isend(&v[1][0], nx, MPI_DOUBLE, rank - 1, 2, comm, &reqs[3]);
  }

  if (rank != size - 1) {
    MPI_Isend(&u[local_ny][0], nx, MPI_DOUBLE, rank + 1, 1, comm, &reqs[6]);
    MPI_Isend(&v[local_ny][0], nx, MPI_DOUBLE, rank + 1, 3, comm, &reqs[7]);
  }

  if (rank != 0) {
    MPI_Waitall(4, reqs, stats);
  }

  if (rank != size - 1) {
    MPI_Waitall(4, reqs + 4, stats + 4);
  }
}
