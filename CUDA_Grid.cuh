#ifndef CUDAGRID_H
#define CUDAGRID_H

#include <algorithm>  // std::min
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <vector>

#include "DeviceKernels.cuh"

#define CUDA_CHECK                                                  \
  {                                                                 \
    cudaDeviceSynchronize();                                        \
    cudaError_t err = cudaGetLastError();                           \
    if (err) {                                                      \
      std::cout << "Error: " << cudaGetErrorString(err) << " line " \
                << __LINE__ << " in CUDA_Grid.cu" << std::endl; \
      exit(1);                                                      \
    }                                                               \
  }

class CUDAGrid {
 protected:
  double* host_p;
  double* host_rho;
  double* host_vx;
  double* host_vy;

  double* sol_d_addr_on_h[4];
  double** sol_device;

  double dt;

 public:
  int x_dim, y_dim;

  double x_min, x_max, y_min, y_max;

  double dx, dy;

  enum class Var { rho, xmom, ymom, E, vx, vy, p };

  CUDAGrid(double xmin, double xmax, double ymin, double ymax, int nx_,
            int ny_);

  ~CUDAGrid();

  CUDAGrid();

  void retrieve_sol();

  void transpose_sol(bool is_x);

  void set_init(bool bubble);

  double calc_t_step(double t, double max_t);

  void copy_back();

  double step(double t, double max_t);

  const double* get(Var type) const;
};

#endif
