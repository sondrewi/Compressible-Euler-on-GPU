#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

//Header file for DeviceKernels file in which
//CUDA kernels and device functions are defined

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "Macros.H"


__global__ void quad_init(double** sol_device,
                          const double dx, const double dy, const int x_dim,
                          const int y_dim);

__global__ void bubble_init(double** sol_device,
                          const double dx, const double dy, const int x_dim,
                          const int y_dim);

__global__ void convert(double** sol_device, int x_dim,
                        int y_dim);

__global__ void register_step(double** sol_device,
                              const double step_const, int n_cells);

__device__ double limiter(double deltai_min, double deltai_plus);

__device__ void prim_to_cons(double (&u)[4]);

__device__ void cons_to_prim(double (&u)[4]);

__device__ void calc_netflux(double (&u_L)[4], double (&u_R)[4],
                               double (&netfluxes)[4]);

__device__ void find_star(double (&u_L)[4], double (&u_R)[4],
                          double (&star_states)[2], double cs_L, double cs_R);

__device__ double f_K(double p_star, double p_K, double A_K, double B_K,
                      double c_sK);

__device__ double f_K_prime(double f_K, double p_star, double p_K, double B_K,
                            double cs_K);

__device__ void set_state(double (&u_K)[4], double (&star_states)[2], const double cs_K);

__global__ void transpose_kern(const double* a, double* aT, int w, int h);

__global__ void calc_ws_kern(double** sol_device, double* block_max_ws, double* block_max_ws_idx,
                             int x_dim, int y_dim);

#endif
