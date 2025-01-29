#include "CUDA_Grid.cuh"

CUDAGrid::CUDAGrid() {}

// Constructor for generic solution object. Solution will have GPU addresses
// to array stored on the CPU, but no array data actually stored on CPU
// until copy_back is called
CUDAGrid::CUDAGrid(double xmin, double xmax, double ymin, double ymax,
                     int nx_, int ny_) {
  //Set dimensions
  x_dim = nx_;
  y_dim = ny_;

  // Set grid domain
  x_min = xmin;
  x_max = xmax;
  y_max = ymax;
  y_min = ymin;

  //Set grid delta in x- and y- directions
  dx = (xmax - xmin) / x_dim;
  dy = (y_max - y_min) / y_dim;

  //Make pointers to host arrays
  host_rho = new double[x_dim * y_dim];
  host_vx = new double[x_dim * y_dim];
  host_vy = new double[x_dim * y_dim];
  host_p = new double[x_dim * y_dim];

  //Allocate memory for 4 variables
  //on device and store addresses on host
  for (int k = 0; k < 4; k++) {
    cudaMalloc((void**)&sol_d_addr_on_h[k], sizeof(double) * x_dim * y_dim);
    CUDA_CHECK;
  }

  //Store array, with the same addresses, on device
  cudaMalloc(&sol_device, 4 * sizeof(double*));
  cudaMemcpy(sol_device, sol_d_addr_on_h, 4 * sizeof(double*),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
}

//Retrieve primitive variables from device
//Conservative variables first need to be converted
void CUDAGrid::retrieve_sol() {
  dim3 blocks(((x_dim * y_dim) + 1023) / 1024);
  dim3 tile(1024);

  convert<<<blocks, tile>>>(sol_device, x_dim, y_dim);
  CUDA_CHECK;

  copy_back();
}

//Transpose arrays
void CUDAGrid::transpose_sol(bool is_x) {
  //Placeholder for transpose
  double* t;
  cudaMalloc((void**)&t, sizeof(double) * (x_dim * y_dim));
  CUDA_CHECK;

  /*//Blocks in x- and y-directions of original matrix
  int x_blocks = (x_dim + 31)/32;
  int y_blocks = (y_dim + 31)/32;

  //Blocks in x- and y-direction of matrix to transpose
  int blocks_i = is_x ? x_blocks : y_blocks;
  int blocks_j = is_x ? y_blocks : x_blocks;

  //Dimensions of matrix to transpose
  int i_dim = is_x ? x_dim : y_dim;
  int j_dim = is_x ? y_dim : x_dim;

  //Initiate blocks and tile for kernel
  dim3 blocks(blocks_i, blocks_j);
  dim3 tile(32,32);*/

  //Dimensions of matrix to transpose
  int i_dim = is_x ? x_dim : y_dim;
  int j_dim = is_x ? y_dim : x_dim;

  dim3 blocks((x_dim*y_dim+1023)/1024);
  dim3 tile(1024);

  //Transpose all 4 arrays
  for (int k = 0; k < 4; k++) {
    transpose_kern<<<blocks, tile>>>(sol_d_addr_on_h[k], t, i_dim, j_dim);
    CUDA_CHECK;
    std::swap(sol_d_addr_on_h[k], t);
  }

  //Free placeholder for the transpose
  cudaFree(t);
  CUDA_CHECK;

  //Modify sol_device array on device as pointers have changed
  cudaMemcpy(sol_device, sol_d_addr_on_h, 4 * sizeof(double*),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
}

//Set initial values for bubble or quadrant
void CUDAGrid::set_init(bool bubble) {
  dim3 blocks(((x_dim * y_dim) + 1023) / 1024);
  dim3 tile(1024);

  if (!bubble) {
    quad_init<<<blocks, tile>>>(sol_device, dx, dy, x_dim, y_dim);
  }

  else {
    bubble_init<<<blocks, tile>>>(sol_device, dx, dy, x_dim, y_dim);
  }

  CUDA_CHECK;
}

//Copy arrays from GPU back to host. Assumed that they
//contain primitive variables on GPU
void CUDAGrid::copy_back() {
  cudaMemcpy(host_rho, sol_d_addr_on_h[0], (x_dim * y_dim) * sizeof(double),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(host_vx, sol_d_addr_on_h[1], (x_dim * y_dim) * sizeof(double),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(host_vy, sol_d_addr_on_h[2], (x_dim * y_dim) * sizeof(double),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(host_p, sol_d_addr_on_h[3], (x_dim * y_dim) * sizeof(double),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
}

//Calculate time-step
double CUDAGrid::calc_t_step(double t, double max_t) {

  //Blocks contain up to 1024 threads
  int block_nr = ((x_dim * y_dim) + 1023) / 1024;

  //Array on host of max wavespeeds found from each block
  double* block_max_ws_host = new double[block_nr];

  //Array on host with index of max cell with max wavespeed
  double* block_max_ws_host_idx = new double[block_nr];

  //Similar arrays on device
  double* block_max_ws_device;
  cudaMalloc(&block_max_ws_device, sizeof(double) * block_nr);
  CUDA_CHECK;

  double* block_max_ws_device_idx;
  cudaMalloc(&block_max_ws_device_idx, sizeof(int) * block_nr);
  CUDA_CHECK;

  dim3 blocks(block_nr);
  dim3 tile(1024);

  //Call kernel to calculate max wavespeed across each cell-block
  calc_ws_kern<<<blocks, tile>>>(sol_device, block_max_ws_device,
                                 block_max_ws_device_idx, x_dim, y_dim);
  CUDA_CHECK;

  //Copy max wavespeeds from each block and associated cell indices back to host
  cudaMemcpy(block_max_ws_host, block_max_ws_device, sizeof(double) * block_nr,
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaMemcpy(block_max_ws_host_idx, block_max_ws_device_idx,
             sizeof(int) * block_nr, cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  //Iterate over array and find maximal wavespeed across all blocks
  double max_speed = block_max_ws_host[0];
  int max_idx = block_max_ws_host_idx[0];

  for (int k = 1; k < block_nr; k++) {
    max_idx =
        (block_max_ws_host[k] > max_speed) ? block_max_ws_host_idx[k] : max_idx;
    max_speed = max(max_speed, block_max_ws_host[k]);
  }

  //Free cuda arrays
  cudaFree(block_max_ws_device);
  cudaFree(block_max_ws_device_idx);
  CUDA_CHECK;

  //Free host arrays
  delete[] block_max_ws_host;
  delete[] block_max_ws_host_idx;

  //Calculate timestep with CFL = 1
  double t_step = min(dy, dx) / max_speed;

  //Check for going beyond final time
  if (t_step + t <= max_t) {
    return t_step;
  }

  else {
    return max_t - t;
  }
}

//Update solution one time-step
double CUDAGrid::step(double t, double max_t) {

  //Block number in x-update equal to nr of rows in y-dimension
  //One row in grid per block
  int rows_per_block = 1;
  int block_nr = (y_dim + rows_per_block -1) /rows_per_block;

  dim3 blocks_x(block_nr);
  dim3 tile_x(x_dim, rows_per_block);

  //Calculate shared memory size
  //Pad shared memory with two extra doubles either side of a row for difference calculations
  //For each cell, we will need to hold 5 values in shared memory. 4 for the old solution and
  //1 as a flexible placeholder
  int sh_width = x_dim + 2;
  size_t shared_mem_size = 5 * sh_width * rows_per_block * sizeof(double);

  //Calculate time-step
  double dt = calc_t_step(t, max_t);
  CUDA_CHECK;

  //Calculate step constant delta t / delta x
  double step_const = dt / dx;

  //Perform update in x-direction
  register_step<<<blocks_x, tile_x, shared_mem_size>>>(sol_device, step_const, x_dim * y_dim);
  CUDA_CHECK;

  //transpose solutions
  transpose_sol(true);

  //Swap pointers to x-momentum and y-momentum as in a y-update
  //y-momentum takes on role of x-momentum in an x-update and
  //vice-versa
  std::swap(sol_d_addr_on_h[1], sol_d_addr_on_h[2]);

  //Copy this change back to device
  cudaMemcpy(sol_device, sol_d_addr_on_h, 4 * sizeof(double*),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  //Calculate new step constant
  step_const = dt / dy;

  //Set new block and tile dimensions
  rows_per_block = 1;
  block_nr = (x_dim + rows_per_block -1) /rows_per_block;

  dim3 blocks_y(block_nr);
  dim3 tile_y(y_dim, rows_per_block);


  //Calculate size of shared memory
  sh_width = y_dim + 2;
  shared_mem_size = 5 * sh_width * rows_per_block * sizeof(double);

  //Make step in y-direction
  register_step<<<blocks_y, tile_y, shared_mem_size>>>(sol_device, step_const, x_dim * y_dim);
  CUDA_CHECK;

  //Transpose solution and swap pointers again for x- and y-momentum
  transpose_sol(false);
  std::swap(sol_d_addr_on_h[1], sol_d_addr_on_h[2]);
  cudaMemcpy(sol_device, sol_d_addr_on_h, 4 * sizeof(double*),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  //Return timestep
  return dt;
}

//Free memory on device and on host
CUDAGrid::~CUDAGrid() {
  cudaFree(sol_d_addr_on_h[0]);
  cudaFree(sol_d_addr_on_h[1]);
  cudaFree(sol_d_addr_on_h[2]);
  cudaFree(sol_d_addr_on_h[3]);
  cudaFree(sol_device);

  delete[] host_p;
  delete[] host_rho;
  delete[] host_vx;
  delete[] host_vy;
}

//Get pointer to host array of solutions
const double* CUDAGrid::get(Var type) const {
  switch (type) {
    case Var::rho:
      return host_rho;
    case Var::vx:
      return host_vx;
    case Var::vy:
      return host_vy;
    case Var::p:
      return host_p;
    default:
      // Handle the error or throw an exception
      throw std::invalid_argument("Invalid ArrayType");
  }
}
