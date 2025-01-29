#include "CUDA_Grid.cuh"
#include "Solver.H"
using CUDA_Var = CUDAGrid::Var;
using Var = Grid::Var;

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(void) {
  std::string sim_type;
  bool bubble = false;
  bool isValidInput = false;

  // Ask user to choose between quadrant and bubble simulation
  while (!isValidInput) {
    std::cout << "Simulate Quadrant (q) or Bubble (b) problem: ";
    std::cin >> sim_type;

    if (sim_type == "b") {
      bubble = true;
      isValidInput = true;
    } else if (sim_type == "q") {
      bubble = false;
      isValidInput = true;
    } else {
      std::cout << "Invalid input. Please enter 'b' for Bubble or 'q' for "
                   "Quadrant.\n";
    }
  }

  // Declare grid on which to solve
  // CUDAGrid CUDA_grid;
  // Grid grid;

  double t_end;
  int x_cells, y_cells;
  double x_min, x_max, y_min, y_max;

  // Conditionally set initial conditions for quadrant
  if (!bubble) {
    t_end = 0.3;
    x_cells = 400;
    y_cells = 400;
    x_min = 0.0;
    x_max = 1.0;
    y_min = 0.0;
    y_max = 1.0;
  }

  // Conditionally set initial conditions for bubble
  else {
    t_end = 0.000482;
    x_cells = 500;
    y_cells = 197;
    x_min = 0.0;
    x_max = 0.225;
    y_min = -0.0445;
    y_max = 0.0445;
  }

  // Set initial time and step_count
  double t_CPU = 0.0;
  double step_count_CPU = 0;

  Grid grid = Grid(x_min, x_max, y_min, y_max, x_cells, y_cells);
  grid.set_init(bubble);

  // Initialize solver object for CPU code
  Solver solver = Solver(grid);

  long long total_time_CPU = 0;

  // solve on CPU
  while (t_CPU < t_end) {
    step_count_CPU += 1;
    auto start_CPU = std::chrono::high_resolution_clock::now();
    double dt_CPU = solver.step(t_CPU, t_end);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    total_time_CPU += std::chrono::duration_cast<std::chrono::microseconds>(
                          end_CPU - start_CPU)
                          .count();
    t_CPU += dt_CPU;
    std::cout << t_CPU << std::endl;
  }

  std::cout << "CPU Average Iteration Time: "
            << static_cast<double>(total_time_CPU) / step_count_CPU
            << std::endl;
  std::cout << "Step count on CPU: " << step_count_CPU << std::endl;

  double t_CUDA = 0.0;

  // Initialise host Grid for CUDA
  CUDAGrid CUDA_grid = CUDAGrid(x_min, x_max, y_min, y_max, x_cells, y_cells);
  CUDA_grid.set_init(bubble);

  long long total_time_GPU = 0;
  double step_count = 0;

  // Solve on GPU
  while (t_CUDA < t_end) {
    auto start_GPU = std::chrono::high_resolution_clock::now();
    double dt_CUDA = CUDA_grid.step(t_CUDA, t_end);
    auto end_GPU = std::chrono::high_resolution_clock::now();
    total_time_GPU += std::chrono::duration_cast<std::chrono::microseconds>(
                          end_GPU - start_GPU)
                          .count();
    t_CUDA += dt_CUDA;
    step_count += 1;
    // t_steps << dt_CPU << "  " << dt_CUDA << std::endl;
  }

  std::cout << "GPU Average Iteration Time: "
            << static_cast<double>(total_time_GPU) / step_count << std::endl;
  std::cout << "Step count on GPU: " << step_count << std::endl;

  CUDA_grid.retrieve_sol();

  double dx = grid.dx;
  double dy = grid.dy;

  // Output file
  std::ofstream sols("Solutions.dat");

  // Write to output
  const double* CUDA_rho = CUDA_grid.get(CUDA_Var::rho);
  const double* CUDA_vx = CUDA_grid.get(CUDA_Var::vx);
  const double* CUDA_vy = CUDA_grid.get(CUDA_Var::vy);
  const double* CUDA_p = CUDA_grid.get(CUDA_Var::p);

  const Eigen::ArrayXXd rho = grid.get(Var::rho).block(1, 1, grid.nx, grid.ny);
  const Eigen::ArrayXXd vx = grid.get(Var::vx).block(1, 1, grid.nx, grid.ny);
  const Eigen::ArrayXXd vy = grid.get(Var::vy).block(1, 1, grid.nx, grid.ny);
  const Eigen::ArrayXXd p = grid.get(Var::p).block(1, 1, grid.nx, grid.ny);

  sols << std::setprecision(14);

  for (int i = 0; i < x_cells; i++) {
    for (int j = 0; j < y_cells; j++) {
      double x = grid.x_min + (i + 0.5) * dx;
      double y = grid.y_min + (j + 0.5) * dy;
      int idx = j * x_cells + i;

      sols << x << "  " << y << "  " << rho(i, j) << "  " << CUDA_rho[idx]
           << "  " << vx(i, j) << "  " << CUDA_vx[idx] << "  " << vy(i, j)
           << "  " << CUDA_vy[idx] << "  " << p(i, j) << "  " << CUDA_p[idx]
           << std::endl;
    }

    sols << std::endl;
  }
}
