#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Dense>
#include <algorithm>  // std::min
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "Riemann.H"
#include "Slope_limiter.H"

// Header for Solver class

class Solver {
 private:
  int nx, ny;

  void calc_lim_slopes(Grid& lim_l, Grid& lim_r, bool is_x);

  void update(double dt, double del, Grid& riemann, bool is_x);

  void hs_update(double dt, double del, Grid& u_L, Grid& u_R);

 public:
  Grid& base_grid;
  double step(double t, double end_t);
  Solver(Grid& solution);
};

#endif
