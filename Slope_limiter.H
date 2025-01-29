#ifndef SL_H
#define SL_H

#include <Eigen/Dense>
#include <algorithm>  // std::min
#include <cmath>
#include <iostream>
#include <vector>

#include "Macros.H"

// Header file for Slope_limiter class
class Slope_limiter {
 public:
  bool is_x;
  int nx, ny;

  // Arrays for slop limiter variables that will have to be
  // available separately
  Eigen::ArrayXXd deltai;
  Eigen::ArrayXXd r;
  Eigen::ArrayXXd Xi_R;
  Eigen::ArrayXXd Xi;

  Slope_limiter(const int nx_, const int ny_, bool is_x_);

  void calc_deltai(const Eigen::ArrayXXd& prev);
  void calc_limiter();
};

#endif
