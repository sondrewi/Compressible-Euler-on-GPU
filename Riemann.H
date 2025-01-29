#ifndef RIEMANN_H
#define RIEMANN_H

#include <Eigen/Dense>
#include <algorithm>  // std::min
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "Grid.H"

// Header for Riemann class which will hold solution to Riemann problem
// on half-step slope limited solutions. Riemann is child class of Grid
class Riemann : public Grid {
 public:
  Riemann(Grid& left, Grid& right, bool is_x_);

  const bool is_x;

  // Reference to left and right states
  const Grid& L;
  const Grid& R;

  // References to velocity components of Riemann
  // One parallel to direction of problem and one orthogonal
  Eigen::ArrayXXd& v;
  Eigen::ArrayXXd& v_o;

  double find_p(double v_L, double v_R, double p_L, double p_R, double cs_L,
                double cs_R, double A_L, double B_L, double A_R, double B_R);

  double f_K(double p_star, double p_K, double A_K, double B_K, double c_sK);
  double f_K_prime(double f_K, double p_star, double p_K, double B_K,
                   double cs_K);
  double calc_v_star(double v_L, double v_R, double p_L, double p_R,
                     double p_star, double cs_L, double cs_R, double A_L,
                     double B_L, double A_R, double B_R);

  void solve();

  void set_state(double rho_K, double v_K, double v_K_orth, double v_star,
                 double p_star, double p_K, double cs_K, bool left, int i,
                 int j);

  void pad();
};

#endif
