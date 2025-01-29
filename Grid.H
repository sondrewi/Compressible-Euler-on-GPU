#ifndef GRID_H
#define GRID_H

#include <Eigen/Dense>
#include <algorithm>  // std::min
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Macros.H"

class Grid {
  // Eigen arrays for the various fields that appear on domain
  // through the course of MUSCL-Hancock algorithm
 protected:
  Eigen::ArrayXXd p;
  Eigen::ArrayXXd rho;
  Eigen::ArrayXXd vx;
  Eigen::ArrayXXd vy;
  Eigen::ArrayXXd xmom;
  Eigen::ArrayXXd ymom;
  Eigen::ArrayXXd E;
  Eigen::ArrayXXd cs;

  Eigen::ArrayXXd f_rho;
  Eigen::ArrayXXd f_xmom;
  Eigen::ArrayXXd f_ymom;
  Eigen::ArrayXXd f_E;

 public:
  // Boolean to indicate if grid is base grid on which solutions
  // between time-steps are stored
  bool base;

  double x_min, x_max, y_min, y_max;
  int nx, ny;

  double dx, dy;

  // Enumeration of different variables
  enum class Var { rho, xmom, ymom, E, vx, vy, p, cs };

  void set_init(bool bubble);
  void prim_to_cons();
  void cons_to_prim();
  Grid();
  Grid(int nx_, int ny_);

  Grid(const double xmin, const double xmax, const double ymin,
       const double ymax, const int x_cells, const int y_cells);

  void set_ghost();
  double calc_tstep(double t, double end_t);

  void calc_cs();

  void calc_flux(bool is_x);

  const Eigen::ArrayXXd& get(Var type) const;
  const Eigen::ArrayXXd& get_flux(Var type) const;

  Eigen::ArrayXXd& set(Var type);
};

#endif
