#include "Riemann.H"
using Var = Grid::Var;

// Constructor for Riemann class. If solving in x-direction,
// Riemann must have dimension (nx+1, ny) and vice versa. Riemann
// grid holds solution at cll interfaces of base grid
Riemann::Riemann(Grid& left_lim, Grid& right_lim, bool is_x_)
    : Grid(left_lim.nx + is_x_, left_lim.ny + !is_x_),
      is_x(is_x_),
      L(left_lim),
      R(right_lim),
      v((is_x_) ? vx : vy),
      v_o((is_x_) ? vy : vx) {
  nx = left_lim.nx;
  ny = left_lim.ny;
}

// Solve Riemann problem
void Riemann::solve() {
  // Variables for left and right states
  double rho_L, rho_R, v_L, v_R, v_L_o, v_R_o, p_L, p_R;

  // References to left- and right-state arrays from
  // half-step limited solutions. Left state will be right
  // limited value of cell i, right state will be left limited
  // value of cell (i+1)
  const Eigen::ArrayXXd& rho_L_arr = R.get(Var::rho);
  const Eigen::ArrayXXd& rho_R_arr = L.get(Var::rho);

  const Eigen::ArrayXXd& v_L_arr = (is_x) ? R.get(Var::vx) : R.get(Var::vy);
  const Eigen::ArrayXXd& v_R_arr = (is_x) ? L.get(Var::vx) : L.get(Var::vy);

  const Eigen::ArrayXXd& v_L_o_arr = (is_x) ? R.get(Var::vy) : R.get(Var::vx);
  const Eigen::ArrayXXd& v_R_o_arr = (is_x) ? L.get(Var::vy) : L.get(Var::vx);

  const Eigen::ArrayXXd& p_L_arr = R.get(Var::p);
  const Eigen::ArrayXXd& p_R_arr = L.get(Var::p);

  // Loop over cells in left and right arrays and set left and right
  // scalar states for each individual Riemann problem. In direction of problem,
  // we only solve (ncells - 1) riemann problem on a given row (or column).
  for (int i = 0; i < nx - is_x; i++) {
    for (int j = 0; j < ny - !is_x; j++) {
      rho_L = rho_L_arr(i, j);
      rho_R = rho_R_arr(i + is_x, j + !is_x);

      v_L = v_L_arr(i, j);
      v_R = v_R_arr(i + is_x, j + !is_x);

      v_L_o = v_L_o_arr(i, j);
      v_R_o = v_R_o_arr(i + is_x, j + !is_x);

      p_L = p_L_arr(i, j);
      p_R = p_R_arr(i + is_x, j + !is_x);

      // Indices, in Riemann arrays, of interfaces
      int rie_idxx = i + is_x;
      int rie_idxy = j + !is_x;

      // No Riemann problem if states either side of cell interface are equal
      if (fabs(rho_L - rho_R) < 1e-14 && fabs(v_L - v_R) < 1e-14 &&
          fabs(p_L - p_R) < 1e-14) {
        rho(rie_idxx, rie_idxy) = rho_L;
        v(rie_idxx, rie_idxy) = v_L;
        v_o(rie_idxx, rie_idxy) = v_L_o;
        p(rie_idxx, rie_idxy) = p_L;
        continue;
      }

      // Define constants as defined in Toro
      double A_L = g5 / rho_L;
      double A_R = g5 / rho_R;
      double B_L = g6 * p_L;
      double B_R = g6 * p_R;

      // Define left and right sound speeds
      double cs_L = sqrt(g0 * p_L / rho_L);
      double cs_R = sqrt(g0 * p_R / rho_R);

      // Calculate pressure in star-state
      double p_star =
          find_p(v_L, v_R, p_L, p_R, cs_L, cs_R, A_L, B_L, A_R, B_R);

      // Calculate velocity in start-state
      double v_star = calc_v_star(v_L, v_R, p_L, p_R, p_star, cs_L, cs_R, A_L,
                                  B_L, A_R, B_R);

      // Determine side of interface relative to contact-discontinuity
      if (v_star > 0) {
        // interface is on left side of contact-discontinuity
        set_state(rho_L, v_L, v_L_o, v_star, p_star, p_L, cs_L, true, rie_idxx,
                  rie_idxy);
      }

      else {
        // interface is on right side of contact-discontinuity
        set_state(rho_R, v_R, v_R_o, v_star, p_star, p_R, cs_R, false, rie_idxx,
                  rie_idxy);
      }
    }
  }

  // In direction of Riemann problem, pad array either side
  // with slope-limited values. These will be equal to base grid
  // values at start of time-step.
  pad();
}

double Riemann::calc_v_star(double v_L, double v_R, double p_L, double p_R,
                            double p_star, double cs_L, double cs_R, double A_L,
                            double B_L, double A_R, double B_R) {
  // Calculation of v_star using formula from Toro
  return (0.5) * (v_L + v_R + f_K(p_star, p_R, A_R, B_R, cs_R) -
                  f_K(p_star, p_L, A_L, B_L, cs_L));
}

double Riemann::find_p(double v_L, double v_R, double p_L, double p_R,
                       double cs_L, double cs_R, double A_L, double B_L,
                       double A_R, double B_R) {
  // Find pressure star state using Newton-Raphson iterative procedure
  // Start with relative error of 1.1 (p_L and p_R could be equal even
  // if velocity or density states are not)
  double p_star = p_L;
  double p_star_old = 1.4 * p_L;
  double f_R, f_L, f_prime_L, f_prime_R, f;

  while ((fabs(p_star - p_star_old) / p_star_old) > 1e-14) {
    // Calculate f_L, f_R, f_prime_L, f_prime_R as defined in Toro
    p_star_old = p_star;
    f_R = f_K(p_star, p_R, A_R, B_R, cs_R);
    f_L = f_K(p_star, p_L, A_L, B_L, cs_L);
    f = f_R + f_L + (v_R - v_L);

    f_prime_R = f_K_prime(f_R, p_star, p_R, B_R, cs_R);
    f_prime_L = f_K_prime(f_L, p_star, p_L, B_L, cs_L);

    p_star = p_star - (f / (f_prime_R + f_prime_L));
  }

  return p_star;
}

// Calculate f_L or f_R, with expressions from Toro
double Riemann::f_K(double p_star, double p_K, double A_K, double B_K,
                    double c_sK) {
  if (p_star > p_K) {
    return (p_star - p_K) * sqrt(A_K / (p_star + B_K));
  }

  else {
    return g7 * c_sK * (pow((p_star / p_K), g8) - 1);
  }
}

// Calculate derivatives w.r.t p of f_L or f_R, with expressions from Toro
double Riemann::f_K_prime(double f_K, double p_star, double p_K, double B_K,
                          double cs_K) {
  if (p_star > p_K) {
    return (f_K / (p_star - p_K)) - (f_K / (2 * (p_star - B_K)));
  }

  else {
    return (cs_K / (p_K * g0)) * pow(p_K / p_star, g9);
  }
}

// Given star states p_star and v_star, set solution to Riemann problem
// in variable arrays of object
void Riemann::set_state(double rho_K, double v_K, double v_K_orth,
                        double v_star, double p_star, double p_K, double cs_K,
                        bool left, int i, int j) {
  // coefficient to indicate if interface is on left (-1) or
  // right (1) side of contact discontinuity
  double side_coef = left ? -1.0 : 1.0;

  // Orthogonal velocity equal to orthogonal velocity
  // of state left or right of contact-discontinuity, respectively
  v_o(i, j) = v_K_orth;

  if (p_star > p_K) {
    // Shock case

    // Calculate shock velocity
    double S_K =
        v_K + side_coef * (cs_K * sqrt((g1 * p_star + g2 * p_K) / (g10 * p_K)));

    if (side_coef * S_K > 0) {
      // Cell interface is in star state.
      // (Positive shock speed if cell interface
      // to right of contact discontinuity. Negative shock
      // speed if left of contact discontinuity)
      rho(i, j) = rho_K * (p_K * g2 + p_star * g1) / (p_star * g2 + p_K * g1);
      v(i, j) = v_star;
      p(i, j) = p_star;
    }

    else {
      // Cell interface takes on initial state K
      rho(i, j) = rho_K;
      v(i, j) = v_K;
      p(i, j) = p_K;
    }
  }

  else {
    // Rarefaction case

    // define sound speed in star region
    double cs_L_star = cs_K * pow(p_star / p_K, g8);

    // define velocity of the tail and head of the rarefaction, respectively
    double S_TK = v_star + side_coef * cs_L_star;
    double S_HK = v_K + side_coef * cs_K;

    if (side_coef * S_TK > 0) {
      // If cell interface left of contact discontinuity (i.e. side_coef = -1),
      // this is case where tail of rarefaction has negative velocity.
      // If cell interface right of contact discontinuity (i.e. side_coef = 1),
      // this is case where tail of rarefaction has positive velocity.

      // Cell interface takes on star state
      rho(i, j) = rho_K * pow(p_star / p_K, g11);
      v(i, j) = v_star;
      p(i, j) = p_star;
    }

    else if (side_coef * S_HK > 0 && side_coef * S_TK <= 0) {
      // Case where cell interface is inside rarefaction

      // define factor present in calculation of rho_fan, v_fan, and p_fan
      double fac = (2 * cs_K + (-side_coef) * g2 * v_K) / (g1 * cs_K);

      // Set rarefaction fan variables
      rho(i, j) = rho_K * pow(fac, g7);
      v(i, j) = ((-side_coef) * cs_K * fac);
      p(i, j) = p_K * pow(fac, g12);
    }

    else {
      // If cell interface left of contact discontinuity (i.e. side_coef = -1),
      // this is case where head of rarefaction has positive velocity.
      // If cell interface right of contact discontinuity (i.e. side_coef = 1),
      // this is case where head of rarefaction has negative velocity.

      // cell interface takes on original left or right state respectively
      rho(i, j) = rho_K;
      v(i, j) = v_K;
      p(i, j) = p_K;
    }
  }
}

// Pad Riemann arrays. In direction of problem, the first and last
// columns will not have a Riemann problem. Limited half-step solutions
// will have values of base-grid at start of time-step. Thus, pad with values
// from back of left-limited state and front of right-limited state
void Riemann::pad() {
  const Eigen::ArrayXXd& old_rho_L = L.get(Var::rho);
  const Eigen::ArrayXXd& old_vx_L = L.get(Var::vx);
  const Eigen::ArrayXXd& old_vy_L = L.get(Var::vy);
  const Eigen::ArrayXXd& old_p_L = L.get(Var::p);

  const Eigen::ArrayXXd& old_rho_R = R.get(Var::rho);
  const Eigen::ArrayXXd& old_vx_R = R.get(Var::vx);
  const Eigen::ArrayXXd& old_vy_R = R.get(Var::vy);
  const Eigen::ArrayXXd& old_p_R = R.get(Var::p);

  if (is_x) {
    rho.row(0) = old_rho_L.row(0);
    rho.row(rho.rows() - 1) = old_rho_R.row(old_rho_R.rows() - 1);

    vx.row(0) = old_vx_L.row(0);
    vx.row(rho.rows() - 1) = old_vx_R.row(old_rho_R.rows() - 1);

    vy.row(0) = old_vy_L.row(0);
    vy.row(rho.rows() - 1) = old_vy_R.row(old_rho_R.rows() - 1);

    p.row(0) = old_p_L.row(0);
    p.row(rho.rows() - 1) = old_p_R.row(old_rho_R.rows() - 1);
  }

  else {
    rho.col(0) = old_rho_L.col(0);
    rho.col(rho.cols() - 1) = old_rho_R.col(old_rho_R.cols() - 1);

    vx.col(0) = old_vx_L.col(0);
    vx.col(rho.cols() - 1) = old_vx_R.col(old_rho_R.cols() - 1);

    vy.col(0) = old_vy_L.col(0);
    vy.col(rho.cols() - 1) = old_vy_R.col(old_rho_R.cols() - 1);

    p.col(0) = old_p_L.col(0);
    p.col(rho.cols() - 1) = old_p_R.col(old_rho_R.cols() - 1);
  }
}
