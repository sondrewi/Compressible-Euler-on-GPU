#include "Slope_limiter.H"
using namespace std;

Slope_limiter::Slope_limiter(const int nx_, const int ny_, bool is_x_) {
  is_x = is_x_;
  nx = nx_;
  ny = ny_;
}

// Calculate delta_i,r, and Xi_R
void Slope_limiter::calc_deltai(const Eigen::ArrayXXd& old_sol) {
  // in direction of limiting deltai_plus must be one cell wider than
  // base mesh in order to hold difference between cells. Discard
  // ghost cells in direction over which we are not limiting
  Eigen::ArrayXXd deltai_plus;
  deltai_plus.resize(nx + is_x, ny + !is_x);

  // Calculate difference between cells in direction of limiting
  deltai_plus = old_sol.block(1, 1, nx + is_x, ny + !is_x) -
                old_sol.block(!is_x, is_x, nx + is_x, ny + !is_x);

  // Calculate ratio of backward difference to forward difference
  r = deltai_plus.block(0, 0, nx, ny) / deltai_plus.block(is_x, !is_x, nx, ny);
  // r has same size has base grid (nx, ny)

  // If forward difference zero, r will be nan -> set to zero
  r = r.unaryExpr([](double val) { return std::isnan(val) ? 0.0 : val; });

  Xi_R = 2 / (1 + r);

  // Remove inf values from r, neagtive inf will make for Xi = 0 in superbee
  // positive inf will make for Xi = min(Xi_r, 2) in superbee
  r = r.unaryExpr([](double val) {
    if (std::isinf(val)) {
      return std::signbit(val) ? -1.0 : 2.0;
    } else {
      return val;  // Leave the value unchanged if it's not infinity
    }
  });

  // calculate delta_i
  deltai = 0.5 * ((1 + om) * deltai_plus.block(0, 0, nx, ny) +
                  (1 - om) * deltai_plus.block(is_x, !is_x, nx, ny));
}

void Slope_limiter::calc_limiter() {
  // Superbee calculation for Xi
  Xi = ((r <= 0.5 && r > 0).cast<double>() * 2 * r) +
       (r <= 1 && r > 0.5).cast<double>() +
       ((r > 1).cast<double>() * Xi_R.min(r.min(2)));
}
