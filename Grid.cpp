#include "Grid.H"

Grid::Grid() {}

// Constructor for generic solution (not a base_grid that will
// hold actual solution in time). No ghost cells
Grid::Grid(int nx_, int ny_) {
  base = false;

  // Set x- and y- dimensions without ghost cells
  nx = nx_;
  ny = ny_;

  int x_dim = nx;
  int y_dim = ny;

  // resize solution arrays, no ghost cells.
  p.resize(x_dim, y_dim);
  rho.resize(x_dim, y_dim);
  vx.resize(x_dim, y_dim);
  vy.resize(x_dim, y_dim);
  xmom.resize(x_dim, y_dim);
  ymom.resize(x_dim, y_dim);
  E.resize(x_dim, y_dim);
}

// Constructor for base grid that will hold solutions at cell centres in time
// Includes 2 ghost cell (1 either side of domain) in x- and y-directions
Grid::Grid(const double xmin, const double xmax, const double ymin,
           const double ymax, const int x_cells, const int y_cells) {
  // Set grid specifications
  x_min = xmin;
  x_max = xmax;
  y_max = ymax;
  y_min = ymin;

  base = true;

  // Set cell number
  nx = x_cells;
  ny = y_cells;

  // Set directional delta between cells
  dx = (xmax - xmin) / x_cells;
  dy = (y_max - y_min) / y_cells;

  // Only necessary to resize Eigen arrays for primitive
  // variables during construction as arrays for other variables
  // will be constructed directly from these.
  rho.resize(x_cells + 2, y_cells + 2);
  vx.resize(x_cells + 2, y_cells + 2);
  vy.resize(x_cells + 2, y_cells + 2);
  p.resize(x_cells + 2, y_cells + 2);
}

// Function to set initial values, in terms of primitive values for
// quadrant problem or bubble
void Grid::set_init(bool bubble) {
  int quadrant;

  if (!bubble) {
    Eigen::Matrix4d init_vals;

    // Set up 4x4 array of initial values for quadrant problem
    // Left to right, bottom to top
    init_vals << 0.138, 1.206, 1.206, 0.029, 0.5323, 0, 1.206, 0.3, 0.5323,
        1.206, 0, 0.3, 1.5, 0, 0, 1.5;

    for (int i = 0; i < nx + 2; i++) {
      for (int j = 0; j < ny + 2; j++) {
        quadrant =
            (int)((i - 0.5) * dx >= 0.5) + 2 * (int)((j - 0.5) * dy >= 0.5);

        rho(i, j) = init_vals(quadrant, 0);
        vx(i, j) = init_vals(quadrant, 1);
        vy(i, j) = init_vals(quadrant, 2);
        p(i, j) = init_vals(quadrant, 3);
      }
    }
  }

  else {
    for (int i = 0; i < nx + 2; i++) {
      for (int j = 0; j < ny + 2; j++) {
        double x = x_min + (i - 0.5) * dx;
        double y = y_min + (j - 0.5) * dy;

        if (x < 0.005) {
          rho(i, j) = init_rho_L;
          vx(i, j) = init_vx_L;
          vy(i, j) = 0;
          p(i, j) = init_p_L;
        }

        else if (sqrt(pow(x - 0.035, 2) + pow(y, 2)) > 0.025) {
          rho(i, j) = init_rho_R;
          vx(i, j) = 0;
          vy(i, j) = 0;
          p(i, j) = init_p_R;
        }

        else {
          rho(i, j) = init_rho_Hel;
          vx(i, j) = 0;
          vy(i, j) = 0;
          p(i, j) = init_p_R;
        }
      }
    }
  }

  // Convert to conservative varibales
  prim_to_cons();
}

// Function to compute conservative variable arrays from existing primitive ones
void Grid::prim_to_cons() {
  xmom = vx * rho;
  ymom = vy * rho;
  E = g4 * p + 0.5 * (xmom * vx + ymom * vy);
}

// Function to compute primitive variable arrays from existing conservative ones
void Grid::cons_to_prim() {
  vx = xmom / rho;
  vy = ymom / rho;

  p = g2 * (E - 0.5 * (xmom * vx + ymom * vy));
}

// Calculate speed of sound
void Grid::calc_cs() { cs = sqrt(g0 * p / rho); }

// Set ghost cells transmissively along edges
void Grid::set_ghost() {
  std::vector<Var> types = {Var::rho, Var::xmom, Var::ymom, Var::E};

  for (const auto& type : types) {
    Eigen::ArrayXXd& grid = set(type);
    grid.row(0) = grid.row(1);
    grid.row(nx + 1) = grid.row(nx);
    grid.col(0) = grid.col(1);
    grid.col(ny + 1) = grid.col(ny);
  }
}

// Calculate fluxes over conservative variables
void Grid::calc_flux(bool is_x) {
  if (is_x) {
    f_rho = xmom;
    f_xmom = xmom * vx + p;
    f_ymom = xmom * vy;
    f_E = (E + p) * vx;
  }

  else {
    f_rho = ymom;
    f_xmom = ymom * vx;
    f_ymom = ymom * vy + p;
    f_E = (E + p) * vy;
  }
}

// Calculate timestep. Use CFL number of 1
double Grid::calc_tstep(double t, double end_t) {
  calc_cs();

  // Optinally use Eigen::Index to check point of maximal wavespeed
  // Eigen::Index max_i, max_j;

  double max_w_speed = (sqrt(vx * vx + vy * vy) + cs).maxCoeff();

  // std::cout << "Max index at: " << max_i << " " << max_j << std::endl;

  double dt = std::min(dx, dy) / max_w_speed;

  // limit timestep so that final time will always be 0.3
  if (t + dt <= end_t) {
    return dt;
  }

  else {
    return end_t - t;
  }
}

// Get the Eigen array for a given variable. Will be useful
// when performing same operation on several variables (slope-limiting)
const Eigen::ArrayXXd& Grid::get(Var type) const {
  switch (type) {
    case Var::rho:
      return rho;
    case Var::xmom:
      return xmom;
    case Var::ymom:
      return ymom;
    case Var::E:
      return E;
    case Var::vx:
      return vx;
    case Var::vy:
      return vy;
    case Var::p:
      return p;
    default:
      // Handle the error or throw an exception
      throw std::invalid_argument("Invalid ArrayType");
  }
}

// Function returning non-const reference to variable array in order
// that it may be set or altered
Eigen::ArrayXXd& Grid::set(Var type) {
  switch (type) {
    case Var::rho:
      return rho;
    case Var::xmom:
      return xmom;
    case Var::ymom:
      return ymom;
    case Var::E:
      return E;
    case Var::vx:
      return vx;
    case Var::vy:
      return vx;
    case Var::p:
      return p;

    default:
      throw std::invalid_argument("Invalid ArrayType");
  }
}

// Getter function for fluxes
const Eigen::ArrayXXd& Grid::get_flux(Var type) const {
  switch (type) {
    case Var::rho:
      return f_rho;
    case Var::xmom:
      return f_xmom;
    case Var::ymom:
      return f_ymom;
    case Var::E:
      return f_E;
    default:
      // Handle the error or throw an exception
      throw std::invalid_argument("Invalid ArrayType");
  }
}
