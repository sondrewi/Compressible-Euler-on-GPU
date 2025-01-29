#include "Solver.H"
using Var = Grid::Var;

// Constructor for solver class. Define number of cells
// in either direction on mesh over which we wish to solve
Solver::Solver(Grid& solution) : base_grid(solution) {
  nx = base_grid.nx;
  ny = base_grid.ny;
}

// Function to make one full time-step
double Solver::step(double t, double end_t) {
  // Calculate time-step based on previous solution
  double dt = base_grid.calc_tstep(t, end_t);

  // Step in x-direction then y-direction
  for (int dir = 0; dir < 2; dir++) {
    bool is_x = (dir == 0);

    // Get spatial delta between cell centres
    double del = (is_x) ? base_grid.dx : base_grid.dy;

    // Construct grids for the left and right-limited states
    Grid u_L = Grid(nx, ny);
    Grid u_R = Grid(nx, ny);

    // Calculate limited slopes and set u_L, u_R accordingly
    calc_lim_slopes(u_L, u_R, is_x);

    // Convert conservative solutions to primitive in order to
    // compute fluxes
    u_L.cons_to_prim();
    u_R.cons_to_prim();

    // Calculate fluxes on left- and right-limited states
    u_L.calc_flux(is_x);
    u_R.calc_flux(is_x);

    // Half-step update on u_L and u_R in-place
    hs_update(dt, del, u_L, u_R);

    // Convert back to primitive variables
    u_L.cons_to_prim();
    u_R.cons_to_prim();

    // Construct Riemann grid to hold solutions for Riemann problem
    Riemann rie = Riemann(u_L, u_R, is_x);

    // Solve Riemann problem at cell interfaces
    rie.solve();

    // Convert Riemann solutions to conservative variables
    rie.prim_to_cons();

    // Calculate flux on Riemann solutions
    rie.calc_flux(is_x);

    // Update solutions using Riemann solutions
    update(dt, del, rie, is_x);

    // Set ghost cells
    base_grid.set_ghost();

    // Convert conservative solutions to primitive in
    // order to be able to compute time-step for next iteration
    base_grid.cons_to_prim();
  }

  // Return time-step value
  return dt;
}

// Function to calculate limited slopes u_L and u_R
void Solver::calc_lim_slopes(Grid& u_L, Grid& u_R, bool is_x) {
  // Construct slope limiter object
  Slope_limiter SL = Slope_limiter(nx, ny, is_x);

  // Vector of variables over which to limit slopes
  std::vector<Var> types = {Var::rho, Var::xmom, Var::ymom, Var::E};

  for (const auto& type : types) {
    // Get reference to arrays on which to set limited slopes
    Eigen::ArrayXXd& var_to_update_L = u_L.set(type);
    Eigen::ArrayXXd& var_to_update_R = u_R.set(type);

    // Calculate deltai and Xi using old variable solution
    SL.calc_deltai(base_grid.get(type));
    SL.calc_limiter();

    // Get reference to old solution, without ghost cells
    const Eigen::ArrayXXd& old_var_sol =
        base_grid.get(type).block(1, 1, nx, ny);

    // Set left and right limited slopes
    Eigen::ArrayXXd half_deltai_Xi = 0.5 * SL.deltai * SL.Xi;
    var_to_update_L = old_var_sol - half_deltai_Xi;
    var_to_update_R = old_var_sol + half_deltai_Xi;
  }
}

// Update base grid solution using available Riemann solution.
void Solver::update(double dt, double del, Grid& riemann, bool is_x) {
  std::vector<Var> types = {Var::rho, Var::xmom, Var::ymom, Var::E};

  for (const auto& type : types) {
    Eigen::ArrayXXd& var_to_update = base_grid.set(type);
    const Eigen::ArrayXXd& flux = riemann.get_flux(type);

    var_to_update.block(1, 1, nx, ny) +=
        (dt / del) *
        (flux.block(0, 0, nx, ny) - flux.block(is_x, !is_x, nx, ny));
  }
}

// Half step update for right- and left limited solution made in-place
void Solver::hs_update(double dt, double del, Grid& u_L, Grid& u_R) {
  std::vector<Var> types = {Var::rho, Var::xmom, Var::ymom, Var::E};

  for (const auto& type : types) {
    Eigen::ArrayXXd& var_to_update_L = u_L.set(type);
    Eigen::ArrayXXd& var_to_update_R = u_R.set(type);

    const Eigen::ArrayXXd& flux_left = u_L.get_flux(type);
    const Eigen::ArrayXXd& flux_right = u_R.get_flux(type);

    const Eigen::ArrayXXd change = (0.5 * dt / del) * (flux_left - flux_right);
    var_to_update_L += change;
    var_to_update_R += change;
  }
}
