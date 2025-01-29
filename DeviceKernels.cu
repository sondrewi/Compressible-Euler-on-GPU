#include "DeviceKernels.cuh"

//Set initial conditions for quadrant problem directly on GPU
__global__ void quad_init(double** sol_device, const double dx, const double dy,
                          const int x_dim, const int y_dim) {

  //Cell ordered in blocks of 1024 since no communication between them required
  //Get cell indices
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int i = k % x_dim;
  int j = k / x_dim;

  //Is cell in lower or upper part of domain
  bool left = ((i + 0.5) * dx <= 0.5);
  bool down = ((j + 0.5) * dy <= 0.5);

  //Declare array to hold primitive initial values
  double u[4];

  //Read into array according to cell's location in mesh
  u[0] = down ? (left ? 0.138 : 0.5323) : (left ? 0.5323 : 1.5);
  u[1] = left ? 1.206 : 0.0;
  u[2] = down ? 1.206 : 0.0;
  u[3] = down ? (left ? 0.029 : 0.3) : (left ? 0.3 : 1.5);

  //Convert solution to conservative variables
  prim_to_cons(u);

  //Read to device
  if (k < (x_dim * y_dim)) {
    sol_device[0][k] = u[0];
    sol_device[1][k] = u[1];
    sol_device[2][k] = u[2];
    sol_device[3][k] = u[3];
  }

  __syncthreads();
}

//Set initial conditions for bubble problem directly on GPU
__global__ void bubble_init(double** sol_device, const double dx, const double dy,
                          const int x_dim, const int y_dim){
  //Same set-up as in quadrant problem
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int i = k % x_dim;
  int j = k / x_dim;

  double x = (i + 0.5) * dx;
  double y = (j + 0.5) * dy - 0.0445;

  bool left = (x < 0.005);
  bool bubble = !left && (sqrt(pow(x - 0.035, 2) + pow(y, 2)) < 0.025);

  double u[4];

  u[0] = left ? init_rho_L : (bubble ? init_rho_Hel : init_rho_R);
  u[1] = left ? init_vx_L : 0.0;
  u[2] = 0.0;
  u[3] = left ? init_p_L : init_p_R;

  prim_to_cons(u);

  if (k < (x_dim * y_dim)) {
    sol_device[0][k] = u[0];
    sol_device[1][k] = u[1];
    sol_device[2][k] = u[2];
    sol_device[3][k] = u[3];
  }
  __syncthreads();
}

//Convert set of conservative solutions on device to primitive
__global__ void convert(double** sol_device, int x_dim, int y_dim) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;

  if (k < (x_dim * y_dim)) {
    double u[4];
    u[0] = sol_device[0][k];
    u[1] = sol_device[1][k];
    u[2] = sol_device[2][k];
    u[3] = sol_device[3][k];

    cons_to_prim(u);

    sol_device[1][k] = u[1];
    sol_device[2][k] = u[2];
    sol_device[3][k] = u[3];
  }
  __syncthreads();
}

//Device kernel to update solution. All effort has been made to keep
//the maximal amount of information possible on the register memory of each thread
//to avoid spillage to the private section of global memory allocated to each thread
__global__ void register_step(double* sol[4],
                              const double step_const, int n_cells) {
  //Initialise shared memory
  extern __shared__ double shared_mem[];

  //Initialise arrays to hold solution at each cell
  double u1[4];
  double u2[4];

  //Initialise pointer to location in shared memory at which cell corresponding
  //to this thread has its value stored
  double* old_sol[4];
  double* hold;

  //Compute global index, in mesh, of corresponding cell
  int glob_idx = blockDim.x * (blockIdx.x + threadIdx.y) + threadIdx.x;

  //Compute index in shared memory at which to store this cell value
  //Shared memory has 5 sections, and this index corresponds to the position in each section
  int shared_id = (blockDim.x + 2) * threadIdx.y + threadIdx.x + 1;

  //Compute shared memory index for the placeholder location
  hold = &shared_mem[(blockDim.x + 2) * blockDim.y * 4 + shared_id];

  __syncthreads();

  //Iterate over variables rho, parellel momentum, perpendicular momentum, and energy
  for (int k = 0; k < 4; k++) {

    //Make old_sol[k] point to appropriate section in shared memory
    old_sol[k] = &shared_mem[(blockDim.x + 2) * blockDim.y * k + shared_id];
    __syncthreads();

    //Read old value to u1[k] and additionally store in old_sol
    u1[k] = (glob_idx < n_cells) ? sol[k][glob_idx] : 1.0;
    __syncthreads();
    *old_sol[k] = u1[k];

    //Set ghost cell value at edges of shared memory rows
    if (threadIdx.x == 0) {
      *(old_sol[k] - 1) = u1[k];
    }

    else if (threadIdx.x == blockDim.x - 1) {
      *(old_sol[k] + 1) = u1[k];
    }

    __syncthreads();

    //Calculate cell(i+1)_value - cell(i)_value
    double deltai_plus = *(old_sol[k] + 1) - u1[k];

    __syncthreads();

    //Calculate cell(i)_value - cell(i-1)_value
    double deltai_min = u1[k] - *(old_sol[k] - 1);

    //If both of these differences not zero, possibility that Xi is not zero
    //Calculate Xi and calculate limited values
    if (deltai_plus != 0 && deltai_min != 0) {
      double deltai_Xi = limiter(deltai_min, deltai_plus);
      u2[k] = u1[k] + 0.5 * deltai_Xi;
      u1[k] -= 0.5 * deltai_Xi;
    }

    //Else slope limited values equal to old solution
    else {
      u2[k] = u1[k];
    }
    __syncthreads();
  }
  __syncthreads();

  // half time-step update
  {
    double netfluxes[4];
    calc_netflux(u1, u2, netfluxes);

    for (int k = 0; k < 4; k++) {
      u1[k] += 0.5 * step_const * netfluxes[k];
      u2[k] += 0.5 * step_const * netfluxes[k];
    }
  }

  //Convert half-step limited values to primitive
  cons_to_prim(u1);
  cons_to_prim(u2);

  __syncthreads();

  // Gather u_L in cell right of this cell
  // Afterwords, we have riemann problem where
  // u2 is left state and u1 is right state
  for (int k = 0; k < 4; k++) {
    __syncthreads();

    *hold = u1[k];
    __syncthreads();

    //We solve n_cols riemann problems
    //Rightmost cell copies its value to right ghost-cell
    if (threadIdx.x + 1 == blockDim.x) {
      *(hold + 1) = u1[k];
    }

    __syncthreads();
    //Get u_L[k] in cell right of this one
    u1[k] = *(hold + 1);
  }

  //Solve Riemann problem if difference in any primitive variable exceeds 1e-14
  if (fabs(u1[0] - u2[0]) > 1e-14 || fabs(u1[1] - u2[1]) > 1e-14 ||
      fabs(u1[3] - u2[3]) > 1e-14) {
    double star_states[2];
    const double cs_L = sqrt(g0 * u2[3] / u2[0]);
    const double cs_R = sqrt(g0 * u1[3] / u1[0]);
    find_star(u2, u1, star_states, cs_L, cs_R);

    // Determine side of interface relative to contact-discontinuity
    if (star_states[0] > 0) {
      // interface is on left side of contact-discontinuity
      // pass left state to set_state
      set_state(u2, star_states, cs_L);
    }

    else {
      // interface is on right side of contact-discontinuity
      // pass right state (u1) to set_state
      set_state(u1, star_states, cs_R);

      //We store solution in u2
      u2[0] = u1[0];
      u2[1] = u1[1];
      u2[2] = u1[2];
      u2[3] = u1[3];
    }
  }

  __syncthreads();

  // u2 now contains solution to riemann
  // problem at right interface w.r.t cell centre

  //convert to conservative variables
  prim_to_cons(u2);

  //Gather Riemann solution at cell interface left of cell_centre
  for (int k = 0; k < 4; k++) {
    // Read riemann solutions to shared memory
    *hold = u2[k];
    __syncthreads();

    //Gather Riemann solution left of cell centre
    u1[k] = *(hold - 1);
    __syncthreads();

    // No Riemann problem at left interface of first cell
    // Left solution is therefore just the old solution
    if (threadIdx.x == 0) {
      u1[k] = *(old_sol[k] - 1);
    }
    __syncthreads();
  }

  // Calculate new solution. Old solution is stored in shared
  // memory and u1 u2 contain left and right states of riemann
  // solution, respectively. Store new solution in u1
  {
    double netfluxes[4];
    calc_netflux(u1, u2, netfluxes);

    for (int k = 0; k < 4; k++) {
      __syncthreads();
      u1[k] = *old_sol[k] + step_const * netfluxes[k];

      __syncthreads();
      if (glob_idx < n_cells) {
        //Read new solution to memory
        sol[k][glob_idx] = u1[k];
      }
    }
  }

  __syncthreads();
}

//Limiter function to compute Xi*delta_i for a cell
__device__ double limiter(double deltai_min, double deltai_plus) {
  double r, Xi_R;

  //Define r and Xi_R
  r = deltai_min / deltai_plus;
  Xi_R = 2 / (1 + r);

  //Calculate Xi using superbee limiter
  double Xi = (r <= 0.5) ? ((r <= 0.0) ? 0 : 2*r) : ((r > 1.0) ? fmin(r, fmin(Xi_R, 2.0)): 1.0);

  //Calculate deltai_i
  double deltai = 0.5 * ((1 + om) * deltai_min + (1 - om) * deltai_plus);

  //Return product of Xi and delta_i
  return Xi * deltai;
}

//Primitive to conservative conversion
__device__ void prim_to_cons(double (&u)[4]) {
  u[3] = g4 * u[3] + 0.5 * u[0] * (u[1] * u[1] + u[2] * u[2]);
  u[1] *= u[0];
  u[2] *= u[0];
}

//Conservative to primitive conversion
__device__ void cons_to_prim(double (&u)[4]) {
  u[1] /= u[0];
  u[2] /= u[0];
  u[3] = g2 * (u[3] - 0.5 * u[0] * (u[1] * u[1] + u[2] * u[2]));
}

//Calculate net flux using conservative variables
//Note that for an x-update, x-momentum is stored in u[1] and
//y-momentum is stored in u[2]. Vice-versa for y-update
__device__ void calc_netflux(double (&u_L)[4], double (&u_R)[4],
                             double (&netfluxes)[4]) {
  double v_L = u_L[1] / u_L[0];
  double v_R = u_R[1] / u_R[0];

  double v_L_o = u_L[2] / u_L[0];
  double v_R_o = u_R[2] / u_R[0];

  double p_L = g2 * (u_L[3] - 0.5 * (u_L[1] * v_L + u_L[2] * v_L_o));
  double p_R = g2 * (u_R[3] - 0.5 * (u_R[1] * v_R + u_R[2] * v_R_o));

  netfluxes[0] = u_L[1] - u_R[1];
  netfluxes[1] = (u_L[1] * v_L + p_L) - (u_R[1] * v_R + p_R);
  netfluxes[2] = u_L[2] * v_L - u_R[2] * v_R;
  netfluxes[3] = ((u_L[3] + p_L) * v_L) - ((u_R[3] + p_R) * v_R);
}

//Function to find star states p_star and v_star for Riemann problem
__device__ void find_star(double (&u_L)[4], double (&u_R)[4],
                          double (&star_states)[2], const double cs_L,
                          const double cs_R) {

  //Calculate constants as defined in Toro
  const double A_L = g5 / u_L[0];
  const double A_R = g5 / u_R[0];
  const double B_L = g6 * u_L[3];
  const double B_R = g6 * u_R[3];

  //Set initial values for pressure
  double p_star = u_L[3];
  double p_star_old = 1.4 * u_L[3];
  double f_R, f_L, f_prime_L, f_prime_R;

  //Calculate p_star using Newton-Raphson iterative procedure
  while ((fabs(p_star - p_star_old) / p_star_old) > 1e-14) {
    p_star_old = p_star;

    //Calculate f_R, f_L, f_prime_R, and f_prime_L as defined in Toro
    f_R = f_K(p_star, u_R[3], A_R, B_R, cs_R);
    f_L = f_K(p_star, u_L[3], A_L, B_L, cs_L);

    f_prime_R = f_K_prime(f_R, p_star, u_R[3], B_R, cs_R);
    f_prime_L = f_K_prime(f_L, p_star, u_L[3], B_L, cs_L);

    //update p_star
    p_star = p_star - ((f_R + f_L + (u_R[1] - u_L[1])) / (f_prime_R + f_prime_L));
  }

  //Set star state velocity
  star_states[0] = 0.5 * (u_L[1] + u_R[1] + f_K(p_star, u_R[3], A_R, B_R, cs_R) -
                          f_K(p_star, u_L[3], A_L, B_L, cs_L));

  //Set star state for pressure
  star_states[1] = p_star;
}

//Functions to calculate f_K as defined in Toro
__device__ double f_K(double p_star, double p_K, double A_K, double B_K,
                      double cs_K) {
  if (p_star > p_K) {
    return (p_star - p_K) * sqrt(A_K / (p_star + B_K));
  }

  else {
    return g7 * cs_K * (pow((p_star / p_K), g8) - 1);
  }
}

//Functions to calculate f_prime_K as defined in Toro
__device__ double f_K_prime(double f_K, double p_star, double p_K, double B_K,
                            double cs_K) {
  if (p_star > p_K) {
    return (f_K / (p_star - p_K)) - (f_K / (2 * (p_star - B_K)));
  }

  else {
    return (cs_K / (p_K * g0)) * pow(p_K / p_star, g9);
  }
}

//Set solution to Riemann problem given star states
__device__ void set_state(double (&u_K)[4], double (&star_states)[2], const double cs_K) {

  //Set side coefficient. -1 indicates contact discontinuity is right
  //of contact discontinuity and vice-versa.
  int side_coef = star_states[0] > 0 ? -1 : 1;

  if (star_states[1] > u_K[3]) {
    // Shock case

    //Calculate shock velocity
    double S_K =
        u_K[1] + side_coef * (cs_K * sqrt((g1 * star_states[1] + g2 * u_K[3]) / (g10 * u_K[3])));

    if (side_coef * S_K > 0) {
      // Cell interface is in star state
      // (positive shock speed if cell interface
      // to right of contact discontinuity negative shock
      // speed if left of contact discontinuity
      u_K[0] = u_K[0]*(u_K[3] * g2 + star_states[1] * g1) / (star_states[1] * g2 + u_K[3] * g1);
      u_K[1] = star_states[0];
      u_K[3] = star_states[1];
    }
    // Otherwise cell interface takes on state of initial state K
    // and so no need to change u_K
  }

  else {
    // Rarefaction case

    // define sound speed in star region
    double cs_K_star = cs_K * pow(star_states[1] / u_K[3], g8);

    // define velocity of the tail and head of the rarefaction
    double S_TK = star_states[0] + side_coef * cs_K_star;
    double S_HK = u_K[1] + side_coef * cs_K;

    if (side_coef * S_TK > 0) {
      // If cell interface left of contact discontinuity (i.e. side_coef =
      // -1), this is case where tail of rarefaction has negative velocity. If
      // cell interface right of contact discontinuity (i.e. side_coef = 1),
      // this is case where tail of rarefaction has positive velocity.

      // Cell interface takes on star state
      u_K[0] = u_K[0]*pow(star_states[1] / u_K[3], g11);
      u_K[1] = star_states[0];
      u_K[3] = star_states[1];
    }

    else if (side_coef * S_HK > 0 && side_coef * S_TK <= 0) {
      // Case where cell interface is inside rarefaction

      // define factor present in calculation of rho_fan, v_fan, and p_fan
      double fac = (2 * cs_K + (-side_coef) * g2 * u_K[1]) / (g1 * cs_K);

      u_K[0] = u_K[0] * pow(fac, g7);
      u_K[1] = ((-side_coef) * cs_K * fac);
      u_K[3] = u_K[3] * pow(fac, g12);
    }

    // If cell interface left of contact discontinuity (i.e. side_coef =
    // -1), this is case where tail of rarefaction has positive velocity. If
    // cell interface right of contact discontinuity (i.e. side_coef = 1),
    // this is case where tail of rarefaction has negative velocity.

    // cell interface takes on left or right state respectively. no need to
    // change u
  }
}

//Kernel to transpose domain. Transpose one variable with this kernel
__global__ void transpose_kern(const double* a, double* aT, int w, int h) {
  //Use blocks of size 32x32
  //Initialise shared memory with extra column to avoid shared memory conflicts
  __shared__ double smem[32][33];

  //Get index of cell to read from global memory
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int read_i = i + j*w;

  __syncthreads();

  // Read to shared memory in coalesced fashion wrt array a,
  //transposing directly within block
  if(i < w && j < h){
    smem[threadIdx.x][threadIdx.y] = a[read_i];
  }

  //Compute indices for writing back to global memory
  i = blockDim.y * blockIdx.y + threadIdx.x;
  j = blockDim.x * blockIdx.x + threadIdx.y;
  int write_i = i + j*h;

  __syncthreads();

  if(i < h && j < w)
  {
    //read to global memory in coalesced fashion
    aT[write_i] = smem[threadIdx.y][threadIdx.x];
  }

  __syncthreads();

  /*int k = blockDim.x * blockIdx.x + threadIdx.x;
  int i = k % w;
  int j = k / w;

  int readi = j * w + i;

  int writei = i * h + j;

  if (k < h * w) {
    aT[writei] = a[readi];
  }*/
}

//Kernel to calculate maximal wavespeed across domain
__global__ void calc_ws_kern(double** sol_device, double* block_max_ws,
                             double* block_max_ws_idx, int x_dim, int y_dim) {

  //We may want index of max wavespeed. One shared memory array for wavespeeds
  //and one for corresponding cell indices. Blocks of maximal size 1024
  __shared__ double maximum[1024];
  __shared__ int idx[1024];

  //Get index, in terms of global memory array of cell corresponding to thread
  int k = blockDim.x * blockIdx.x + threadIdx.x;

  //Read index to shared memory
  idx[threadIdx.x] = k;

  if (k < x_dim * y_dim) {
    //Get conservative variable solutions from
    //global memory and convert to primitive variables

    double sol_prim[4];
    sol_prim[0] = sol_device[0][k];
    sol_prim[1] = sol_device[1][k];
    sol_prim[2] = sol_device[2][k];
    sol_prim[3] = sol_device[3][k];

    cons_to_prim(sol_prim);

    //Calculate wavespeed in cell
    maximum[threadIdx.x] =
        sqrt(sol_prim[1] * sol_prim[1] + sol_prim[2] * sol_prim[2]) +
        sqrt(g0 * sol_prim[3] / sol_prim[0]);
  }

  //If thread has index beyond size of arrays, set wavespeed to zero
  else {
    maximum[threadIdx.x] = 0.0;
  }

  __syncthreads();


  //Reduce over each block to get block maximum of wavespeed along with index of corresponding cell
  for (int i = 0; i < 10; i++) {
    int offset = 512 / pow(2, i);
    if (threadIdx.x < offset) {
      idx[threadIdx.x] = (maximum[threadIdx.x] > maximum[threadIdx.x + offset])
                           ? idx[threadIdx.x]
                           : idx[threadIdx.x + offset];
      maximum[threadIdx.x] =
          fmax(maximum[threadIdx.x], maximum[threadIdx.x + offset]);
    }

    __syncthreads();
  }

  //Write max wavespeed to device array and similar for corresponding cell index
  if (threadIdx.x == 0) {
    block_max_ws[blockIdx.x] = maximum[threadIdx.x];
    block_max_ws_idx[blockIdx.x] = idx[threadIdx.x];
  }
}
