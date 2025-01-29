#ifndef MACROS_H
#define MACROS_H

#define g0 1.4
#define g1 (1.4 + 1.0)
#define g2 (1.4 - 1.0)
#define g3 (1.0 / (g0 + 1.0))
#define g4 (1.0 / (g0 - 1.0))
#define g5 (2.0 / (g0 + 1.0))
#define g6 ((g0 - 1.0) / (g0 + 1.0))
#define g7 (2.0 / (g0 - 1.0))
#define g8 ((g0 - 1.0) / (2.0 * g0))
#define g9 ((g0 + 1.0) / (2.0 * g0))
#define g10 (2.0 * g0)
#define g11 (1.0 / g0)
#define g12 (2.0 * g0 / (g0 - 1.0))
#define om (0.0)

#define init_p_R (101325)
#define init_rho_R (1.29)
#define init_cs_R (sqrt(g0 * init_p_R / init_rho_R))
#define init_S (1.22 * init_cs_R)
#define init_vx_L \
  (2 * init_S / (g0 + 1) - 2 * pow(init_cs_R, 2) / (init_S * (g0 + 1)))
#define init_rho_L (init_S * init_rho_R / (init_S - init_vx_L))
#define init_p_L                                    \
  (init_p_R * (2 * init_S + (g0 - 1) * init_vx_L) / \
   (2 * init_S - (g0 + 1) * init_vx_L))
#define init_rho_Hel (0.214)

#endif
