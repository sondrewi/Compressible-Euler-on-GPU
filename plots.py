import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

data_37 = np.loadtxt("BUB_37_Sols.dat")
data_185 = np.loadtxt("BUB_185_Sols.dat")
data_482 = np.loadtxt("BUB_482_Sols.dat")
data_Quad = np.loadtxt("QUAD_400_Sols.dat")

x = data_185[:, 0]
y = data_185[:, 1]
'''rho_CPU_Quad = data_Quad[:, 2]
rho_CUDA_Quad = data_Quad[:, 3]'''
rho_CUDA_37 = data_37[:, 3]
rho_CUDA_185 = data_185[:, 3]
rho_CUDA_482 = data_482[:, 3]
'''vx_CPU = data[:, 4]
vx_CUDA = data[:, 5]
vy_CPU = data[:, 6]
vy_CUDA = data[:, 7]'''
p_CPU_482 = data_482[:, 8]
p_CUDA_37 = data_37[:, 9]
p_CUDA_185 = data_185[:, 9]
p_CUDA_482 = data_482[:, 9]
'''p_CPU_Quad = data_Quad[:, 8]
p_CUDA_Quad = data_Quad[:, 9]'''

nx, ny = 500, 197
rho_CUDA_37 = rho_CUDA_37.reshape((nx, ny))
rho_CUDA_185 = rho_CUDA_185.reshape((nx, ny))
rho_CUDA_482 = rho_CUDA_482.reshape((nx, ny))
'''rho_CPU_Quad = rho_CPU_Quad.reshape((nx, ny))
p_CPU_Quad = p_CPU_Quad.reshape((nx, ny))
rho_CUDA_Quad = rho_CUDA_Quad.reshape((nx, ny))
p_CUDA_Quad = p_CUDA_Quad.reshape((nx, ny))'''


'''vx_CUDA = vx_CUDA.reshape((nx, ny))
vx_CPU = vx_CPU.reshape((nx, ny))
vy_CUDA = vy_CUDA.reshape((nx, ny))
vy_CPU = vy_CPU.reshape((nx, ny))'''
p_CUDA_37 = p_CUDA_37.reshape((nx, ny))
p_CUDA_185 = p_CUDA_185.reshape((nx, ny))
p_CUDA_482 = p_CUDA_482.reshape((nx, ny))
p_CPU_482 = p_CPU_482.reshape((nx, ny))
x = x.reshape((nx, ny))
y = y.reshape((nx, ny))

#diff_rho = (rho_CPU - rho_CUDA) / np.max(rho_CPU)
#diff_vx = (vx_CPU - vx_CUDA) / np.max(vx_CPU)
#diff_vy = (vx_CPU - vx_CUDA) / np.max(vy_CPU)
diff_p_482 = (p_CPU_482 - p_CUDA_482) / \
    (np.max(p_CPU_482) - np.min(p_CPU_482))
# print(diff_p_482.max())

mask = y[0, :] >= 0
x_up = x[:, mask]
y_up = y[:, mask]
rho_CUDA_37_up = rho_CUDA_37[:, mask]
rho_CUDA_185_up = rho_CUDA_185[:, mask]
rho_CUDA_482_up = rho_CUDA_482[:, mask]
p_CPU_482_up = p_CPU_482[:, mask]
p_CUDA_37_up = p_CUDA_37[:, mask]
p_CUDA_185_up = p_CUDA_185[:, mask]
p_CUDA_482_up = p_CUDA_482[:, mask]
diff_p_482_up = diff_p_482[:, mask]

fig, axs = plt.subplots(4, 1, figsize=(9, 10))
plt.subplots_adjust(hspace=0.3)

levels_37 = np.linspace(rho_CUDA_37.min(),
                        rho_CUDA_37.max(), num=25)
levels_185 = np.linspace(rho_CUDA_185.min(),
                         rho_CUDA_185.max(), num=25)
levels_482 = np.linspace(rho_CUDA_482.min(),
                         rho_CUDA_482.max(), num=25)

colors = [
    (0, 0, 0.7),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1.0, 0.4, 0.4),
    (1, 0, 0),
    (0.5, 0, 0.5),
]

y_ticks = [0.0, 0.02, 0.0445]

# Create a colormap object
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

axs[0].tick_params(axis='both', which='major', labelsize=10)

# Plot Pressure with Density contours
c1 = axs[0].pcolormesh(x_up, y_up, p_CUDA_37_up,
                       cmap=custom_cmap, shading='auto')
cont1 = axs[0].contour(x_up, y_up, rho_CUDA_37_up, colors='black',
                       levels=levels_37, linewidths=0.5)
axs[0].set_title('CUDA: $\hat{t}=0.6$',
                 fontsize=10, pad=0)
axs[0].set_aspect('equal')
axs[0].set_yticks(y_ticks)
axs[0].set_yticklabels(["0", "0.02", "0.0445"])
axs[0].set_xticks([])
axs[0].set_xticklabels([])

cb1 = fig.colorbar(c1, ax=[axs[0], axs[1], axs[2], axs[3]],
                   location='top', pad=0.07, fraction=0.045)
cb1.ax.tick_params(labelsize=10, top=True, bottom=False)
desired_num_ticks = 6  # Specify the number of ticks you want
cb1.locator = MaxNLocator(nbins=desired_num_ticks)
cb1.update_ticks()

axs[1].tick_params(axis='both', which='major', labelsize=10)

c2 = axs[1].pcolormesh(x_up, y_up, p_CUDA_185_up,
                       cmap=custom_cmap, shading='auto')
cont2 = axs[1].contour(x_up, y_up, rho_CUDA_185_up, colors='black',
                       levels=levels_185, linewidths=0.5)

axs[1].set_title('CUDA: $\hat{t}=3.0$', fontsize=10, pad=0)
axs[1].set_xticks([])
axs[1].set_yticks(y_ticks)
axs[1].set_yticklabels(["0", "0.02", "0.0445"])
axs[1].set_xticklabels([])
axs[1].set_aspect('equal')

axs[2].tick_params(axis='both', which='major', labelsize=10)

c3 = axs[2].pcolormesh(x_up, y_up, p_CUDA_482_up,
                       cmap=custom_cmap, shading='auto')
cont3 = axs[2].contour(x_up, y_up, rho_CUDA_482_up, colors='black',
                       levels=levels_185, linewidths=0.5)
axs[2].set_title('CUDA: $\hat{t}=7.8$', fontsize=10, pad=0)
axs[2].set_xticks([])
axs[2].set_yticks(y_ticks)
axs[2].set_xticklabels([])
axs[2].set_yticklabels(["0", "0.02", "0.0445"])
axs[2].set_aspect('equal')

axs[2].tick_params(axis='both', which='major', labelsize=10)
c4 = axs[3].pcolormesh(x_up, y_up, diff_p_482_up,
                       cmap='viridis', shading='auto')
# cont2 = axs[2].contour(x_up, y_up, rho_CUDA_, colors='black',
# levels=levels, linewidths=0.5)
axs[3].set_title('Normalised Difference in Pressure', fontsize=10, pad=-15)
axs[3].set_aspect('equal')
axs[3].set_yticks(y_ticks)
axs[3].set_yticklabels(["0", "0.02", "0.0445"])

plt.savefig('Sols_BUB_press.png')
