# Ce code modelise l'APS sur la fenêtre Omega (taille l1 x l2) de N images
import os, sys
from inspect import getsourcefile
from os.path import abspath, dirname

os.chdir(dirname(abspath(getsourcefile(lambda:0))))
plt.savefig('figure_1_APS.eps', format='eps', bbox_inches='tight', dpi=1200)

###
import numpy as np
import pylab as plt
from numpy import pi, sin, sqrt
from numpy.random import rand, multivariate_normal, normal
from mpl_toolkits.axes_grid1 import make_axes_locatable

N = 8
h = int(N/2)
l = int(N/h)
l1 = 3
l2 = 6
L = l1 * l2


alpha = normal(0, 1, N)
Alpha = np.repeat(alpha, L).reshape((N,L)) + normal(0, 0.2, (N,L))

z_min, z_max = -np.abs(Alpha).max(), np.abs(Alpha).max()

f, axarr = plt.subplots(h, l,figsize=(5, 5))
contour_sets = []
for i in range(h):
    for j in range(l):
        print("i,j", i,j)
        x, y = np.meshgrid(np.arange(0, l1, 1),np.arange(0, l2, 1))
        z = Alpha[(i)*l+j].reshape((l2, l1))
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
        im = axarr[i,j].contourf(x, y, z, 50, cmap=plt.cm.rainbow, levels=levels, vmax=z_max, vmin=z_min)

f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.3, hspace=0.3)
cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=z_min, vmax=z_max))
sm._A = []
plt.colorbar(sm, cax=cb_ax)
plt.suptitle("Modélisation de l'APS pour {} images de tailles {}x{}".format(N, l1, l2))

plt.show()