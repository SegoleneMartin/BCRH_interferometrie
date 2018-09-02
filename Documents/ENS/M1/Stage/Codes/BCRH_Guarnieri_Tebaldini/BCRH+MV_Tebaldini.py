"""Ce code reconstitue toute les étapes du papier de Tebaldini: 
1) calcul de la borne, 
2) simulation des images
3) extraction des phases
4) calcul du MV sur theta

(SECTION 1): Calcul borne et Variance estimateur MV et MVMAP en fonction de N. A chaque fois on enregistre la figure sous le nom 'MV_N_L=*' et on stocke les liste dans le fichier npy du même nom.
(SECTION 2): Calcul borne et Variance estimateur MV et MVMAP en fonction de L. A chaque fois on enregistre la figure sous le nom 'MV_L_N=*' et on stocke les liste dans le fichier npy du même nom.
(SECTION 3): MV (ou MVMAP au choix) en générant directement les phases, en fonction de N.
(SECTION 4): MV (ou MVMAP au choix) en générant directement les phases, en fonction de L.
"""

### Bibliothèque personnelle 

import os, sys
from inspect import getsourcefile
from os.path import abspath, dirname

dos_1 = '/Users/segolenemartin/Documents/ENS/M1/Stage/Codes'  # Dossier dans lequel se trouve le fichier Bibliotheque_fonctions
os.chdir(dos_1)
import Bibliotheque_fonctions as bib

### Bibliothèque python 
import numpy as np
import pylab as plt
from numpy import pi, sin, sqrt
from numpy.random import rand, multivariate_normal, normal
from numpy.linalg import inv
from numpy import transpose as t
from scipy.optimize import minimize, brute
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from joblib import Parallel, delayed
import time

os.chdir('/Users/segolenemartin/Documents/ENS/M1/Stage/Codes/BCRH_Guarnieri_Tebaldini/')  
# dossier dans lequel se trouve le script et dans lequel on souhaite enregister les figures

### Paramètres
          
lamb = 56       # longeur d'onde
g_0 = 0.7       # coef de correlation initial
r = 0.975       # intervient dans correlation
sigma = 1       # variance de alpha
dt = 12         # interval de temps en jours
N_exp = 400
K = 30          # nbr d'iteration de Newton

f = lambda x: 4*pi*dt*x/lamb           # fonction dérivée de phi

def g_temp(x):                           # fonction de correlation temporelle
    if x != 0:
        return g_0*r**(np.abs(x)*dt)
    else:
        return g_0*r**(np.abs(x)*dt)+(1-g_0)
g_temp = np.vectorize(g_temp)
    
    
### CALCUL BORNE & VARIANCE ESTIMATEUR MV (EN FONCTION DE N)
 
L = 10
N_exp = 30000

"""Choix aléatoire d'un theta_0 dans [theta_min, theta_max] pour la simulation"""
theta_min = 3/365
theta_max = 12/365
theta_0 = rand()*(theta_max-theta_min) + theta_min

nbr_image = []
borne= []
var_est_theta_1 = []
var_est_theta_2 = []
biais_est_theta = []

liste_N = np.concatenate(([2,3,4, 6, 8], np.arange(10, 20, 3)))

for N in liste_N:
    
    print("N", N)
    
    # CALCUL BCRH
    """calcul matrice G (correlation)"""
    a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
    G = g_temp(b-a)
    G_inv = inv(G)
    """calcul de theta et V"""
    theta = f(np.arange(0,N-1)-(N-1))
    theta.reshape(N-1,1)
    V = sigma**2*(np.eye(N-1)+1)
    X = 2*L*(G*G_inv-np.eye(N))[:-1,:-1]
    borne.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
    
    # SIMULATION
    """Construction de la matrice pour le MV"""
    Mat1 = 1/(np.transpose(theta).dot(inv(inv(X)+V)).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)+V))) # MVMAP
    Mat2 = 1/(np.transpose(theta).dot(X).dot(theta))*((np.transpose(theta)).dot(X)) # MV
    
    # def action(i):
    #     """Tirage de omega_0 """
    #     omega_0 = multivariate_normal(np.zeros(N-1), V)
    #     omega_0 = np.concatenate((omega_0, np.zeros(1)))
    #     """construction des vraies différences de phase"""
    #     psi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
    #     """Construction de C(theta_0, omega_0)"""
    #     C = bib.Cov_y(theta_0, omega_0, G, f, L)    
    #         
    #     """tirage de y selon normal(0,C) via Cholesky"""
    #     y = bib.GeneVectGaussien(C)[1]

   #       # ETAPE 1 DU MV: EXTRACTION DES DIFFERENCES DE PHASE
    #     psi_exp = bib.extraction_phase(y, K, G_inv, N, L)
    #     
    #     # ETAPE 2 DU MV: MV SUR PSI_EXP
    #     return(Mat1.dot(psi_exp), Mat2.dot(psi_exp))
    #     
    # var = Parallel(n_jobs=-1)(map(delayed(action), range(N_exp)))
    # est_theta_exp_1, est_theta_exp_2 = zip(*var)
    
    est_theta_exp_1 = []
    est_theta_exp_2 = []
    
    for exp in range(N_exp):
        
        """Tirage de omega_0 """
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        """construction des vraies différences de phase"""
        psi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
        """Construction de C(theta_0, omega_0)"""
        C = bib.Cov_y(theta_0, omega_0, G, f, L)    
            
        """tirage de y selon normal(0,C) via Cholesky"""
        y = bib.GeneVectGaussien(C)[1]

        # ETAPE 1 DU MV: EXTRACTION DES DIFFERENCES DE PHASE
        psi_exp = bib.extraction_phase(y, K, G_inv, N, L)
        
        # ETAPE 2 DU MV: MV SUR PSI_EXP
        est_theta_exp_1.append(Mat1.dot(psi_exp))
        est_theta_exp_2.append(Mat2.dot(psi_exp))
    
    """ Calcul variance et moyenne de theta sur les experiences"""
    var_est_theta_1.append(sqrt(np.sum((np.array(est_theta_exp_1)-theta_0)**2)/N_exp))
    var_est_theta_2.append(sqrt(np.sum((np.array(est_theta_exp_2)-theta_0)**2)/N_exp))
    
    biais_est_theta.append((np.sum(abs(np.array(est_theta_exp_2)-theta_0))/N_exp))
    
    # print("biais sur theta 1 : ",np.sum(np.array(est_theta_exp_1)-theta_0)/N_exp)
    print("biais sur theta 2 : ",np.sum(abs(np.array(est_theta_exp_2)-theta_0))/N_exp)
    
## figure variance

plt.figure(1)
plt.loglog(liste_N, var_est_theta_1, label='écart-type sur v (mm/jour) (MVMAP)', color='steelblue', linestyle='--' )
plt.loglog(liste_N, var_est_theta_2, label='écart-type sur v (mm/jour) (MV)', color='blue', linestyle='--' )
plt.loglog(liste_N, borne, label="BCRH ", color='orange')
plt.xlabel("nombre d'images N")
plt.title("MV en générant les images, L={}, sigma={}, N_exp={}".format(L, sigma, N_exp))
plt.legend()
plt.show()

plt.savefig('MV_N_L=%s.eps'% L, format='eps', bbox_inches='tight', dpi=1200)
np.save('MV_N_L=%s.npy'% L, np.array([liste_N, var_est_theta_1, var_est_theta_2, borne, biais_est_theta]))

## figure biais

plt.figure(2)
plt.loglog(liste_N, biais_est_theta, label='biais sur v (mm/jour)', color='steelblue', linestyle='--' )
plt.loglog(liste_N, theta_0*np.ones(len(liste_N)), label="theta_0", color='red')
plt.xlabel("nombre d'images N")
plt.title("Biais sur MV en générant les images, L={}, sigma={}, N_exp={}".format(L, sigma, N_exp))
plt.legend()
plt.show()

plt.savefig('biais_MV_N_L=%s.eps'% L, format='eps', bbox_inches='tight', dpi=1200)

### CALCUL BORNE & VARIANCE ESTIMATEUR MV (EN FONCTION DE L)
 
N = 10
N_exp = 30000

"""Choix aléatoire d'un theta_0 dans [theta_min, theta_max] pour la simulation"""
theta_min = 3/365
theta_max = 12/365
theta_0 = rand()*(theta_max-theta_min) + theta_min

"""calcul matrice G (correlation)"""
a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
G = g_temp(b-a)
G_inv = inv(G)
"""calcul de theta et V"""
theta = f(np.arange(0,N-1)-(N-1))
theta.reshape(N-1,1)
V = sigma**2*(np.eye(N-1)+1)
U = (G*G_inv-np.eye(N))[:-1,:-1]

liste_L = np.concatenate(([2,3,4, 6, 8], np.arange(10, 13, 4)))
borne= []
var_est_theta_1 = []
var_est_theta_2 = []
biais_est_theta = []

for L in liste_L:
    
    print("L", L)
    
    # CALCUL BCRH
    X = 2*L*U
    borne.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
    
    # SIMULATION
    """Construction de la matrice pour le MV"""
    Mat1 = 1/(np.transpose(theta).dot(inv(inv(X)+V)).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)+V))) # MVMAP
    Mat2 = 1/(np.transpose(theta).dot(X).dot(theta))*((np.transpose(theta)).dot(X))
    est_theta_exp = [] # MV
    
    # def action(i):
    #     print("action ", i)
    #     """Tirage de omega_0 """
    #     omega_0 = multivariate_normal(np.zeros(N-1), V)
    #     omega_0 = np.concatenate((omega_0, np.zeros(1)))
    #     """construction des vraies différences de phase"""
    #     psi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
    #     """Construction de C(theta_0, omega_0)"""
    #     C = bib.Cov_y(theta_0, omega_0, G, f, L)    
    #         
    #     """tirage de y selon normal(0,C) via Cholesky"""
    #     y = bib.GeneVectGaussien(C)[1]

    #     # ETAPE 1 DU MV: EXTRACTION DES DIFFERENCES DE PHASE
    #     psi_exp = bib.extraction_phase(y, K, G_inv, N, L)
    #     
    #     # ETAPE 2 DU MV: MV SUR PSI_EXP
    #     return(Mat1.dot(psi_exp), Mat2.dot(psi_exp))
    #     
    # var = Parallel(n_jobs=-1)(map(delayed(action), range(N_exp)))
    # est_theta_exp_1, est_theta_exp_2 = zip(*var)
        
    est_theta_exp_1 = []
    est_theta_exp_2 = []
    
    for exp in range(N_exp):
        
        """Tirage de omega_0 """
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        """construction des vraies différences de phase"""
        psi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
        """Construction de C(theta_0, omega_0)"""
        C = bib.Cov_y(theta_0, omega_0, G, f, L)    
            
        """tirage de y selon normal(0,C) via Cholesky"""
        y = bib.GeneVectGaussien(C)[1]

        # ETAPE 1 DU MV: EXTRACTION DES DIFFERENCES DE PHASE
        psi_exp = bib.extraction_phase(y, K, G_inv, N, L)
        
        # ETAPE 2 DU MV: MV SUR PSI_EXP
        est_theta_exp_1.append(Mat1.dot(psi_exp))
        est_theta_exp_2.append(Mat2.dot(psi_exp))
    
    """ Calcul variance et moyenne de theta sur les experiences"""
    var_est_theta_1.append(sqrt(np.sum((np.array(est_theta_exp_1)-theta_0)**2)/N_exp))
    var_est_theta_2.append(sqrt(np.sum((np.array(est_theta_exp_2)-theta_0)**2)/N_exp))
    
    biais_est_theta.append(np.sum(abs(np.array(est_theta_exp_2)-theta_0))/N_exp)
    print("biais sur theta 2 : ",np.sum(abs(np.array(est_theta_exp_2)-theta_0))/N_exp)
    
## figure variance

plt.figure(3)
plt.loglog(liste_L, var_est_theta_1, label='écart-type sur v (mm/jour) (avec V)', color='steelblue', linestyle='--' )
plt.loglog(liste_L, var_est_theta_2, label='écart-type sur v (mm/jour) (sans V)', color='blue', linestyle='--' )
plt.loglog(liste_L, borne, label="BCRH ", color='orange')
plt.xlabel("nombre de pixels L")
plt.title("MV en générant les images, N={}, sigma={}, N_exp={}".format(N, sigma, N_exp))
plt.legend()
plt.show()

# plt.savefig('MV_L_N=%s.eps'% N, format='eps', bbox_inches='tight', dpi=1200)
# np.save('MV_L_N=%s.npy'% N, np.array([liste_L, var_est_theta_1, var_est_theta_2, borne, biais_est_theta]))

## figure biais

plt.figure(4)
plt.loglog(liste_L, biais_est_theta, label='biais sur v (mm/jour)', color='steelblue', linestyle='--' )
plt.loglog(liste_L, theta_0*np.ones(len(liste_L)), label="theta_0", color='orange')
plt.xlabel("nombre de pixels L")
plt.title("Biais sur MV en générant les images, N={}, sigma={}, N_exp={}".format(N, sigma, N_exp))
plt.legend()
plt.show()


### SECTION 3: MV EN GENERANT DIRECTEMENT LES PHASES (EN FONCTION DE N)
"""Ici on génére directement les phase à partir de theta_0, omega_0, et un bruit corrélé. Pour choisir le MVMAP, il suffit de decommenter la ligne où la matrice Mat est définie et commenter la précedente."""


"""Choix aléatoire d'un theta_0 dans [theta_min, theta_max] pour la simulation"""
theta_min = 3/365
theta_max = 12/365
theta_0 = rand()*(theta_max-theta_min) + theta_min
 
"""Choix de L"""
N_exp = 5000
L = 8
nbr_image = []
borne= []
var_est_theta = []

for N in range(120, 140, 5):
    
    nbr_image.append(N)
    print("N", N)
    
    # CALCUL BCRH
    """calcul matrice G (correlation)"""
    a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
    G = g_temp(b-a)
    G_inv = inv(G)
    """calcul de theta, X et V"""
    theta = f(np.arange(0,N-1)-(N-1))
    theta.reshape(N-1,1)
    X = 2*L*(G*G_inv-np.eye(N))[:-1,:-1]
    V = sigma**2*(np.eye(N-1)+1)
    # A = np.concatenate((np.zeros(N-1).reshape(N-1,1), np.eye(N-1)), axis=1)
    borne.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
    
    # SIMULATION
    est_psi_exp = []
    est_theta_exp = []
        
    for exp in range(N_exp):
        
        # print("exp", exp)
        
        """Tirage de omega_0 """
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        """ Tirage du bruit"""
        b = multivariate_normal(np.zeros(N-1), inv(X)+V)
        """construction des vraies différences de phase et de celles avec du bruit"""
        psi_reel = theta*theta_0 + omega_0[:-1]
        psi_exp = theta*theta_0 + b
        """Construction de la matrice pour le MV"""
        Mat = 1/(np.transpose(theta).dot(inv(inv(X)+V)).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)+V)))    
        # Mat = 1/(np.transpose(theta).dot(inv(inv(X))).dot(theta))*((np.transpose(theta)).dot(inv(inv(X))))       
        
        # MV SUR PSI_EXP
        est_theta_exp.append(Mat.dot(psi_exp))
        
    """ Calcul variance et moyenne de theta sur les experiences"""
    var_est_theta.append(sqrt(np.sum((np.array(est_theta_exp)-theta_0)**2)/N_exp))
    # print("biais sur theta : ",np.sum(np.array(est_theta_exp)-theta_0)/N_exp)

plt.figure(5)
plt.loglog(nbr_image, var_est_theta, label='écart-type sur v (mm/jour)', color='steelblue' )
plt.loglog(nbr_image, borne, label="BCRH ", color='orange')
plt.xlabel("nombre d'images N")
plt.title("MVMAP en générant directement les phases, L={}, sigma={}, N_exp={}".format(L, sigma, N_exp))
plt.legend()
plt.show()




### SECTION 4: MV EN GENERANT DIRECTEMENT LES PHASES (EN FONCTION DE L)
"""Ici on génére directement les phase à partir de theta_0, omega_0, et un bruit corrélé. Pour choisir le MVMAP, il suffit de decommenter la ligne où la matrice Mat est définie et commenter la précedente."""


"""Choix de N"""
N = 10
N_exp = 30000

"""Choix aléatoire d'un theta_0 dans [theta_min, theta_max] pour la simulation"""
theta_min = 3/365
theta_max = 12/365
theta_0 = rand()*(theta_max-theta_min) + theta_min

"""calcul matrice G (correlation)"""
a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
G = g_temp(b-a)
G_inv = inv(G)

"""calcul de theta et V"""
theta = f(np.arange(0,N-1)-(N-1))
theta.reshape(N-1,1)
V = sigma**2*(np.eye(N-1)+1)
    
liste_L = []
borne= []
var_est_theta = []

for L in range(2, 40, 3):
    
    liste_L.append(L)
    print("L", L)
    
    # CALCUL BCRH
    """calcul de theta, X et V"""
    X = 2*L*(G*G_inv-np.eye(N))[:-1,:-1]
    borne.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
    
    # SIMULATION
    est_psi_exp = []
    est_theta_exp = []
    """Construction de la matrice pour le MV"""
    # Mat = 1/(np.transpose(theta).dot(inv(inv(X)+V)).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)+V))) # MVMAP
    Mat = 1/(np.transpose(theta).dot(inv(inv(X))).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)))) #MV
    for exp in range(N_exp):
        
        """Tirage de omega_0 """
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        """ Tirage du bruit"""
        b = multivariate_normal(np.zeros(N-1), inv(X)+V)
        """construction des vraies différences de phase et de celles avec du bruit"""
        psi_reel = theta*theta_0
        psi_exp = theta*theta_0 + b
        
        # MV SUR PSI_EXP
        est_theta_exp.append(Mat.dot(psi_exp))
        
    """ Calcul variance et moyenne de theta sur les experiences"""
    var_est_theta.append(sqrt(np.sum((np.array(est_theta_exp)-theta_0)**2)/N_exp))
    print("biais sur theta : ",np.sum(abs(np.array(est_theta_exp)-theta_0))/N_exp)

plt.figure(6)
plt.loglog(liste_L, var_est_theta, label='écart-type sur v (mm/jour)', color='steelblue' )
plt.loglog(liste_L, borne, label="BCRH ", color='orange')
plt.xlabel("nombre de pixels L")
plt.title("MV en générant directement les phases, N={}, sigma={}, N_exp={}".format(N, sigma, N_exp))
plt.legend()
plt.show()


