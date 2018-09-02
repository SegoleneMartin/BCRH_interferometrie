"""Ce code propose l'évaluation de la qualité de l'extraction des différences de phases à partir des images.
(SECTION 1) Presentation de plusieurs méthodes d'extraction,
(SECTION 2) A N fixé, BCR sur chacun des \Psi_n ainsi que la variance de leur estimation,
(SECTION 3) Plot pour différents L de la borne + variance estimation par methode itérative pour chaque \Psi_n 
(SECTION 4) Evolution variance et biais sur chacun des \Psi_n en fonction de L pour la méthode itérative """

import os, sys
from inspect import getsourcefile
from os.path import abspath, dirname

dos_1 = '/Users/segolenemartin/Documents/ENS/M1/Stage/Codes'  # Dossier dans lequel se trouve le fichier Bibliotheque_fonctions
os.chdir(dos_1)
import Bibliotheque_fonctions as bib

### 

os.chdir('/Users/segolenemartin/Documents/ENS/M1/Stage/Codes/BCRH_Guarnieri_Tebaldini/')  # dossier dans lequel se trouve le script et dans lequel on souhaite enregister les figures

# plt.savefig('variance_phase_L=32.eps', format='eps', bbox_inches='tight', dpi=1200)

import numpy as np
import pylab as plt
from numpy import pi, sin, sqrt
from numpy.random import rand, multivariate_normal, normal
from scipy.optimize import minimize, brute
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from numpy.linalg import inv, pinv
from joblib import Parallel, delayed
import time

# PARAMETRES            
lamb = 56           # longeur d'onde
g_0 = 1           # coef de correlation initial
r = 0.99
sigma = 0.1         # variance de alpha
sigma_b = 0.01      # variance bruit
dt = 12             # interval de temps en jours
N_exp = 100         # nbr de simulations
K = 100             # nombre d'itérations de l'algorithme

f = lambda x: 4*pi*dt*x/lamb           # fonction dérivée de phi

def g_temp(x):                          # fonction de correlation temporelle
    if x != 0:
        return g_0*r**(np.abs(x)*dt)
    else:
        return g_0*r**(np.abs(x)*dt)+(1-g_0)
g_temp = np.vectorize(g_temp)

### SECTION 1: Test des différentes méthodes d'extraction de phase

""" On teste ici 3 méthodes d'extraction des différences de phase:
 - La methode "moyennage" qui correspond à prendre la moyenne sur L --> phi_moy
 - La methode consistant à minimiser la forme quadratique avec le solver python (BFGS) --> phi_solv
 - La methode itérative décrite dans le rapport --> phi_exp.
 
 On remarque que les deux dernières méthodes donnent le même resultats mais le solver python est beaucoup plus lent. On compare les résultats aux différences de phase réelles."""
 
N = 20
L = 10
K = 1000

liste_MV = []

for tire in range(20):
    
    # Construction des données réelles 
    V = sigma**2*(np.eye(N-1)+1)
    theta_0 = 5/365
    omega_0 = multivariate_normal(np.zeros(N-1), V)
    omega_0 = np.concatenate((omega_0, np.zeros(1)))
    a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
    G = g_temp(b-a)
    G_inv = np.linalg.inv(G)    
    phi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]  # les vraies différences de phase
    theta = f(np.arange(0,N-1)-(N-1))
    theta.reshape(N-1,1)
    X = 2*L*(G*G_inv-np.eye(N))[:-1,:-1]
    
    # Construction des données experimentales
    C = bib.Cov_y(theta_0, omega_0, G, f, L)
    y = bib.GeneVectGaussien(C)[1]  
    
    psi_initial = np.random.rand(N)*2*pi
    
    # METHODE MOYENNAGE
    phi_moy = []
    for n in range(0, N-1):
        phi_moy.append(np.angle(1/L*np.sum(y[n*L:n*L+L]*np.conj(y[(N-1)*L:]))))
    
    # METHODE BFGS
    """
    R = np.zeros((N,N),dtype=complex)
    for l in range(L):
        for n in range(N):
            for m in range(N):
                R[n,m] += y[n*L+l]*np.conj(y[m*L+l])
    R = R/L
    
    t_1 = time.clock()   
    sol = minimize(bib.fonction_minimisee, np.zeros(N-1) ,  args=(y, G_inv, N, L),method='BFGS')
    t_1 = time.clock() - t_1
    print("temps BFGS ", t_1)  
    phi_solv = sol.x  
    """
    # METHODE ITERATIVE
    t_2 = time.clock()    
    phi_exp = bib.extraction_phase_norme(y,psi_initial, K, G_inv, N, L)
    t_2 = time.clock() - t_2 
    #print("temps Iterative ", t_2)
    
    # AFFICHAGE
    print("omega ", omega_0[:5])
    print("phi_reel ", phi_reel[:5])
    # print("phi_moy ", phi_moy[-5:])
    # print("phi_solv ", phi_solv[-5:])
    print("phi_exp ", phi_exp[:5])
    print("EQM", np.sum((phi_reel-phi_exp)**2)/(N-1))
    
    
    """Construction de la matrice pour le MV"""
    Mat1 = 1/(np.transpose(theta).dot(inv(inv(X)+V)).dot(theta))*((np.transpose(theta)).dot(inv(inv(X)+V))) # MVMAP
    Mat2 = 1/(np.transpose(theta).dot(X).dot(theta))*((np.transpose(theta)).dot(X)) # MV
    
    est_theta_1 = Mat1.dot(phi_exp)
    est_theta_2 = Mat2.dot(phi_exp)
    
    print(" ")
    print("theta_0", theta_0)
    print("MVMAP ", abs(est_theta_1-theta_0))
    print("MV ", abs(est_theta_2-theta_0))

    liste_MV.append(abs(est_theta_2-theta_0))

print("Moyenne MV ", np.sum(liste_MV)/len(liste_MV))

###

liste_N = np.concatenate(([2,3,4, 6, 8], np.arange(10, 20, 3)))
K = 100
L = 6
N_exp = 10000
theta_0 = 5/365

liste_norme = []
for N in liste_N:
    print('N', N)
    
    norme_exp = []
    a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
    G = g_temp(b-a)
    G_inv = np.linalg.inv(G)
    V = sigma**2*(np.eye(N-1)+1)

    for exp in range(N_exp):
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        C = bib.Cov_y(theta_0, omega_0, G, f, L)
        y = bib.GeneVectGaussien(C)[1]
        
        norme = bib.extraction_phase_norme(y, K, G_inv, N, L)
        norme_exp.append(norme)
    liste_norme.append(np.sum(np.array(norme_exp))/N_exp)
    
plt.figure(1)
plt.plot(liste_N, liste_norme, label="erreur de convergence", color='orange', marker='x')
plt.legend()
plt.xlabel('N')
plt.title("K={}, L={}".format(K, L))
plt.show()

### ERREUR DE CONVERGENCE
liste_K = np.arange(10, 250, 10)
liste_N = [4, 10, 15]
liste_L = [4, 10]

N_exp = 300
theta_0 = 5/365
 
fig, axes = plt.subplots(len(liste_N), len(liste_L))

for i, N in enumerate(liste_N):
    print("N", N)
    for j, L in enumerate(liste_L):
        print("L", L)
        
        liste_norme = []

        for K in liste_K:
            print('K', K)
            
            norme_exp = []
            a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
            G = g_temp(b-a)
            G_inv = np.linalg.inv(G)
            V = sigma**2*(np.eye(N-1)+1)
        
            for exp in range(N_exp):
                omega_0 = multivariate_normal(np.zeros(N-1), V)
                omega_0 = np.concatenate((omega_0, np.zeros(1)))
                C = bib.Cov_y(theta_0, omega_0, G, f, L)
                y = bib.GeneVectGaussien(C)[1]
                psi_initial = np.random.rand(N)*2*pi
                
                norme = bib.extraction_phase_norme(y, K, G_inv, N, L)
                norme_exp.append(norme)
                
            liste_norme.append(np.sum(np.array(norme_exp))/N_exp)
        
        axes[i,j].plot(liste_K, liste_norme)
        axes[i,j].set_xlabel('K')
        axes[i,j].set_title("N = {}, L = {}".format(N, L))
        

plt.suptitle("Erreur de convergence")    
fig.subplots_adjust(wspace=0.4, hspace=0.9)
plt.show()


### SECTION 2: comparaison methode par moyennage et methode de minimisation itérative

"""
abscisse : n = numero de la phase 
ordonnée: variance de l'estimation via différentes méthodes, et BCR sur phi[n] 
"""

N = 10  # N est fixé
L = 10
N_exp = 300

"""données réelles"""
theta_0 = 5/365
a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
G = g_temp(b-a)
G_inv = np.linalg.inv(G)
V = sigma**2*(np.eye(N-1)+1)

"""Borne"""
X = 2*L*(G[:-1,:-1]*G_inv[:-1,:-1]-np.eye(N-1))
X_inv = inv(X)
borne = []
for n in range(N-1):
    borne.append(np.sqrt(X_inv[n, n]))

""" Ecart type par moyennage sur L, et par methode itérative"""
liste_phi_moy = []
liste_phi_min = []

for exp in range(N_exp): 

    omega_0 = multivariate_normal(np.zeros(N-1), V)
    omega_0 = np.concatenate((omega_0, np.zeros(1)))
    phi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
    
    # données simulées
    C = bib.Cov_y(theta_0, omega_0, G, f, L)
    y = bib.GeneVectGaussien(C)[1]
    
    # METHODE MOYENNAGE
    phi_moy = []
    for n in range(0, N-1):
        phi_moy.append(np.angle(1/L*np.sum(y[n*L:n*L+L]*np.conj(y[(N-1)*L:]))))
    liste_phi_moy.append(phi_moy-phi_reel)
    
    # METHODE ITERATIVE
    phi_exp = bib.extraction_phase(y, K, G_inv, N, L)
    liste_phi_min.append(phi_exp-phi_reel)
    
var_phi_moy = np.sqrt(np.sum((np.array(liste_phi_moy))**2, axis=0)/N_exp)
var_phi_min = np.sqrt(np.sum((np.array(liste_phi_min))**2, axis=0)/N_exp)

# PLOT
plt.figure(1)
plt.plot(np.arange(0, N-1), borne, label="BCR ", color='orange', marker='x')
plt.plot(np.arange(0, N-1), var_phi_moy, label="moyennage sur L ", color='red', marker='x')
plt.plot(np.arange(0, N-1), var_phi_min, label="methode itérative ", color='blue', marker='x')
plt.legend()
axes = plt.gca()
axes.xaxis.set_ticks(range(N-1))
plt.xlabel('n')
plt.title("N={}, L={}".format(N, L))
plt.show()


### SECTION 3: plot pour différents L de la borne + variance estimation par methode itérative

N = 6 # N est fixé
liste_L = [2, 4, 16]
N_exp = 100

"""Données réelles"""
theta_0 = 5/365
a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
G = g_temp(b-a)
G_inv = np.linalg.inv(G)
V = sigma**2*(np.eye(N-1)+1)
 
"""Calcul variance et borne pour L dans liste_L"""
for i, L in enumerate(liste_L):
    
    print("L", L)
    
    plt.figure(i)
    
    # METHODE ITERATIVE 
    liste_phi_min = []
    for exp in range(N_exp): 
        
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        phi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
        C = bib.Cov_y(theta_0, omega_0, G, f, L)
        y = bib.GeneVectGaussien(C)[1]
        phi_exp = bib.extraction_phase(y, K, G_inv, N, L)
        liste_phi_min.append(phi_exp-phi_reel)
        
    print("Biais sur psi", np.sum(np.array(liste_phi_min), axis=0)/N_exp)
    
    var_phi_min = np.sqrt(np.sum((np.array(liste_phi_min))**2, axis=0)/N_exp)
    plt.plot(np.arange(0, N-1), var_phi_min, label='ecart-type phase',color=(i/len(liste_L), 0, 1-i/len(liste_L)), marker='o')

    # Borne
    borne = []
    X = 2*L*(G[:-1,:-1]*G_inv[:-1,:-1]-np.eye(N-1))
    X_inv = inv(X)
    for n in range(N-1):
        borne.append(np.sqrt(X_inv[n,n]))
    plt.plot(np.arange(0, N-1), np.array(borne), color=(i/len(liste_L), 0, 1-i/len(liste_L)),label='BCR', marker='x')

    plt.legend()
    axes = plt.gca()
    axes.xaxis.set_ticks(range(N-1))
    plt.xlabel('n')
    plt.title("L={}".format(L))
    
plt.show()


### SECTION 4: evolution variance et biais en fonction de L pour la méthode itérative BFGS 

N = 12 # N est fixé
liste_L = np.concatenate(([1,2,3,4], np.arange(5, 20, 3)))
N_exp = 4000
K = 200

"""Données réelles"""
theta_0 = 5/365
a, b = np.meshgrid(np.arange(0,N),np.arange(0,N))
G = g_temp(b-a)
G_inv = np.linalg.inv(G)
V = sigma**2*(np.eye(N-1)+1)
 
var_phi_min = []
biais_phi_min = []
borne = []

"""Calcul variance et borne en fonction de L"""
for i, L in enumerate(liste_L):
    print("L", L)
    
    # BFGS 
    liste_phi_min = []
    for exp in range(N_exp): 
        
        omega_0 = multivariate_normal(np.zeros(N-1), V)
        omega_0 = np.concatenate((omega_0, np.zeros(1)))
        phi_reel = f(np.arange(0,N-1)-(N-1))*theta_0 + omega_0[:-1]
        C = bib.Cov_y(theta_0, omega_0, G, f, L)
        y = bib.GeneVectGaussien(C)[1]
        phi_exp = bib.extraction_phase(y, K, G_inv, N, L)
        liste_phi_min.append(phi_exp-phi_reel)
        
    print("Biais sur psi", np.sum(np.array(liste_phi_min), axis=0)/N_exp)
    
    var_phi_min.append(np.sqrt(np.sum((np.array(liste_phi_min))**2, axis=0)/N_exp))
    biais_phi_min.append(np.abs(np.sum(np.array(liste_phi_min), axis=0)/N_exp))

    # Borne
    X = 2*L*(G[:-1,:-1]*G_inv[:-1,:-1]-np.eye(N-1))
    X_inv = inv(X)
    borne.append(np.sqrt(np.diag(X_inv)))
  
## PLOT

fig1, axarr1 = plt.subplots(2, 2)
axarr1[0,0].loglog(liste_L, np.array(var_phi_min)[:,0], color='blue', label='ecart type')
axarr1[0,0].loglog(liste_L, np.array(borne)[:,0], color='orange', label='borne')
axarr1[0,0].set_title("Psi_{}".format(0))
axarr1[0,0].legend()

axarr1[0,1].loglog(liste_L, np.array(var_phi_min)[:,int(N/4)], color='blue')
axarr1[0,1].loglog(liste_L, np.array(borne)[:,int(N/4)], color='orange')
axarr1[0,1].set_title("Psi_{}".format(int(N/4)))

axarr1[1,0].loglog(liste_L, np.array(var_phi_min)[:,int(N/2)], color='blue')
axarr1[1,0].loglog(liste_L, np.array(borne)[:,int(N/2)], color='orange')
axarr1[1,0].set_title("Psi_{}".format(int(N/2)))

axarr1[1,1].loglog(liste_L, np.array(var_phi_min)[:,int(3*N/4)], color='blue')
axarr1[1,1].loglog(liste_L, np.array(borne)[:,int(3*N/4)], color='orange')
axarr1[1,1].set_title("Psi_{}".format(int(3*N/4)))

fig1.subplots_adjust(wspace=0.4, hspace=0.4)
plt.suptitle('En fonction de L pour N={}'.format(N))


fig2, axarr2 = plt.subplots(2, 2)
axarr2[0,0].plot(liste_L, np.array(biais_phi_min)[:,0], color='blue', label='biais')
axarr2[0,0].set_title("Psi_{}".format(0))
axarr2[0,0].legend()

axarr2[0,1].plot(liste_L, np.array(biais_phi_min)[:,int(N/4)], color='blue')
axarr2[0,1].set_title("Psi_{}".format(int(N/4)))

axarr2[1,0].plot(liste_L, np.array(biais_phi_min)[:,int(N/2)], color='blue')
axarr2[1,0].set_title("Psi_{}".format(int(N/2)))

axarr2[1,1].plot(liste_L, np.array(biais_phi_min)[:,int(3*N/4)], color='blue')
axarr2[1,1].set_title("Psi_{}".format(int(3*N/4)))

fig2.subplots_adjust(wspace=0.4, hspace=0.4)
plt.suptitle('En fonction de L pour N={}, K={}, N_exp={}'.format(N, K, N_exp))

plt.show()

#plt.savefig('variance_phase_subplot.eps', format='eps', bbox_inches='tight', dpi=1200)
