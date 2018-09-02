"""Ce code propose plusieurs améliorations de la BCRH donnée par l'article "Hybrid Cramer-Rao Bounds for Crustal Displacement Field Estimators in SAR Interferométry"
SECTION 1: ajout d'une correlation spatiale dans les images
SECTION 2: ajout d'une variation lente de l'APS en plus de la corrélation spatiale"""

import os, sys
from inspect import getsourcefile
from os.path import abspath, dirname

os.chdir(dirname(abspath(getsourcefile(lambda:0))))
# plt.savefig('figure_1_amelioration.eps', format='eps', bbox_inches='tight', dpi=1200)

###

import numpy as np
import pylab as plt
from numpy import pi, sin, sqrt
from numpy.random import rand, multivariate_normal
from numpy.linalg import inv

# PARAMETRES            
lamb = 56           # longeur d'onde
g_0 = 0.7           # coef de correlation initial
r = 0.975
sigma = 1           # variance de alpha
sigma_b = 0.1
dt = 12             # interval de temps en jours
p = 0.00001          # module g_spat (correlation spatiale)

# FONCTIONS
f = lambda x: 4*pi*dt*x/lamb           # fonction dérivée de phi
f_inv = lambda x: 1/(4*pi*dt*x)*lamb   # inverse de la fonction f

l_1 = 1
l_2 = 1
L = l_1 * l_2

### SECTION 1: ajout d'une correlation spatiale dans les images


# fonctions pour construire la matrice G de correlation spatio-temporelle
def g_temp(x):                                  # fonction de correlation temporelle
    if x != 0:
        return g_0*r**(np.abs(x)*dt)
    else:
        return g_0*r**(np.abs(x)*dt)+(1-g_0)
g_temp = np.vectorize(g_temp)
        
g_spat = lambda dr, dx: np.exp(-p*(dr**2+dx**2))  # fonction de correlation spatiale

def sous_bloc(x,k):   # renvoie le bloc d'indice k de G[n,m]
    A = np.zeros((l_1,l_1))
    for i in range(l_1):
        for j in range(i+1):
            A[i,j] = g_spat(i-j,k)
            A[j,i] = g_spat(i-j,k)
    return g_temp(x)*A
            
def g(x):             # renvoie le bloc (n,m) de G
    A = np.ndarray((L,L))
    for i in range(l_2):
        for j in range(i+1):
            bloc = sous_bloc(x,i-j)
            A[i*l_1:i*l_1+l_1,j*l_1:j*l_1+l_1] = bloc
            A[j*l_1:j*l_1+l_1,i*l_1:i*l_1+l_1] = bloc
    return A

borne = []
borne_ameliore = []
nbr_image = []

for N in range(2, 30, 3):
    
    try:
        # BORNE AVEC CORRELATION SPATIALE
        # calcul matrice G (correlation spatio-temporelle)
        G_corr = np.ndarray((L*N, L*N))
        for n in range(N):
            for m in range(n+1):
                bloc = g(n-m)
                G_corr[n*L:n*L+L,m*L:m*L+L] = bloc
                G_corr[m*L:m*L+L,n*L:n*L+L] = bloc
        G_corr_inv = np.linalg.inv(G_corr)
        # calcul de theta, X et V
        theta = f(np.arange(0,N-1)-(N-1))
        theta.reshape(N-1,1)
        
        X_corr = np.ndarray((N-1,N-1))
        for n in range(N-1):
            for m in range(n+1):
                trace = np.trace(G_corr[n*L:n*L+L,m*L:m*L+L].dot(G_corr_inv[m*L:m*L+L,n*L:n*L+L]))
                if n != m:
                    X_corr[n,m] = 2*trace
                    X_corr[m,n] = 2*trace
                else :
                    X_corr[n,m] = 2*(trace-L)
        V = sigma*(np.eye(N-1)+1)
        
        # COMPARAISON AVEC BCRH INITIALE (SANS CORRELATION SPATIALE)
        # calcul matrice G (correlation)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        G_inv = np.linalg.inv(G)
        # calcul X
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
        # borne_ameliore.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X_corr)+V)).dot(theta))))
        borne_ameliore.append(sqrt(1/(theta.dot(X_corr).dot(theta)-theta.dot(X_corr).dot(inv(X_corr+inv(V))).dot(X_corr).dot(theta))))
        borne.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X)+V)).dot(theta))))
        nbr_image.append(N)
    
    except np.linalg.LinAlgError:
        print("Matrice non inversible")

# PLOT
plt.figure(1)

plt.loglog(nbr_image, borne, label='BCRH sans correlation spatiale (mm/jour)', linestyle='--' )
plt.loglog(nbr_image, borne_ameliore, label='BCRH avec correlation spatiale (mm/jour)',  linestyle=':' , linewidth=3)
plt.xlabel("Nombre d'images N")
# plt.title('L={}'.format(L))
plt.legend()
plt.show()


### SECTION 1: ajout d'une correlation spatiale dans les images (EN FONCTION DE L)


# fonctions pour construire la matrice G de correlation spatio-temporelle
def g_temp(x):                                  # fonction de correlation temporelle
    if x != 0:
        return g_0*r**(np.abs(x)*dt)
    else:
        return g_0*r**(np.abs(x)*dt)+(1-g_0)
g_temp = np.vectorize(g_temp)
        
g_spat = lambda dr, dx: np.exp(-p*(dr**2+dx**2))  # fonction de correlation spatiale

def sous_bloc(x,k):   # renvoie le bloc d'indice k de G[n,m]
    A = np.zeros((l_1,l_1))
    for i in range(l_1):
        for j in range(i+1):
            A[i,j] = g_spat(i-j,k)
            A[j,i] = g_spat(i-j,k)
    return g_temp(x)*A
            
def g(x):             # renvoie le bloc (n,m) de G
    A = np.ndarray((L,L))
    for i in range(l_2):
        for j in range(i+1):
            bloc = sous_bloc(x,i-j)
            A[i*l_1:i*l_1+l_1,j*l_1:j*l_1+l_1] = bloc
            A[j*l_1:j*l_1+l_1,i*l_1:i*l_1+l_1] = bloc
    return A


borne = []
borne_ameliore = []
liste_L = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
N = 5

for l_1 in liste_L:
    l_2 = l_1
    L = l_1*l_2
    
    try:
        # BORNE AVEC CORRELATION SPATIALE
        # calcul matrice G (correlation spatio-temporelle)
        G_corr = np.ndarray((L*N, L*N))
        for n in range(N):
            for m in range(n+1):
                bloc = g(n-m)
                G_corr[n*L:n*L+L,m*L:m*L+L] = bloc
                G_corr[m*L:m*L+L,n*L:n*L+L] = bloc
        G_corr_inv = np.linalg.inv(G_corr)
        # calcul de theta, X et V
        theta = f(np.arange(0,N-1)-(N-1))
        theta.reshape(N-1,1)
        
        X_corr = np.ndarray((N-1,N-1))
        for n in range(N-1):
            for m in range(n+1):
                trace = np.trace(G_corr[n*L:n*L+L,m*L:m*L+L].dot(G_corr_inv[m*L:m*L+L,n*L:n*L+L]))
                if n != m:
                    X_corr[n,m] = 2*trace
                    X_corr[m,n] = 2*trace
                else :
                    X_corr[n,m] = 2*(trace-L)
        V = sigma*(np.eye(N-1)+1)
        
        # COMPARAISON AVEC BCRH INITIALE (SANS CORRELATION SPATIALE)
        # calcul matrice G (correlation)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        G_inv = np.linalg.inv(G)
        # calcul X
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
        # borne_ameliore.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X_corr)+V)).dot(theta))))
        borne_ameliore.append(sqrt(1/(theta.dot(X_corr).dot(theta)-theta.dot(X_corr).dot(inv(X_corr+inv(V))).dot(X_corr).dot(theta))))
        borne.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X)+V)).dot(theta))))
        nbr_image.append(N)
    
    except np.linalg.LinAlgError:
        print("Matrice non inversible")

# PLOT
plt.figure(1)

plt.loglog(liste_L, borne, label='BCRH sans correlation spatiale (mm/jour)', linestyle='--' )
plt.loglog(liste_L, borne_ameliore, label='BCRH avec correlation spatiale (mm/jour)',  linestyle=':' , linewidth=3)
plt.xlabel("L")
# plt.title('L={}'.format(L))
plt.legend()
plt.show()



### SECTION 2: Ajout d'une variation lente dans l'APS BIS

p = 1          # module g_spat (correlation spatiale)
borne = []
borne_ameliore = []
nbr_image = []

for N in range(3, 40, 1):
    
    try:
        # METHODE AVEC CORRELATION SPATIALE ET VARIATION APS
        # calcul matrice G (correlation spatio-temporelle)
        G_corr = np.ndarray((L*N, L*N))
        for n in range(N):
            for m in range(n+1):
                bloc = g(n-m)
                G_corr[n*L:n*L+L,m*L:m*L+L] = bloc
                G_corr[m*L:m*L+L,n*L:n*L+L] = bloc
        G_corr_inv = np.linalg.inv(G_corr)
        # calcul de theta, X et V
        theta_corr = np.zeros((N-1)*L)
        for n in range(n-1):
            theta_corr[n*L:n*L+L] = f(n-(N-1))*np.ones(L)
        
        X_corr = np.ndarray(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            for m in range(n+1):
                for l in range(L): 
                    for k in range(l+1):
                        
                        trace = G_corr[n*L+l,m*L+k]*G_corr[n*L+l,m*L+k]
                        if n != m or (n == m and l != k):
                            X_corr[n*L+l, m*L+k] = 2*trace
                            X_corr[n*L+k, m*L+l] = 2*trace
                            X_corr[m*L+l, n*L+k] = 2*trace
                            X_corr[m*L+k, n*L+l] = 2*trace
                            
                        else :
                            terme = G_corr[n*L+l, n*L:n*L+L]*G_corr_inv[n*L+l, n*L:n*L+L]
                            terme[l] = 0
                            trace = -(N-1)*np.sum(G_corr[n*L+l, n*L:n*L+L]*G_corr_inv[n*L+l, n*L:n*L+L])-np.sum(terme)
                            X_corr[n*L+l, n*L+l] = 2*trace
                    
        V_alpha = np.ones(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            V_alpha[n*L:n*L+L, n*L:n*L+L] += np.ones((L,L))
        V_alpha = sigma**2*V_alpha
        V_bruit = np.ones(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            for m in range(N-1):
                V_bruit[n*L:n*L+L, m*L:m*L+L] += np.eye(L)
        V_bruit = sigma_b**2*V_bruit
        V_corr = V_alpha + V_bruit
        
        

        # COMPARAISON AVEC BCRH INITIALE (SANS CORRELATION SPATIALE)
        # calcul matrice G (correlation)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        G_inv = np.linalg.inv(G)
        # calcul X, V, theta
        theta = f(np.arange(0,N-1)-(N-1))
        theta.reshape(N-1,1)
        V = sigma*(np.eye(N-1)+1)
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
        
        borne_ameliore.append(sqrt(1/(theta_corr.dot(np.linalg.inv(np.linalg.inv(X_corr)+V_corr)).dot(theta_corr))))
        borne.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X)+V)).dot(theta))))
        nbr_image.append(N)
    
    except np.linalg.LinAlgError:
        print("Matrice non inversible")

# PLOT
plt.figure(2)
plt.loglog(nbr_image, borne_ameliore, label='BCRH avec variation APS (mm/jour)' )
plt.loglog(nbr_image, borne, label='BCRH sans variation APS (mm/jour)' )
plt.xlabel("nombre d'images N")
plt.legend()
plt.show()




### SECTION 3: Ajout d'une variation lente dans l'APS (EN FONCTION DE L)

p = 1          # module g_spat (correlation spatiale)
borne = []
borne_ameliore = []
liste_L = [1, 2, 3, 4, 5]
N = 5

for l_1 in liste_L:
    l_2 = l_1
    L = l_1*l_2
    
    try:
        # METHODE AVEC CORRELATION SPATIALE ET VARIATION APS
        # calcul matrice G (correlation spatio-temporelle)
        G_corr = np.ndarray((L*N, L*N))
        for n in range(N):
            for m in range(n+1):
                bloc = g(n-m)
                G_corr[n*L:n*L+L,m*L:m*L+L] = bloc
                G_corr[m*L:m*L+L,n*L:n*L+L] = bloc
        G_corr_inv = np.linalg.inv(G_corr)
        # calcul de theta, X et V
        theta_corr = np.zeros((N-1)*L)
        for n in range(n-1):
            theta_corr[n*L:n*L+L] = f(n-(N-1))*np.ones(L)
        
        X_corr = np.ndarray(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            for m in range(n+1):
                for l in range(L): 
                    for k in range(l+1):
                        
                        trace = G_corr[n*L+l,m*L+k]*G_corr[n*L+l,m*L+k]
                        if n != m or (n == m and l != k):
                            X_corr[n*L+l, m*L+k] = 2*trace
                            X_corr[n*L+k, m*L+l] = 2*trace
                            X_corr[m*L+l, n*L+k] = 2*trace
                            X_corr[m*L+k, n*L+l] = 2*trace
                            
                        else :
                            terme = G_corr[n*L+l, n*L:n*L+L]*G_corr_inv[n*L+l, n*L:n*L+L]
                            terme[l] = 0
                            trace = -(N-1)*np.sum(G_corr[n*L+l, n*L:n*L+L]*G_corr_inv[n*L+l, n*L:n*L+L])-np.sum(terme)
                            X_corr[n*L+l, n*L+l] = 2*trace
                    
        V_alpha = np.ones(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            V_alpha[n*L:n*L+L, n*L:n*L+L] += np.ones((L,L))
        V_alpha = sigma**2*V_alpha
        V_bruit = np.ones(((N-1)*L,(N-1)*L))
        for n in range(N-1):
            for m in range(N-1):
                V_bruit[n*L:n*L+L, m*L:m*L+L] += np.eye(L)
        V_bruit = sigma_b**2*V_bruit
        V_corr = V_alpha + V_bruit
        
        

        # COMPARAISON AVEC BCRH INITIALE (SANS CORRELATION SPATIALE)
        # calcul matrice G (correlation)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        G_inv = np.linalg.inv(G)
        # calcul X, V, theta
        theta = f(np.arange(0,N-1)-(N-1))
        theta.reshape(N-1,1)
        V = sigma*(np.eye(N-1)+1)
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
        
        borne_ameliore.append(sqrt(1/(theta_corr.dot(np.linalg.inv(np.linalg.inv(X_corr)+V_corr)).dot(theta_corr))))
        borne.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X)+V)).dot(theta))))
        nbr_image.append(N)
    
    except np.linalg.LinAlgError:
        print("Matrice non inversible")

# PLOT
plt.figure(2)
plt.loglog(liste_L, borne_ameliore, label='BCRH avec variation APS (mm/jour)' )
plt.loglog(liste_L, borne, label='BCRH sans variation APS (mm/jour)' )
plt.xlabel("L")
plt.legend()
plt.show()

