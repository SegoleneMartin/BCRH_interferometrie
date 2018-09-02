import numpy as np
from numpy.linalg import cholesky
from numpy.random import multivariate_normal
import cmath
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
  
    
    
    
    
def GeneVectGaussien(Cov):
    """
    Les paramètres d'entrée:
    Cov = Matrice de covariance des données, Cov doit etre de taille NxN, hermitienne définie positive;
    
    Les paramètres de sortie:
    y_r = vecteur gaussien réel de taille N, de moyenne nulle, de covariance Cov:
    y_c = vecteur gaussien circulaire (complexe) de taille N, de moyenne nulle, de covariance Cov;
    """
    
    N = Cov.shape[0]
    
    # Matrice de loi Gausienne, centrée et réduite:
    X = 1/np.sqrt(2)*(multivariate_normal(np.zeros(N), np.eye(N)) + 1j*multivariate_normal(np.zeros(N), np.eye(N)));
    
    # Matrice de loi Gausienne, centrée et de covariance Cov:
    C = cholesky(Cov)
    y_c = C.dot(X)          # vecteur gaussien circulaire (complexe)
    y_r = C.dot(multivariate_normal(np.zeros(N), np.eye(N)))   # vecteur gaussien réel
    
    return y_r, y_c
  
    
    
    
      
def Cov_y(theta, omega, G, f, L):
    """
    Les paramètres d'entrée:
    theta : scalaire 
    omega : vecteur de taille N
    G : matrice de corrélation temporelle
    f : fonction de phase
    L : Nombre de pixels
    
    Les paramètres de sortie:
    C : matrice de covariance associée à la variable y(theta, omega)
    """
    
    N = np.array(omega).shape[0]
    phi = f(np.arange(0,N-1)-(N-1))*theta
    phi = np.concatenate((phi,np.zeros(1))) + omega
    
    C = np.zeros((N*L, N*L), dtype=complex)
    for n in range(N):
        for m in range(n+1):
            bloc = G[n,m]*np.exp(1j*(phi[n]-phi[m]))*np.eye(L)
            C[n*L:n*L+L, m*L:m*L+L] = bloc
            C[m*L:m*L+L, n*L:n*L+L] = np.conj(bloc)
    
    return(C)
  
    
    
    
    
def extraction_phase(y, K, G_inv, N, L):
    """
    Les paramètres d'entrée:
    y : vecteur NxL
    K : nombre d'iteration de l'algo de Newton
    G_inv : matrice de corrélation temporelle inverse
    N : nbr d'images
    L : nombre de pixels
    
    Les paramètres de sortie:
    phi_exp : différencs de phases (par rapport à l'image N) vecteur taille N-1
    """
    # construction de la matrice R
    R = np.zeros((N,N),dtype=complex)
    for l in range(L):
        for n in range(N):
            for m in range(N):
                R[n,m] += y[n*L+l]*np.conj(y[m*L+l])
    R = R/L

    phi_exp = np.zeros(N)
    for k in range(K):
        intermediaire = np.zeros(N, dtype=complex)
        intermediaire[N-1] = 1
        for n in range(N-1):
            for p in range(N):
                if p != n: 
                    intermediaire[n] += G_inv[n,p]*R[n,p]*np.exp(1j*phi_exp[p])/(1-G_inv[n,n])
        phi_exp = np.angle(intermediaire)
        # if k%10==0:
        #     print('rang ', k)
        #     print("fonction ", fonction_minimisee(phi_exp[:-1], y, G_inv, N, L))
    
    return(phi_exp[:-1])
  
    
    
    
    
def extraction_phase_optimise(y, psi_initial, G_inv, N, L):
    """
    Les paramètres d'entrée:
    y : vecteur NxL
    psi_initial : valeur de départ pour l'itération
    K : nombre d'iteration de l'algo de Newton
    G_inv : matrice de corrélation temporelle inverse
    N : nbr d'images
    L : nombre de pixels
    
    Les paramètres de sortie:
    norme : distance entre le resultat de la derniere itération et l'avant dernière
    """
    # construction de la matrice R
    R = np.zeros((N,N),dtype=complex)
    for l in range(L):
        for n in range(N):
            for m in range(N):
                R[n,m] += y[n*L+l]*np.conj(y[m*L+l])
    R = R/L

    #phi_exp = np.zeros(N)
    phi_exp = psi_initial
    erreur = 10
    k = 1
    
    while erreur > 10**(-5):
        k+=1
        intermediaire = np.zeros(N, dtype=complex)
        intermediaire[N-1] = 1
        for n in range(N-1):
            for p in range(N):
                if p != n: 
                    intermediaire[n] += G_inv[n,p]*R[n,p]*np.exp(1j*phi_exp[p])/(1-G_inv[n,n])
        phi_old = phi_exp
        phi_exp = np.angle(intermediaire)
        erreur = np.sqrt(np.sum((phi_exp[:-1]-phi_old[:-1])**2)/N)
        
        """
        if k%10==0:
            print('rang ', k)
            print("fonction ", fonction_minimisee(phi_exp[:-1], y, G_inv, N, L))
        """
    #return(np.sum((phi_exp[:-1]-phi_old[:-1])**2)/N)
    print(" ")
    print("erreur", np.sqrt(np.sum((phi_exp[:-1]-phi_old[:-1])**2)/N))
    return(phi_exp[:-1])
    
    
    
    
def extraction_phase_norme(y, K, G_inv, N, L):
    """
    Les paramètres d'entrée:
    y : vecteur NxL
    psi_initial : valeur de départ pour l'itération
    K : nombre d'iteration de l'algo de Newton
    G_inv : matrice de corrélation temporelle inverse
    N : nbr d'images
    L : nombre de pixels
    
    Les paramètres de sortie:
    norme : distance entre le resultat de la derniere itération et l'avant dernière
    """
    # construction de la matrice R
    R = np.zeros((N,N),dtype=complex)
    for l in range(L):
        for n in range(N):
            for m in range(N):
                R[n,m] += y[n*L+l]*np.conj(y[m*L+l])
    R = R/L

    phi_exp = np.zeros(N)
    for k in range(K):
        intermediaire = np.zeros(N, dtype=complex)
        intermediaire[N-1] = 1
        for n in range(N-1):
            for p in range(N):
                if p != n: 
                    intermediaire[n] += G_inv[n,p]*R[n,p]*np.exp(1j*phi_exp[p])/(1-G_inv[n,n])
        phi_old = phi_exp
        phi_exp = np.angle(intermediaire)
        # if k%10==0:
        #     print('rang ', k)
        #     print("fonction ", fonction_minimisee(phi_exp[:-1], y, G_inv, N, L))

    return(np.sum((phi_exp[:-1]-phi_old[:-1])**2)/N)

    
    
    
def fonction_minimisee(phi, *param):
    
    y, G_inv, N, L = param[0], param[1], param[2], param[3]
    """
    Les paramètres d'entrée:
    phi : vecteur de taille N-1 qui correspond aux différences de phases
    y : vecteur NxL
    G_inv : matrice de corrélation temporelle inverse
    
    Les paramètres de sortie:
    scalaire: evaluation de la forme quadratique en phi
    """
    
    eta = np.exp(1j*np.array(phi))
    eta = np.concatenate((eta, np.array([1])))
    
    # construction de la matrice R
    R = np.zeros((N,N),dtype=complex)
    for l in range(L):
        for n in range(N):
            for m in range(N):
                R[n,m] += y[n*L+l]*np.conj(y[m*L+l])
    R = R/L
    return(np.transpose(np.conj(eta)).dot(G_inv*R).dot(eta))
    

