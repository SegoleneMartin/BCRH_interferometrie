"""Ce code a pour but de reproduire les figures de l'article "Hybrid Cramer-Rao Bounds for Crustal Displacement Field Estimators in SAR Interferométry" : 
(SECTION 1) borne en fonction de N pour différents L,
(SECTION 2) borne pour différentes corrélations,  
(SECTION 3) comparaison avec la borne déterministe sur v, 
(SECTION 4) extension à 2 paramètres déterministes"""

import os, sys
from inspect import getsourcefile
from os.path import abspath, dirname
from numpy.linalg import inv, pinv
from numpy import transpose as t

os.chdir(dirname(abspath(getsourcefile(lambda:0))))

### 

import numpy as np
import pylab as plt
from numpy import pi, sin, sqrt
from numpy.random import rand, multivariate_normal, normal

# PARAMETRES            
lamb = 56           # longeur d'onde
g_0 = 0.7           # coef de correlation initial
r = 0.975           # intervient dans la correlation
sigma = 1           # variance de alpha
dt = 12             # interval de temps en jours

# FONCTIONS
f = lambda x: 4*pi*dt*x/lamb           # fonction dérivée de phi

f_inv = lambda x: 1/(4*pi*dt*x)*lamb   # inverse de la fonction f

def g_temp(x):                         # fonction de correlation temporelle
    if x != 0:
        return g_0*r**(np.abs(x)*dt)
    else:
        return g_0*r**(np.abs(x)*dt)+(1-g_0)
g_temp = np.vectorize(g_temp)

### SECTION 1: BCRH en fonction de N, pour plusieurs L différents

liste_N =  np.arange(3, 80, 2)
#for L in [1, 5, 30, 300]:
    
for L in [10]:
    
    borne = []

    for N in liste_N:

        try :
            """calcul matrice G (correlation)"""
            a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
            G = g_temp(b-a)
            
            """calcul de theta, X et V"""
            theta = f(np.arange(0,N-1)-(N-1))
            theta.reshape(N-1,1)
            X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
            V = sigma**2*(np.eye(N-1)+1)
            
            """calcul BCRH"""
            borne.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
            
        except np.linalg.LinAlgError:
            print("Matrice non inversible")

    # PLOT
    plt.loglog(liste_N, np.array(borne), label="L = {}".format(L))
plt.loglog(liste_N, np.sqrt(1/np.array(liste_N)), color ='g', label= "f(x) = 1/x")
plt.figure(1)
plt.xlabel("nombre d'images N")
plt.ylabel("BCRH en mm/jour")
plt.legend()
plt.show()

# plt.savefig('figure_1_Guarnieri.eps', format='eps', bbox_inches='tight', dpi=1200)

### SECTION 2: BCRH en fonction de N, pour des correlations différentes

L = 10
sigma_v_2 = (lamb/(4*pi*dt))**2*(2*sigma**2+(1-g_0**2*r**(2*dt))/(2*g_0**2*r**(2*dt)*L))

borne = []
liste_N = []

for N in range(4, 90, 1):
    
    """calcul matrice G (correlation)"""
    a, b = np.meshgrid(np.arange(N),np.arange(N))
    G = g_temp(b-a)
    
    """calcul de theta, X et V"""
    theta = f(np.arange(N-1)-(N-1))
    theta.reshape(N-1,1)
    X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
    V = sigma**2*(np.eye(N-1)+1)

    """calcul BCRH"""
    borne.append(sqrt(1/(theta.dot(np.linalg.inv(np.linalg.inv(X)+V)).dot(theta))))
    liste_N.append(N)


# PLOT

plt.figure(2)
plt.loglog(liste_N, borne, label='temp decorrelation + thermal noise' )
plt.loglog(liste_N, sqrt((lamb/(4*pi*dt))**2*(1-r**(2*dt))/(2*L*r**(2*dt)*(np.array(liste_N)-1))), linestyle='--',label='temp decorrelation')
plt.loglog(liste_N, sqrt(sigma_v_2*6/(np.array(liste_N)**3-np.array(liste_N))), linestyle='--', label='thermal noise')
plt.xlabel("nombre d'images N")
plt.ylabel("BCRH en mm/jour")
plt.legend()
plt.show()

# plt.savefig('figure_2_Guarnieri.eps', format='eps', bbox_inches='tight', dpi=1200)

### SECTION 3: Comparaison BCRH et borne deterministe sur theta

L = 20

borne_hybride = []
borne_deter_theta = []
liste_N = []

for N in range(2, 90, 2):
    
    try :
        """calcul matrice G (correlation)"""
        a, b = np.meshgrid(np.arange(1,N+1),np.arange(1,N+1))
        G = g_temp(b-a)
        
        """calcul de theta, X et V"""
        theta = f(np.arange(1,N)-(N-1))
        theta.reshape(N-1,1)
        G_inv = inv(G)
        X = 2*L*(G*G_inv-np.eye(N))[:-1, :-1]
        V = sigma**2*(np.eye(N-1)+1)
        
        borne_hybride.append(sqrt(1/(theta.dot(np.linalg.inv(inv(X)+V)).dot(theta))))
        borne_deter_theta.append(sqrt(1/(theta.dot(X).dot(theta))))
        liste_N.append(N)
        
    except np.linalg.LinAlgError:
        print("Matrice non inversible")

# PLOT
plt.figure(3)

plt.loglog(liste_N, np.array(borne_hybride), label="Borne hybride", color='orange')
plt.loglog(liste_N, np.array(borne_deter_theta), label="Borne deterministe", color='green')
plt.xlabel("nombre d'images N")
plt.ylabel("BCRH en mm/jour")
plt.legend()
plt.show()

plt.savefig('figure_3_Guarnieri.eps', format='eps', bbox_inches='tight', dpi=1200)

### SECTION 4: Extension à deux paramètres

"""Dans cette section on prend comme paramètre deterministe (v, delta_h) et on plot la borne sur les deux paramètres. Les baseline à chaque temps sont tirées uniformément entre B_min et B_max."""

L = 10
B_min = 0                   # baseline minimal en mm
B_max = 60000               # baseline maximale en mm
R = 853*10**6               # distance du DAR au scatterer
angle_inci = 23/180*pi      # angle d'incidence en radians
N_exp = 300

borne_v = []
borne_dh = []
liste_N = []

for N in range(10, 100, 1):
    
    borne_v_exp = []
    borne_dh_exp = []
    liste_N.append(N)
    
    V = sigma*(np.eye(N-1)+1)
    a, b = np.meshgrid(np.arange(N),np.arange(N))
    G = g_temp(b-a)
    X = 2*L*(G*np.linalg.inv(G)-np.eye(N))[:-1,:-1]
    Mat = np.linalg.inv(np.linalg.inv(X)+V)
    
    for experience in range(N_exp):
        
        B = rand(N-1)*(B_max - B_min) + B_min
        theta = np.concatenate((f(np.arange(N-1)-(N-1)).reshape(N-1,1), (B/(R*sin(angle_inci))).reshape(N-1,1)), axis=1)
        borne = np.linalg.inv((np.transpose(theta)).dot(Mat.dot(theta)))
        borne_v_exp.append(sqrt(borne[0,0]))
        borne_dh_exp.append(sqrt(borne[1,1]))
        
    borne_v.append(np.sum(borne_v_exp)/N_exp)
    borne_dh.append(1000*np.sum(borne_dh_exp)/N_exp)
    
    
# PLOT
plt.figure(4)
plt.loglog(liste_N, borne_v, label='BCRH sur v (mm/jour)' )
plt.xlabel("nombre d'images N")
plt.legend()
plt.savefig('figure_4_Guarnieri.eps', format='eps', bbox_inches='tight', dpi=1200)

plt.figure(5)
plt.loglog(liste_N, borne_dh, label='BCRH sur dh (m)' )
plt.xlabel("nombre d'images N")
plt.legend()
# plt.savefig('figure_5_Guarnieri.eps', format='eps', bbox_inches='tight', dpi=1200)
plt.show()


## Algo pour deux paramètres en faisant varier B_perp_max # loi uniforme

"""On fait l'experience pour plusieurs B_perp_max"""

L = 10
B_min = 0
R = 853*10**6               # distance du SAR au scatterer
angle_inci = 23/180*pi      # angle d'incidence en radians
N_exp = 20


for B_max in np.linspace(1000, 1000000, 5):
    
    borne_v = []
    borne_dh = []
    liste_N = []

    for N in range(10, 100, 1):
        
        borne_v_exp = []
        borne_dh_exp = []
        liste_N.append(N)
        V = sigma*(np.eye(N)+1)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))
        Mat = np.linalg.inv(np.linalg.inv(X)+V)
        
        for experience in range(N_exp):
            
            B = rand(N)*(B_max - B_min) + B_min
            theta = np.concatenate((f(np.arange(N)-N).reshape(N,1), (B/(R*sin(angle_inci))).reshape(N,1)), axis=1)
            borne = np.linalg.inv((np.transpose(theta)).dot(Mat.dot(theta)))
            borne_dh_exp.append(sqrt(borne[1,1]))
            
        borne_dh.append(np.sum(borne_dh_exp)/N_exp)
        
    plt.loglog(liste_N, np.array(borne_dh)/1000, label='B_max = {}'.format(B_max/1000) )
    
# PLOT

plt.xlabel("nombre d'images")
plt.ylabel('BCRH sur dh en m')
plt.title('BCRH sur dh pour différentes valeurs de B_max')
plt.legend()

plt.show()

## Algo pour deux paramètres en faisant varier les baselines selon une loi normale

"""Ici les baseline à chaque temps sont tirées selon une loi normale (O, B_var). On fait l'experience pour plusieurs B_var."""

L = 10
B_min = 0
R = 853*10**6               # distance du SAR au scatterer
angle_inci = 23/180*pi      # angle d'incidence en radians
N_exp = 20
B_moy = 30000


for var_B in np.linspace(1000, 100000, 5):
    
    borne_v = []
    borne_dh = []
    liste_N = []

    for N in range(10, 100, 1):
        
        borne_v_exp = []
        borne_dh_exp = []
        liste_N.append(N)
        V = sigma*(np.eye(N)+1)
        a, b = np.meshgrid(np.arange(N),np.arange(N))
        G = g_temp(b-a)
        X = 2*L*(G*np.linalg.inv(G)-np.eye(N))
        Mat = np.linalg.inv(np.linalg.inv(X)+V)
        
        for experience in range(N_exp):
            
            B = normal(B_moy, var_B, N)
            [(x if x>0 else 0) for x in B]
            theta = np.concatenate((f(np.arange(N)-N).reshape(N,1), (B/(R*sin(angle_inci))).reshape(N,1)), axis=1)
            borne = np.linalg.inv((np.transpose(theta)).dot(Mat.dot(theta)))
            borne_dh_exp.append(sqrt(borne[1,1]))
            
        borne_dh.append(np.sum(borne_dh_exp)/N_exp)
    plt.loglog(liste_N, np.array(borne_dh)/1000, label='B_var = {}'.format(var_B/1000) )
    
# PLOT
plt.xlabel("nombre d'images N")
plt.ylabel('BCRH sur dh (m)')
plt.title('BCRH sur dh pour différentes valeurs de B_var')
plt.legend()

plt.show()
