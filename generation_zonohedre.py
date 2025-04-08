#############Generation de zonotope en 3 dimensions 

#   #   #   #   Intro   #   #   #   #
import numpy as np
import math as math
import random as rd
from scipy.stats import *
import time
from matplotlib import pyplot as plt

zeta_3 = 1.2020569031595942853997381615
zeta_2 = math.pi**2/6
zeta_4 = math.pi**4/90


t1 = time.perf_counter() 

#taille moyenne de la génération
n = 3600

#Taille de controle de la série des zonotopes (exp de log de l'autre série)
n_prime = int(n/10)         

#n_second sert à controler la taille de la liste des proba.
n_second = int(n/100)                

#solution de l'equation du col
x = 1 - (3 * zeta_3/math.pi**2/n)**(1/3)

#!!! la taille moyenne est 6*n  which is weird 


#   #   #   #   Fonctions préalables  #   #   #   #   

#Trouve la place de x dans la suite croissante L
def find(x, L):  
    if (len(L)<= 10):
        for i in range (len(L)):
            if (x < L[i]):
                return i
        return len(L)
    i = math.floor(len(L)/2)
    if (x < L[i]):
        return (find(x,L[:i]))
    else:
        return i + (find( x,L[i:]))

#generalisation de phi : indicatrice du nombre de triplet premiers entre eux dont la somme est n. 
def nb_generateur_primitifs (n): 
    result = 0
    if (n ==1):
        result = 3
    if (n >= 2):
        for i in range (1, n):
            if (math.gcd(i, n) == 1):
                result += 6
            if (i < n - 1):
                for j in range (i + 1, n):
                    if (math.gcd(i, math.gcd(j - i, n - j)) == 1):
                        result += 4        
    return result

# La série des générateurs primitifs nommée A (Liste des sommes partielles)
def gene_A(x, m): 
    A = [liste_gene[1]*x]
    borne = int(n/(m**2))
    for i in range (2, borne):
        A.append(A[len(A)-1]+liste_gene[i]* x**i)
    return A


#   #   #   #   Calculs préalables    #   #   #   #

#Liste des générateurs primitifs   ######## Calcul fait dans un csv (en n^3) 
import csv
liste_gene = [0]
with open('/home/theo/Documents/coding/these/nb_gene_primitifs.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        liste_gene.append(row)
        if(len(liste_gene) == 1000): 
            print(row)
t2 = time.perf_counter() 
print('calcul des nb de generateurs primitifs', t2 - t1 )

#calcul de [A], la liste des sommes partielles de A(x**m) (somme jusqu'à n/m**2 )
A = [[1]]
for m in range (1, n_prime):
    A.append(gene_A(x**m,m))
t3 = time.perf_counter() 
print('sommes partielles calculées', t3 - t2 )

#calcul des probas de tirer k : K est la liste des probas cumulées \prod(exp( A(x**m)/m))
# On passe à l'exponentielle à la fin
K = [0]
for m in range (1,n_prime):
    K.append(K[m-1] + A[m][len(A[m])-1]/m)
K.append(K[len(K)-1] + (n_prime*x**(n_prime-1) - (n_prime-1)*x**n_prime) / (1-x)**2)   ## ce calcul est trop bizarre 
for m in range (len(K)):
    K[m] = np.exp(K[m]-K[len(K)-1]) 
t4 = time.perf_counter() 
print('probas pour K calculées', t4 - t3 )

#calcul des probas de tirer N : taille du chemin primitif (en fonction de la serie des géné)
N = [[0]]
for m in range (1, n_prime): 
    N.append([A[m][i]/A[m][len(A[m])-1] for i in range (1,len(A[m]))])  ########pourquoi m et pas i ?? 
t5 = time.perf_counter() 
print('probas pour N calculées', t5 - t4 )


print ('\n', K, '\n')


#   #   #   #   Generateur de Boltzmann #   #   #   #

#générateur des chemins primitifs (Gamma GP) en fonction de la taille (norme 1)
def generateur_primitif (j) : 
    n  = rd.random()
    i = find(n,N[j])
    ## le premier terme de N correspond au chemin de longueur 1, donc il faut rajouter 1 pour arriver à n.
    n = i+1                
    coin = math.floor(rd.random()*3)
    #print('generation prim ', p)
    if (i == 0):
        
        return(coin, 1-coin) 
    p = n 
    q = n 
    while (math.gcd(p,math.gcd(q,n)) != 1):
        p = math.floor(rd.random()*(n)) 
        q = math.floor(rd.random()*(n))
        #print('p donne ', p)
    if (coin == 0):
        return(p, i+1-p)
    return (p, p -i -1)

def generateur_zonogone ():
#générateur des zonogones (Gamma Zono)
    gamma = []
    longueur = 0
    tirage = rd.random()
    #print('tirage =',tirage)
    k = find(tirage, K)   ### c'est ici qu'on peut faire une truc avec le log wesh
    print('K = ', k) 
    for i in range (1,k+2):
        #print('etape n ', i)
        a = 0
        while (a == 0 or i > k+1 ):   # je ne comprends pas cette boucle (mais je ne comprends pas la différence dans l'algo)
            a = poisson.rvs(mu= A[i][len(A[i])-1]/i, size=1)[0]
            #print('le poisson donne', a)
        for alpha in range (1, a+1):
            #print ('et de ', alpha)
            p, q = generateur_primitif(i) 
            longueur = longueur + abs(p)*i + abs(q)*i
            #print(p, q)
            if (gamma == []):
                gamma.append([p*i, q*i])
            else:    #boucle qui sert à what ??? 
                if (p == 0):
                    gamma.insert(0, [0, q*i])
                else:   
                    flag = 0 
                    while (flag < len(gamma) and gamma[flag][0]==0):
                        flag += 1
                    bool1 = True
                    while (flag < len(gamma) and bool1): 
                        if (q/p > gamma[flag][1]/gamma[flag][0]) :
                            bool1 = False 
                        else:
                            flag += 1
                    if (flag == len(gamma)):
                        gamma.append([p*i, q*i])   
                    else:
                        gamma.insert(flag, [p*i, q*i])
    print(longueur)
    return gamma, longueur


#   #   #   #   Main    #   #   #   #




#   #   #   #   Affichage   #   #   #   #





#########################################################Générateur de Boltzmann##################################################################

from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch

def generateur_primitif (j) : 
#générateur des chemins primitifs (Gamma GP) en fonction de la taille (norme 1)
    n  = rd.random()
    i = find(n,N[j])
    p = i+1
    coin = math.floor(rd.random()*2)
    #print('generation prim ', p)
    if (i == 0):
        return(coin, 1-coin) 
    while (math.gcd(p,i+1) != 1):
        p = math.floor(rd.random()*(i+1)+1) 
        #print('p donne ', p)
    if (coin == 0):
        return(p, i+1-p)
    return (p, p -i -1)

def generateur_zonogone ():
#générateur des zonogones (Gamma Zono)
    gamma = []
    longueur = 0
    tirage = rd.random()
    #print('tirage =',tirage)
    k = find(tirage, K)   ### c'est ici qu'on peut faire une truc avec le log wesh
    print('K = ', k) 
    for i in range (1,k+2):
        #print('etape n ', i)
        a = 0
        while (a == 0 or i > k+1 ):   # je ne comprends pas cette boucle (mais je ne comprends pas la différence dans l'algo)
            a = poisson.rvs(mu= A[i][len(A[i])-1]/i, size=1)[0]
            #print('le poisson donne', a)
        for alpha in range (1, a+1):
            #print ('et de ', alpha)
            p, q = generateur_primitif(i) 
            longueur = longueur + abs(p)*i + abs(q)*i
            #print(p, q)
            if (gamma == []):
                gamma.append([p*i, q*i])
            else:    #boucle qui sert à what ??? 
                if (p == 0):
                    gamma.insert(0, [0, q*i])
                else:   
                    flag = 0 
                    while (flag < len(gamma) and gamma[flag][0]==0):
                        flag += 1
                    bool1 = True
                    while (flag < len(gamma) and bool1): 
                        if (q/p > gamma[flag][1]/gamma[flag][0]) :
                            bool1 = False 
                        else:
                            flag += 1
                    if (flag == len(gamma)):
                        gamma.append([p*i, q*i])   
                    else:
                        gamma.insert(flag, [p*i, q*i])
    print(longueur)
    return gamma, longueur



fig = plt.figure(1, figsize=(7.5,7.5), dpi=120)
nb_zonogones = 20
moyenne = 0
liste_gamma = []
liste_GAMMA = []
ax = []
moyenne_nb_face = 0
for j in range (nb_zonogones):
    x_max = 0 
    x_min = 0
    y_max = 0
    y_min = 0
    gamma, longueur = generateur_zonogone()
    moyenne += longueur
    liste_gamma.append(gamma)
    GAMMA = [(0,0)]
    moyenne_nb_face += 1
    for i in range (len(gamma)):
        x, y = GAMMA[-1]
        if (y > y_max):
            y_max = y
        GAMMA.append((x+ gamma[i][0]*2/longueur , y+ gamma[i][1]*2/longueur))
    x_max = x
    for i in range (len(gamma)):
        x, y = GAMMA[-1]
        if (y< y_min):
            y_min = y
        GAMMA.append((x - gamma[i][0]*2/longueur, y - gamma[i][1]*2/longueur))
    for i in range(len(GAMMA)):
        x, y = GAMMA[i] 
        GAMMA[i] = (x , y- y_min )
    for i in range(1, len(gamma)):
        if (gamma[i] != gamma[i-1]):
            moyenne_nb_face += 1
    liste_GAMMA.append(GAMMA)
    poly = Polygon(GAMMA)
    x,y = poly.exterior.xy
    ax.append(fig.add_subplot(111))
    ax[j].plot(x, y, color='#6699cc', alpha=0.7,
        linewidth=1, solid_capstyle='round', zorder=2)
    ax[j].set_title("Génération de taille d'environ 800000")
print(moyenne/nb_zonogones)
print(moyenne_nb_face/nb_zonogones)
print(gamma)


alph = 1/40/41

x = [alph*(i*(i+1))/2 for i in range (40)]
y = [((2*x[i])**(1/2) - x[i]) +.5 for i in range (len(x))]
x_1 = [1 - alph*((39-i)*(40-i))/2 for i in range (40)]
y_1 = [y[len(y)-1 - i] for i in range (len(y))]
X = x + x_1
Y = y + y_1
ay = fig.add_subplot(111)
ay.plot(X, Y, color='black', alpha=0.7,
    linewidth=1, solid_capstyle='round', zorder=2)

x = [alph*(i*(i+1))/2 for i in range (40)]
y = [- ((2*x[i])**(1/2) - x[i]) +.5 for i in range (len(x))]
x_1 = [1 - alph*((39-i)*(40-i))/2 for i in range (40)]
y_1 = [y[len(y)-1 - i] for i in range (len(y))]
X = x + x_1
Y = y + y_1
ay = fig.add_subplot(111)
ay.plot(X, Y, color='black', alpha=0.7,
    linewidth=1, solid_capstyle='round', zorder=2)


