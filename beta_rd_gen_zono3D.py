#############Generation de zonotope en 3 dimensions 

#   #   #   #   Intro   #   #   #   #
###Pour installer des packages pour Python3, utiliser pip3.
import numpy as np
import math as math
import random as rd
import scipy.stats as stt
import time
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import plotly.figure_factory as ff




zeta_3 = 1.2020569031595942853997381615
zeta_2 = math.pi**2/6
zeta_4 = math.pi**4/90


t1 = time.perf_counter() 

'''pour n = 100, la longueur est 13000 (130x)
pour n = 200, la longueur est 36000 (180x)
n = 300  64480
n = 400 91000
pour n = 500, la longueur est 130000 (260x)
n = 600 171320 
pour n = 1000, la longueur est 350000 (350x)
pour n = 2000, la longueur est 920000 (460x)
n = 3000 1604640
pour n = 3600, la longueur est 2054000 (570x)'''

listesss = ([100, 1000, 3000, 9000],
            [280, 2840, 8680, 26350])  #facteur 2.9 entre les deux

#taille moyenne de la génération
n = 400

#Taille de controle de la série des zonotopes (exp de log de l'autre série)
n_prime = int(n/10)         #n_prime et n_second sont calculés pour n < 3600 (état actuel des choses)

#n_second sert à controler la taille de la liste des proba.
n_second = int(n/10)                

#solution de l'equation du col
x = math.exp(- (4 * zeta_4/zeta_3/n)**(1/4) )


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
    borne = int(n/(m))
    i = 2
    A.append(A[len(A)-1]+liste_gene[i]* x**i)
    while (i < borne and A[len(A)-1]/ A[len(A)-2] > 1 + 1/(10*n)):
        A.append(A[len(A)-1]+liste_gene[i]* x**i)
        i += 1
    return A


#   #   #   #   Calculs préalables    #   #   #   #

#Liste des générateurs primitifs   ######## Calcul fait dans un csv (en n^3) 
import csv
liste_gene = [0]
with open('/home/theo/Documents/coding/these/nb_gene_primitifs.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        liste_gene.append(int(row[0]))
t2 = time.perf_counter() 

liste_phi = [0]
with open('/home/theo/Documents/coding/these/nb_gene_primitifs2D.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        liste_phi.append(int(row[0]))
print('calcul des nb de generateurs primitifs', t2 - t1 )

#print(liste_phi)

#calcul de [A], la liste des sommes partielles de A(x**m) (somme jusqu'à n/m**2 )
A = [[1]]
for m in range (1, n):
    A.append(gene_A(x**m,m))
    if (A[m][-1] < 1/(10*n)):
        break
t3 = time.perf_counter() 
print(len(A[1]))
print('sommes partielles A calculées', t3 - t2 )

'''#test de print A. 
plt.plot([A[i][len(A[i])-1] for i in range (len(A))])
plt.ylabel('Sommes partielles des SG den_second t.g. x^i.')
'''

#calcul des probas de tirer k : K est la liste des probas cumulées \prod(exp( A(x**m)/m))
# On passe à l'exponentielle à la fin
K = [0]
for m in range (1,len(A)):
    K.append(K[m-1] + A[m][len(A[m])-1]/m)
K.append(K[len(K)-1] + (n_prime*x**(n_prime-1) - (n_prime-1)*x**n_prime) / (1-x)**2)   ## ce calcul est trop bizarre 
for m in range (len(K)):
    K[m] = np.exp(K[m]-K[len(K)-1]) 
t4 = time.perf_counter() 
print('probas pour K calculées', t4 - t3 )
print(' Taille de K :',len(K), '\n')

'''#test de print k. 
plt.plot(K)
plt.ylabel('Sommes partielles des probas de tirer k le nombre de fois quon va faire des poissons.')
'''

#calcul des probas de tirer N : taille du chemin primitif (en fonction de la serie des géné)
N = [[1]]
for m in range (1, len(A)): 
    N.append([A[m][i]/A[m][len(A[m])-1] for i in range (1,len(A[m]))])  
t5 = time.perf_counter() 
print('probas pour N calculées', t5 - t4 , '\n')

'''#test de print N. 
plt.plot(N[30])
plt.ylabel('Taille du chemin primitif (seulement en fonction de la solution du col à la puissance i).')
'''

# plt.show()

#   #   #   #   Generateur de Boltzmann #   #   #   #

#générateur des chemins primitifs (Gamma GP) en fonction de la taille (norme 1)
def generateur_primitif (j) : 
    n  = rd.random()
    i = find(n,N[j])
    # le premier terme de N correspond au chemin de longueur 1, donc il faut rajouter 1 pour arriver à n.
    n = i+1                
    #print('generation prim ', n)
    if (i ==0) :
        coin = math.floor(rd.random()*3)  
        if (coin == 0):
            return(1.,0.,0.) 
        elif (coin == 1): 
            return (0.,1.,0.)
        else:
            return (0.,0.,1.)
    p = n 
    q = n 
    # on décide le nombre de dimension du chemin
    dimension = (rd.random())
    if (dimension > liste_phi[n]/liste_gene[n]):
        while ( math.gcd(p,math.gcd(q-p,n-q))!=1 and (p==0 or q-p==0 or n-q==0)):
            p0 = math.floor(rd.random()*n) 
            q0 = math.floor(rd.random()*n)
            p = min(p0,q0)
            q = max(p0,q0)
            #print('p donne ', p)
        coin1 = math.floor(rd.random()*2) 
        coin2 = math.floor(rd.random()*2)
        return (float(p), (-1)**coin1*float(q-p), (-1)**coin2*float(n-q))
    else:
        while (math.gcd(p,n) != 1):
            p = math.floor(rd.random()*n+1) 
            #print('p donne ', p)
        coin1 = math.floor(rd.random()*2)
        coin2 = math.floor(rd.random()*3)
        if (coin1 == 0):
            q = n - q
        else:
            q = p-n 
        if (coin2 == 0):
            return (0., float(p), float(q))
        elif (coin2 == 1): 
            return (float(p), 0., float(q))
        else:
            return (float(p), float(q), 0.)


#test de générateur primi
#print(N[5])
#print(generateur_primitif(5))

 
def generateur_zonogone ():
#générateur des zonogones (Gamma Zono), petit gamma = liste des générateurs
    gamma = []
    longueur = 0
    tirage = rd.random()
    #print('tirage =',tirage)
    k = find(tirage, K) 
    #print('K = ', k) 
    for i in range (1,k+1):
        print('etape n ', i)
        a = stt.poisson.rvs(mu= A[i][len(A[i])-1]/i, size=1)[0]
        #print('le poisson donne', a)
        if (a != 0):
            for alpha in range (1, a+1):
                #print ('et de ', alpha)
                p, q, r = generateur_primitif(i) 
                longueur = longueur + abs(p)*i + abs(q)*i + abs(r)*i
                #print(p, q, r)
                norme = (p**2+ q**2 + r**2)**(1/2)
                #Si le générateur est déjà dans la liste, on le grandit plutot que d'en recréer un autre 
                # (pour faire l'enveloppe convexe plus tard, c'est mieux)
                pas_ajouté = True
                for a in gamma :
                    prod_norme = (a[0]**2 + a[1]**2 + a[2]**2)**(1/2)*norme
                    prod_sca = p*a[0]+ q*a[1] + r*a[2]
                    if (prod_norme == prod_sca):
                        pas_ajouté = False
                        a = [a[0] + p*i, a[1] + q*i, a[2] + r*i]
                if (pas_ajouté):
                    gamma.append([p*i, q*i, r*i])
    #print('longueur du zono' ,longueur)
    return gamma, longueur


#   #   #   #   Main    #   #   #   #
moyenne = 0
nombre = 1
liste_Generateurs = []
for i in range (nombre):
    x, longueur = generateur_zonogone()
    liste_Generateurs = liste_Generateurs + x
    print(x,longueur)
    moyenne += longueur/nombre

### Ploting the distribution of generators' length. 
for i in range (len(liste_Generateurs)):
    liste_Generateurs[i]= sum([abs(number) for number in liste_Generateurs[i]])
hist_data = [liste_Generateurs]
group_labels = ['taille des générateurs'] # name of the dataset
fig = ff.create_distplot(hist_data, group_labels)
fig.show()

print(moyenne)


'''
nb_zonotopes = 2

#taille moyenne des zonotopes
moyenne = 0

#liste des générateurs
liste_gamma = []
liste_GAMMA = []

ax = []

moyenne_nb_face = 0
for j in range (nb_zonotopes):
    gamma, longueur = generateur_zonogone()
    moyenne += longueur/nb_zonotopes
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
print(moyenne)
print(moyenne_nb_face/nb_zonotopes)
print(gamma)


#   #   #   #   Affichage   #   #   #   #

fig = plt.figure(1, figsize=(7.5,7.5), dpi=120)'''

