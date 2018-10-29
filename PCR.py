#PCR
#Ejercicio3
# Los arrays `u` y `v` representan dos series en funcion del tiempo `t`.
# Grafique las dos series de datos en una misma grafica y guardela sin mostrarla en 'serie.pdf'
# Calcule la covarianza entre `u` y `v` e imprima su valor.

import numpy as np
import matplotlib.pylab as plt
import urllib.request

urllib.request.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", "WDBC.dat")
data=np.genfromtxt("WDBC.dat", delimiter=",", replace_space='_',dtype="unicode")
for i in range(len(data[0,:])):
    for j in range(len(data[:,0])):
        if(data[j,i]=="M"):
            data[j,i]=1
        elif(data[j,i]=="B"):
            data[j,i]=2
v=[]
for i in range(2,len(data[0,:])):
	v.append(data[i,:])
for i in range(len(v)):
	v[i]=v[i].astype(float)
	v[i]=(v[i]-np.mean(v[i]))/(np.sqrt(np.var(v[i])))
#vect = np.vstack([v[0],v[1],v[2],v[3]])
matriz_cov = np.cov(v)
##usar mi mierda
#(matriz_cov)
val, vector=np.linalg.eig(matriz_cov)
print(val)
for i in range(len(val)):
	ap=val[i]
	bp=vector[i]
	print("El valor propio es", ap, "y su autovector es", bp)

v1=vector[0]
v2=vector[1]

print(np.shape(v1))
p1=np.dot(v, v1)/np.linalg.norm(v1)
p2=np.dot(v, v2)/np.linalg.norm(v2)



# HW3

