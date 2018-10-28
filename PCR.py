#PCR
#Ejercicio3
# Los arrays `u` y `v` representan dos series en funcion del tiempo `t`.
# Grafique las dos series de datos en una misma grafica y guardela sin mostrarla en 'serie.pdf'
# Calcule la covarianza entre `u` y `v` e imprima su valor.

import numpy as np
import matplotlib.pylab as plt

data=np.genfromtxt("WDBC.dat", delimiter=",", replace_space='_',dtype="unicode")
for i in range(len(data[0,:])):
    for j in range(len(data[:,0])):
        if(data[j,i]=="M"):
            data[j,i]=1
        elif(data[j,i]=="B"):
            data[j,i]=0
v0=data[:,0]
v1=data[:,1]
v2=data[:,2]
v3=data[:,3]
v=[]
v.append(4)
#for i in range(len(data[0,:])):


print(np.mean(v0.astype(float)))
#v0=(data[0,:]-np.mean(data[0,:]))/(np.sqrt(np.var(data[0,:])))
# HW3
