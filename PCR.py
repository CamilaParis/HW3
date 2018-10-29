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
            data[j,i]=2
v=[]
for i in range(len(data[0,:])):
	v.append(data[i,:])
for i in range(len(data[0,:])):
	v[i]=v[i].astype(float)
	v[i]=(v[i]-np.mean(v[i]))/(np.sqrt(np.var(v[i])))
#vect = np.vstack([v[0],v[1],v[2],v[3]])
matriz_cov = np.cov(v)
#(matriz_cov)
val, vector=np.linalg.eig(matriz_cov)
print(val)
for i in range(len(val)):
	ap=val[i]
	bp=vector[i]
	print("El valor propio es", ap, "y su autovector es", bp)
r=0.0
r2=0.0
r3=0.0
vr=0.0
vr2=0.0
vr3=0.0
for i in range(len(val)):
	if (val[i]>r):
		r=val[i]
		vr=i
	elif (r>val[i]>r2):
		r2=val[i]
		vr2=i
	elif (r2>val[i]>r3):
		r3=val[i]
		vr3=i
print(vr, vr2)
v1=vector[0]
v2=vector[1]
va=v[0]
vb=v[1]
for i in range(len(v[0])):
    p1=np.dot(va[i], v1)/np.linalg.norm(v1)
    p2=np.dot(vb[i], v1)/np.linalg.norm(v1)
    p4=np.dot(va[i], v2)/np.linalg.norm(v2)
    p5=np.dot(vb[i], v2)/np.linalg.norm(v2)
P1=[p1+p2]
P2=[p4+p5]
plt.figure()
plt.scatter(P1,P2)
plt.show()
# HW3
