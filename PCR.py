#PCR
import numpy as np
import matplotlib.pylab as plt
#Almacene los datos del archivo WDBC.dat
#No entendi la instruccion, si se tiene descargado el archivo WDBC.dat, entonces esta bien, si tocaba descargalo usar los siguientes dos comentarios en python 2 o los proximos 2 en python 3
#import urllib
#response = urllib.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat","WDBC.dat")
#import urllib.request
#urllib.request.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", "WDBC.dat")

data=np.genfromtxt("WDBC.dat", delimiter=",", replace_space='_',dtype="unicode")
datn=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

for i in range(len(data[0,:])):
    for j in range(len(data[:,0])):
        if(data[j,i]=="M"):
            data[j,i]=1
        elif(data[j,i]=="B"):
            data[j,i]=2

for i in range(len(datn)):
    datn[i]=(datn[i]-np.mean(datn[i]))/(np.sqrt(np.var(datn[i])))
#Calcule, con su implementacion propia, la matriz de covarianza de los datos y la imprima
m=np.zeros((30,30))
for i in range(30):
    for j in range(30):
        m[i,j]=(np.sum((datn[:,i]-np.average(datn[:,i]))*(datn[:,j]-np.average(datn[:,j]))))/(568)
#mirar con np.shape
matriz_cov = m
print("Matriz de covarianza:",matriz_cov)
#Calcule los autovalores y autovectores de la matriz de covarianza y los imprima (para esto puede usar los paquetes de linalg de numpy). Su mensaje debe indicar explıcitamente cual es cada autovector y su autovalor correspondiente.
val, vector=np.linalg.eig(matriz_cov)
print(val)
for i in range(len(val)):
    ap=val[i]
    bp=vector[i]
    print("El valor propio es", ap, "y su autovector es", bp)
#Imprima un mensaje que diga cuales son los parametros mas importantes en base a las componentes de los autovectores
r=0.0
r2=0.0
vr=0.0
vr2=0.0
#valores propios mayores
for i in range(len(val)):
    if (val[i]>r):
        r=val[i]
        vr=i
    elif (r>val[i]>r2):
        r2=val[i]
        vr2=i
print("Los parametros mas importantes son los vectores de indice",vr, vr2,"que son", vector[vr],"y",vector[vr2])
#Haga una proyeccion de sus datos en el sistema de coordenadas PC1, PC2 y grafique estos datos. Use un color distinto para el diagnostico maligno y el benigno y la guarde dicha grafica sin mostrarla en ApellidoNombre_PCA.pdf.
v1=vector[:,0]
v2=vector[:,1]
p1=np.dot(datn, v1)/np.linalg.norm(v1)
p2=np.dot(datn, v2)/np.linalg.norm(v2)
p1r=[]
p1b=[]
p2r=[]
p2b=[]


for j in range(len(data[:,0])):
    if(data[j,1]=="1"):
        p1r.append(p1[j])
        p2r.append(p2[j])
    elif(data[j,1]=="2"):
        p1b.append(p1[j])
        p2b.append(p2[j])
plt.figure()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(p1r,p2r, c="r")
plt.scatter(p1b,p2b, c="b")
plt.savefig("ParisCamila_PCA.pdf")


#Imprima un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion, si no sirve o si puede ayudar al diagnostico para ciertos pacientes, argumentando claramente su posicion.
print("Se puede observar la diferenciacion de los datos de tumores malignos y benignos en la grafica, aunque la dependencia es mayor para el PC1 porque los datos de los tumores son muy similares para PC2")
# HW3

