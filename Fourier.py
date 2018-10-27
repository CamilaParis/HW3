#Fourier
import numpy as np
import matplotlib.pylab as plt



data_signal=np.genfromtxt("signal.dat", delimiter=",")
x=data_signal[:,0]
y=data_signal[:,1]
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x,y, c="r")
plt.savefig("ParisCamila_signal.pdf")
plt.close()

def Fourier(numpuntos,f):
    fourier=[]
    for nn in range(numpuntos):
        cont=0.
        for k in range(len(f)):
            cont=cont+f[k]*np.exp(-1j*2.*np.pi*k*float(nn)/numpuntos)
        fourier.append(cont)
    return fourier
n=len(y) 
a=Fourier(n,y)
plt.figure
plt.xlabel("x")
plt.ylabel("T. de Fourier")
plt.grid()
plt.plot(x,a)
plt.savefig("ParisCamila_TF.pdf")
# HW3
