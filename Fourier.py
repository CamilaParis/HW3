#Fourier
import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d

data_signal=np.genfromtxt("signal.dat", delimiter=",")
x=data_signal[:,0]
y=data_signal[:,1]
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x,y,c="g")
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
fourierT=Fourier(n,y)
dt=x[1]-x[0]
freq = fftfreq(n, dt)
plt.figure
plt.xlabel("Frecuencia")
plt.ylabel("T. de Fourier")
plt.grid()
plt.xlim(-1000,1000)
plt.plot(freq,np.abs(fourierT),c="blue")
plt.savefig("ParisCamila_TF.pdf")
plt.close()

def pasaBajos(d,f):
    for i in range(len(d)):
        if(abs(d[i])>1000):
            f[i]=0.0
    return f
plt.figure()
fourierFilt=pasaBajos(freq,fourierT)
fourierInv=np.fft.ifft(fourierT)
plt.plot(x,fourierInv,c="purple")
plt.savefig("ParisCamila_filtrada.pdf")

data_inc=np.genfromtxt("incompletos.dat", delimiter=",")
x1=data_inc[:,0]
y1=data_inc[:,1]
f1 = interp1d(x1, y1, kind="quadratic")
f2 = interp1d(x1, y1, kind="cubic")
xl = np.linspace(0.00039063,0.02851562,512)
fourier1=Fourier(n,f1(xl))
fourier2=Fourier(n,f2(xl))
dt2=xl[1]-xl[0]
freq2 = fftfreq(n, dt2)
plt.figure
plt.xlabel("Frecuencia")
plt.ylabel("T. de Fourier")
plt.grid()
plt.plot(freq,np.abs(fourierT),c="purple")
plt.scatter(freq2,np.abs(fourier1),c="magenta")
plt.scatter(freq2,np.abs(fourier2),c="cyan")
plt.savefig("ParisCamila_TF_interpola.pdf")

# HW3

