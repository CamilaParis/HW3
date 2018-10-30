#Fourier
import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d

#Almacene los datos de signal.dat y de incompletos.dat
data_signal=np.genfromtxt("signal.dat", delimiter=",")
data_inc=np.genfromtxt("incompletos.dat", delimiter=",")
#Haga una grafica de los datos de signal.dat y guarde dicha grafica sin mostrarla en ApellidoNombre_signal.pdf.
x=data_signal[:,0]
y=data_signal[:,1]
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x,y,c="g")
plt.savefig("ParisCamila_signal.pdf")
plt.close()
#Haga la transformada de Fourier de los datos de la senal usando su implementacion propia de la transformada discreta de fourier.
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
gh=Fourier(n,y)
dt=x[1]-x[0]
freq = fftfreq(n, dt)
#Haga una grafica de la transformada de Fourier y guarde dicha grafica sin mostrarla en ApellidoNombre_TF.pdf. Esta grafica debe ser en funcion de las frecuencias
plt.figure
plt.xlabel("Frecuencia")
plt.ylabel("T. de Fourier")
plt.grid()
plt.xlim(-1000,1000)
plt.plot(freq,np.abs(fourierT),c="blue")
plt.savefig("ParisCamila_TF.pdf")
plt.close()
#Imprima un mensaje donde indique cuales son las frecuencias principales de su senal.
print("Las frecuencias principales de la señal corresponden a la frecuencia de los dos picos cercanos a cero (ambos picos son iguales pero uno se encuantra en el eje positivo y el otro en el negativo de las frecuencias)")
#Haga un filtro pasa bajos con frecuencia de corte fc = 1000Hz. realice la transformada inversa y haga una grafica de la senal filtrada. Guarde dicha grafica sin mostrarla en ApellidoNombre_filtrada.pdf.
def pasaBajosm(d,f):
    for i in range(len(d)):
        if(abs(d[i])>1000):
            f[i]=0.0
    return f
def pasaBajosq(d,f):
    for i in range(len(d)):
        if(abs(d[i])>500):
            f[i]=0.0
    return f
plt.figure()
fourierFilt=pasaBajosm(freq,fourierT)
fourierInv=np.fft.ifft(fourierFilt)
plt.plot(x,fourierInv,c="purple")
plt.savefig("ParisCamila_filtrada.pdf")
##############################Escriba un mensaje en la terminal explicando por que no puede hacer la transformada de Fourier de los datos de incompletos.dat

x1=data_inc[:,0]
y1=data_inc[:,1]
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(x1,y1,c="g")
plt.show()
#Haga una interpolacion cuadratica y una cubica de sus datos incompletos.dat con 512 puntos. Haga la trasformada de Fourier de cada una de las series de datos interpoladas.
f1 = interp1d(x1, y1, kind="quadratic")
f2 = interp1d(x1, y1, kind="cubic")
xl = np.linspace(0.00039063,0.02851562,512)
fourier1=Fourier(n,f1(xl))
fourier2=Fourier(n,f2(xl))
dt2=xl[1]-xl[0]
freq2 = fftfreq(n, dt2)
#Haga una grafica con tres subplots de las tres transformada de Fourier (datos de signal.dat y datos interpolados) y guardela sin mostrarla en ApellidoNombre_TF_interpola.pdf.
plt.figure
plt.xlabel("Frecuencia")
plt.ylabel("T. de Fourier")
plt.grid()
plt.subplot(311)
plt.plot(freq,np.abs(gh),c="purple")
plt.subplot(312)
plt.plot(freq2,np.abs(fourier1),c="magenta")
plt.subplot(313)
plt.plot(freq2,np.abs(fourier2),c="cyan")
plt.savefig("ParisCamila_TF_interpola.pdf")
#################################Imprima un mensaje donde describa las diferencias encontradas entre la transformada de Fourier de la senal original y las de las interpolaciones.
#Aplique el filtro pasabajos con una frecuencia de corte fc = 1000Hz y con una frecuencia de corte de fc = 500Hz.
filt1m=pasaBajosm(freq,fourierT)
filt2m=pasaBajosm(freq2,fourier1)
filt3m=pasaBajosm(freq2,fourier2)
filt1q=pasaBajosq(freq,fourierT)
filt2q=pasaBajosq(freq2,fourier1)
filt3q=pasaBajosq(freq2,fourier2)
inv1m=np.fft.ifft(filt1m)
inv2m=np.fft.ifft(filt2m)
inv3m=np.fft.ifft(filt3m)
inv1q=np.fft.ifft(filt1q)
inv2q=np.fft.ifft(filt2q)
inv3q=np.fft.ifft(filt3q)
#Haga una grafica con dos subplots (uno para cada filtro) de las 3 senales filtradas y guardela sin mostrarla en ApellidoNombre_2Filtros.pdf.
plt.subplot(211)
plt.plot(x,inv1m,c="red")
plt.plot(xl,inv2m,c="cyan")
plt.plot(xl,inv3m,c="b")
plt.subplot(212)
plt.plot(x,inv1q,c="red")
plt.plot(xl,inv2q,c="cyan")
plt.plot(xl,inv3q,c="b")
plt.savefig("ParisCamila_2Filtros.pdf")
# HW3



