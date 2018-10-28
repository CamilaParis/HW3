#Fourier2D

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq

photo = plt.imread("arbol.png")
x=photo[:,0]
y=photo[:,1]
n=len(y) 
dt=x[1]-x[0]
fft_x = np.abs(fft(y))
freq = fftfreq(n, dt)
plt.figure()
plt.xlim(-0.75,0.75)
plt.plot(freq,fft_x)
plt.savefig("ParisCamila_FT2D.pdf")
# HW3
