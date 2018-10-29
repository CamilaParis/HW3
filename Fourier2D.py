#Fourier2D

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from matplotlib.colors import LogNorm

photo = plt.imread("arbol.png")
f = np.fft.fft2(photo)
fs= np.fft.fftshift(f)
freq=np.fft.fftfreq(len(f[0]))
plt.imshow((abs(f)), cmap="gray")
plt.colorbar()
plt.show()
plt.plot(freq, abs(f))
plt.show()

for i in range(np.shape(f)[0]):
	for j in range(np.shape(f)[0]):
		if(abs(fs[i,j])>4100 and abs(fs[i,j])<5000):
			fs[i,j]=0.0
plt.imshow(np.abs(fs), norm=LogNorm(vmin=5))
plt.show()
plt.plot(freq, abs(fs))
plt.show()
img=np.fft.ifft2(fs)
plt.imshow(abs(img),cmap="gray")
plt.show()
#plt.savefig("ParisCamila_FT2D.pdf")
#print(np.shape(photo))
# HW3

