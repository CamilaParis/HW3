#Fourier2D

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq

photo = plt.imread("arbol.png")
f = np.fft.fft2(photo)
fs= np.fft.fftshift(f)
f1=np.fft.fftfreq(len(fs[0]))
f2=np.fft.fftfreq(len(fs[1]))
plt.imshow(np.log10(abs(fs)), cmap="gray")
plt.colorbar()
plt.show()
plt.plot(f1, abs(fs))
plt.plot(f2, abs(fs))
plt.show()
cop=np.copy(fs)

for i in range(np.shape(fs)[0]):
	for j in range(np.shape(fs)[1]):
		if(abs(cop[i,j])>4100 and abs(cop[i,j])<5000):
			cop[i,j]=0.0
plt.plot(f1, abs(cop))
plt.plot(f2, abs(cop))
plt.show()
img=np.fft.ifft2(cop)
plt.imshow(abs(img),cmap="gray")
plt.show()
#plt.savefig("ParisCamila_FT2D.pdf")
#print(np.shape(photo))
# HW3
