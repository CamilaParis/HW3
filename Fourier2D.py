#Fourier2D

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq

photo = plt.imread("arbol.png")
f = np.fft.fft2(photo)
plt.imshow(abs(f), cmap="gray")
plt.show()
#plt.savefig("ParisCamila_FT2D.pdf")
#print(np.shape(photo))
# HW3
