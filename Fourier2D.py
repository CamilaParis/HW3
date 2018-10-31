#Fourier2D

import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from matplotlib.colors import LogNorm

#Almacene la imagen arbol.png en una arreglo de numpy.
photo = plt.imread("arbol.png")
#Usando los paquetes de scipy, realice la transformada de Fourier de la imagen.
f = np.fft.fft2(photo)
fs= np.fft.fftshift(f)
freq=np.fft.fftfreq(len(f[0]))
#Eligiendo una escala apropiada, haga una grafica de dicha transformada y guardela sin mostrarla en ApellidoNombre_FT2D.pdf.
plt.imshow(20*np.abs(fs), norm=LogNorm(vmin=1))
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Frecuencia")
plt.savefig("ParisCamila_FT2D.pdf")
plt.figure()
plt.plot(freq,(abs(fs)))
#plt.show()
#Haga un filtro que le permita eliminar el ruido periodico de la imagen. Para esto haga pruebas de como debe modificar la transformada de Fourier. Despues de las pruebas un rango de 4100 a 5000 en el filtro parece ser adecuado.
for i in range(np.shape(f)[0]):
	for j in range(np.shape(f)[0]):
		if(abs(fs[i,j])>4100 and abs(fs[i,j])<5000):
			fs[i,j]=0.0
#Grafique la transformada de Fourier despues del proceso de filtrado, esta vez en escala LogNorm y guarde dicha grafica sin mostrarla en ApellidoNombre_FT2D_filtrada.pdf.
plt.imshow(20*np.abs(fs), norm=LogNorm(vmin=1))
plt.colorbar()
plt.xlabel("Frecuencia")
plt.ylabel("Frecuencia")
plt.savefig("ParisCamila_FT2D_filtrada.pdf")
plt.figure()
plt.plot(freq, abs(fs))
#plt.show()
#Haga la transformada de Fourier inversa y grafique la imagen filtrada. Verifique que su filtro elimina el ruido periodico y guarde dicha imagen sin mostrarla en ApellidoNombre_Imagen_filtrada.pdf
img=np.fft.ifft2(fs)
plt.imshow(abs(img),cmap="gray")
plt.savefig("ParisCamila_Imagen_filtrada.pdf")

# HW3

