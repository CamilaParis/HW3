#PCR
#Ejercicio3
# Los arrays `u` y `v` representan dos series en funcion del tiempo `t`.
# Grafique las dos series de datos en una misma grafica y guardela sin mostrarla en 'serie.pdf'
# Calcule la covarianza entre `u` y `v` e imprima su valor.

import numpy as np
import matplotlib.pylab as plt

data=np.genfromtxt("WDBC.dat", delimiter=",", replace_space='_',dtype="unicode")

# HW3
