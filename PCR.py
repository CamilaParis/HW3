#PCR
import numpy as np
import matplotlib.pylab as plt
#import urllib
import urllib.request

urllib.request.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat", "WDBC.dat")
#response = urllib.urlretrieve("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat","WDBC.dat")
data=np.genfromtxt("WDBC.dat", delimiter=",", replace_space='_',dtype="unicode")
datn=np.genfromtxt("WDBC.dat", delimiter=",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

for i in range(len(data[0,:])):
    for j in range(len(data[:,0])):
        if(data[j,i]=="M"):
            data[j,i]=1
        elif(data[j,i]=="B"):
            data[j,i]=2

#for i in range(len(v)):
#v[i]=v[i].astype(float)
for i in range(len(datn)):
    datn[i]=(datn[i]-np.mean(datn[i]))/(np.sqrt(np.var(datn[i])))

m=np.zeros((30,30))
for i in range(30):
    for j in range(30):
        m[i,j]=(np.sum((datn[:,i]-np.average(datn[:,i]))*(datn[:,j]-np.average(datn[:,j]))))/(568)

matriz_cov = m
val, vector=np.linalg.eig(matriz_cov)
print(val)
for i in range(len(val)):
    ap=val[i]
    bp=vector[i]
    print("El valor propio es", ap, "y su autovector es", bp)

v1=vector[:,0]
v2=vector[:,1]

print(np.shape(v1))
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
plt.scatter(p1r,p2r, c="r")
plt.scatter(p1b,p2b, c="b")

plt.show()

print (datn)
# HW3

